"""Reference-shaped CMJ with a prescribed height, conservative free flight.

The source posture path is retained. A C2 propulsion clock/COM trajectory sets
the launch velocity. Ground wrenches are inverse-dynamics outputs, distributed
within projected support regions; this is not a measured new force-plate trial.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,mujoco,quadprog
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation,RotationSpline
from retarget_measured_jump import MeasuredJumpRig
from centroidal_jump import Polynomial

ROOT=Path(__file__).resolve().parents[1];G=9.80665

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--height-cm',type=float,default=30)
    ap.add_argument('--hz',type=int,default=250);ap.add_argument('--label',default='retimed_250hz')
    args=ap.parse_args();out=ROOT/'results/power_comparison/cmj_adult_target'/args.label;out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/power_comparison/cmj_analysis';z=np.load(source/'motion.npz');setup=json.loads((source/'setup.json').read_text())
    r=MeasuredJumpRig();b,m,d=r.b,r.m,r.d;mass=float(m.body_mass.sum());W=mass*G
    bottom=float(setup['countermovement_end_s']);old_to=float(setup['takeoff_s']);old_end=float(z['times'][-1])
    cs=CubicSpline(z['times'],z['com']);joint=CubicSpline(z['times'],z['qpos'][:,r.ids])
    oldrot=RotationSpline(z['times'],Rotation.from_quat(np.roll(z['qpos'][:,3:7],-1,axis=1)))
    velocity=np.sqrt(2*G*args.height_cm/100)
    u=np.linspace(0,1,1001);shape=np.interp(bottom+(old_to-bottom)*u,z['times'],z['force_world_N'][:,2]);shape[-1]=0
    A=trapezoid(shape,u);B=trapezoid((1-u)*shape,u);ratio=B/A;stroke=float(cs(old_to)[2]-cs(bottom)[2])
    roots=np.roots([G*(ratio-.5),ratio*velocity,-stroke]);T=float(min(v.real for v in roots if abs(v.imag)<1e-8 and v.real>0))
    takeoff=bottom+T;k1=velocity/float(cs(old_to,1)[2]);k2=(-G-float(cs(old_to,2)[2])*k1*k1)/float(cs(old_to,1)[2])
    clock=Polynomial(bottom,1,0,old_to,k1,k2,T)
    com_poly=Polynomial(cs(bottom),cs(bottom,1),cs(bottom,2),cs(old_to),
        np.r_[cs(old_to,1)[:2]*k1,velocity],[0,0,-G],T)
    flight_T=(old_end-old_to)*k1;end=takeoff+flight_T
    air_clock=Polynomial(old_to,k1,k2,old_end,1/k1,0,flight_T)
    assert min(float(clock.value(t,1)[0]) for t in np.linspace(0,T,100))>0
    assert min(float(air_clock.value(t,1)[0]) for t in np.linspace(0,flight_T,100))>0
    toe=brentq(lambda t:float(clock.value(t-bottom)[0])-float(setup['heel_rise_start_s']),bottom,takeoff)
    angular_correction=None;air_rot=None

    def source_time(t):
        if t<=bottom:return t,1.,0.
        p,ss=(clock,t-bottom) if t<=takeoff else (air_clock,t-takeoff)
        return tuple(float(p.value(ss,i)[0]) for i in range(3))
    def COM(t,stance=False):
        if t<=bottom:return cs(t)
        if t<=takeoff or stance:return com_poly.value(t-bottom)
        s=t-takeoff;return cs(old_to)+s*np.r_[cs(old_to,1)[:2]*k1,velocity]+.5*s*s*np.array([0,0,-G])
    def configuration(t,stance=False):
        tau,_,_=source_time(t) if not stance or t<=takeoff else tuple(float(clock.value(t-bottom,i)[0]) for i in range(3))
        if t>takeoff and not stance and air_rot is not None:R=air_rot(t)
        else:
            R=oldrot(tau)
            if t>bottom and angular_correction is not None:R=R*Rotation.from_rotvec(angular_correction.value(t-bottom))
        q=b.q0.copy();q[3:7]=np.roll(R.as_quat(),1);q[r.ids]=joint(tau);q[:3]=0;q=b.complete(q)
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        q[:3]=COM(t,stance)-m.body_mass@d.xipos/mass
        return q
    def state(t,stance=False):
        h=5e-5
        def vel(tt):
            v=np.zeros(m.nv);mujoco.mj_differentiatePos(m,v,2*h,configuration(tt-h,stance),configuration(tt+h,stance));return v
        q=configuration(t,stance);vf=vel(t);af=(vel(t+h)-vel(t-h))/(2*h)
        uu=np.r_[vf[:6],vf[r.dofs]];ud=np.r_[af[:6],af[r.dofs]];B,curve=r.coordinates(q,uu)
        return q,B@uu,B@ud+curve,uu,ud

    # End stance with the acceleration required by free flight, without a
    # position/velocity reset or an external angular-momentum correction.
    q,v,a,_,_=state(takeoff,True);b.pose(q,v,a);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M)
    trial=a.copy();trial[:6]=0;d.qacc[:]=trial;bias=np.zeros(m.nv);mujoco.mj_rne(m,d,1,bias)
    free=np.linalg.solve(M[:6,:6],-bias[:6])
    angular_correction=Polynomial(np.zeros(3),0,0,np.zeros(3),0,free[3:6]-a[3:6],T)
    qto,vto,ato,_,_=state(takeoff,True);b.pose(qto,vto,ato);mujoco.mj_subtreeVel(m,d);Hto=d.subtree_angmom[1].copy()
    qrot=qto[3:7].copy();rot_times=[takeoff];rot_values=[np.roll(qrot,-1)]

    def omega(t,quat):
        tau,rate,_=source_time(t);q=b.q0.copy();q[3:7]=quat/np.linalg.norm(quat);q[r.ids]=joint(tau);q[:3]=0;q=b.complete(q)
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        uu=np.r_[np.zeros(6),joint(tau,1)*rate];B,_=r.coordinates(q,uu);relative=B@uu
        mat=np.zeros((3,m.nv));mujoco.mj_angmomMat(m,d,mat,1)
        return np.linalg.solve(mat[:,3:6],Hto-mat@relative)
    def qdot(t,qq):
        w=omega(t,qq);a0=qq[0];av=qq[1:]
        return .5*np.r_[-av@w,a0*w+np.cross(av,w)]
    dense=np.linspace(takeoff,end,max(3,int(np.ceil(flight_T*1000))+1))
    for tt,tn in zip(dense[:-1],dense[1:]):
        h=tn-tt;k_1=qdot(tt,qrot);k_2=qdot(tt+h/2,qrot+h*k_1/2);k_3=qdot(tt+h/2,qrot+h*k_2/2);k_4=qdot(tn,qrot+h*k_3)
        qrot+=h*(k_1+2*k_2+2*k_3+k_4)/6;qrot/=np.linalg.norm(qrot)
        rot_times.append(tn);rot_values.append(np.roll(qrot,-1))
    # Ghost samples constrain the spline's launch-side angular derivative.
    # Otherwise an endpoint interpolation error is amplified by acceleration
    # probes that straddle takeoff, creating a false impulse in inverse dynamics.
    qb=qto[3:7].copy();past=[]
    for k in range(1,7):
        tt=takeoff-(k-1)*.001;h=-.001
        k_1=qdot(tt,qb);k_2=qdot(tt+h/2,qb+h*k_1/2);k_3=qdot(tt+h/2,qb+h*k_2/2);k_4=qdot(tt+h,qb+h*k_3)
        qb+=h*(k_1+2*k_2+2*k_3+k_4)/6;qb/=np.linalg.norm(qb);past.append((tt+h,np.roll(qb,-1).copy()))
    air_rot=RotationSpline([p[0] for p in past[::-1]]+rot_times,
        Rotation.from_quat([p[1] for p in past[::-1]]+rot_values))

    # Fixed projected support regions, as in force-plate inverse dynamics.
    # The geometry mismatch/slip is audited separately; no perfect sticking
    # contact or experimentally observed new jump is claimed.
    b.pose(configuration(0));points=[];bodies=[];fore=[];locals_=[]
    for side in ['r','l']:
        heel=m.body('calcn_'+side).id;toe_bid=m.body('toes_'+side).id
        origin=d.site_xpos[m.site('subject_sole_reference_'+side).id].copy();ball=d.xanchor[b.joints['mtp_angle_'+side]].copy()
        for bid,yy,ff in [(heel,origin[1]+.01,False),(toe_bid,ball[1],True),(toe_bid,origin[1]-.245,True)]:
            for xx in [-.037,.037]:
                p=np.array([origin[0]+xx,yy,0.]);points.append(p);bodies.append(bid);fore.append(ff);locals_.append(d.xmat[bid].reshape(3,3).T@(p-d.xpos[bid]))
    points=np.array(points);reference_points=points.copy();fore=np.array(fore);locals_=np.array(locals_)
    times=np.unique(np.r_[np.arange(0,end,1/args.hz),bottom,toe,takeoff,end])
    arrays={k:[] for k in ['times','qpos','qvel','qacc','u','udot','com','velocity','angular_momentum','mode','body_xpos','body_xmat','contact_positions_m','contact_forces_N','force_world_N','root_residual','material_contact_positions_m']}
    max_flight_root=0.
    for k,t in enumerate(times):
        q,v,a,uu,ud=state(float(t));b.pose(q,v,a);mujoco.mj_subtreeVel(m,d)
        material=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local in zip(bodies,locals_)])
        points=material.copy();points[:,2]=0.
        rne=np.zeros(m.nv);mujoco.mj_rne(m,d,1,rne);flight=t>=takeoff-1e-9
        F=np.zeros_like(points);external=np.zeros(m.nv)
        if not flight:
            active=np.flatnonzero(fore if t>=toe else np.ones(len(points),bool));jacs=[]
            for j in active:
                J=np.zeros((3,m.nv));Jr=np.zeros_like(J);mujoco.mj_jac(m,d,J,Jr,points[j],bodies[j]);jacs.append(J)
            E=np.concatenate([J[:,:6].T for J in jacs],axis=1);target=rne[:6]/W
            scale=np.maximum(np.linalg.norm(E,axis=1),.01);E/=scale[:,None];target/=scale
            inequalities=[]
            for j in range(len(active)):
                for x,y,zv in [(0,0,1)]+[(float(np.cos(p)),float(np.sin(p)),float(.8*np.cos(np.pi/16))) for p in np.linspace(0,2*np.pi,16,endpoint=False)]:
                    row=np.zeros(3*len(active));row[3*j:3*j+3]=[x,y,zv];inequalities.append(row)
            # Preserve translational balance exactly; disclose the best achievable
            # rotational balance rather than inventing an external support torque.
            # This diagnostic is not a strict forward-contact simulation.
            rot=E[3:];trg=target[3:]
            try:sol=quadprog.solve_qp(rot.T@rot+1e-8*np.eye(E.shape[1]),rot.T@trg,np.vstack([E[:3],inequalities]).T,np.r_[target[:3],np.zeros(len(inequalities))],meq=3)[0]
            except ValueError:
                (out/'contact_failure.json').write_text(json.dumps(dict(time_s=float(t),mode='toe' if t>=toe else 'plant',root_demand=rne[:6].tolist()),indent=2));raise
            F[active]=sol.reshape(-1,3)*W
            for j,J in zip(active,jacs):external+=J.T@F[j]
        residual=rne[:6]-external[:6]
        if flight:max_flight_root=max(max_flight_root,float(abs(residual).max()))
        vals=dict(times=t,qpos=q,qvel=v,qacc=a,u=uu,udot=ud,com=m.body_mass@d.xipos/mass,
            velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),mode=2 if flight else 1 if t>=toe else 0,
            body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),contact_positions_m=points,contact_forces_N=F,force_world_N=F.sum(0),root_residual=residual,material_contact_positions_m=material)
        for name,value in vals.items():arrays[name].append(value)
        if k%max(1,args.hz//2)==0:print('RETIMED_CMJ',round(float(t),3),'F',F.sum(0).round(1),'root',float(abs(residual).max()),flush=True)
    data={k:np.array(v) for k,v in arrays.items()};np.savez_compressed(out/'motion.npz',**data)
    report=dict(status='candidate_not_published',target_airborne_COM_height_m=args.height_cm/100,
        actual_airborne_COM_rise_m=float(data['com'][data['mode']==2,2].max()-cs(old_to)[2]),takeoff_s=takeoff,
        propulsion_duration_s=T,peak_GRF_N=float(data['force_world_N'][:,2].max()),peak_GRF_BW=float(data['force_world_N'][:,2].max()/W),
        root_residual_max_SI=float(abs(data['root_residual']).max()),flight_root_residual_max_SI=max_flight_root,
        mass_kg=mass,reference_motion_sha256=hashlib.sha256((source/'motion.npz').read_bytes()).hexdigest(),
        force_amplitudes_recomputed=True,new_experimental_force_plate_data=False,
        support_model='Projected rigid-foot support regions; prescribed reference-shaped motion, not forward contact simulation.')
    (out/'candidate_summary.json').write_text(json.dumps(report,indent=2))
    (out/'setup.json').write_text(json.dumps(dict(profile=r.profile,rig_type='measured_cmj_standard',physics_hz=args.hz,
        body_names=[m.body(i).name for i in range(m.nbody)],coordinate_names=b.coordinate_names,mass_kg=mass,height_m=1.8,
        takeoff_s=takeoff,countermovement_end_s=bottom,heel_rise_start_s=toe,movement_onset_s=.348,
        ground_reaction_and_COP_are_inputs=False,human_motion_validated=False,
        scope='Reference-shaped 30cm CMJ inverse dynamics; no landing. Foot support regions prescribed; force distribution solved.'),indent=2))
    (out/'contact_definition.json').write_text(json.dumps(dict(points=[dict(body_id=bid,body=m.body(bid).name,
        world_reference_m=p.tolist(),local_m=local.tolist(),forefoot=bool(f)) for bid,p,f,local in zip(bodies,reference_points,fore,locals_)],mass_kg=mass,gravity_m_s2=G,ground_z_m=0),indent=2))
    print('RETIMED_CMJ_CANDIDATE',json.dumps(report),flush=True)

if __name__=='__main__':main()
