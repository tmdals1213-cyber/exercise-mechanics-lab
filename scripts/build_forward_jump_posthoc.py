"""Prescribed small-forward jump, conservative free flight, no target box.

Ground posture is fitted to material foot frames and a declared COM path.
Force sharing and apparatus placement follow only after the mechanics audit.
This is conditional inverse dynamics, not a maximum-capability prediction.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,mujoco,quadprog
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from scipy.optimize import least_squares,brentq
from scipy.integrate import trapezoid
from retarget_measured_jump import MeasuredJumpRig
from centroidal_jump import Polynomial,s5
from mechanics_engine_adapter import sample

ROOT=Path(__file__).resolve().parents[1];G=9.80665

def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])

def right_jacobian(v,inverse=False):
    th=np.linalg.norm(v);S=skew(v)
    if inverse:
        c=1/12 if th<1e-5 else 1/th**2-(1+np.cos(th))/(2*th*np.sin(th))
        return np.eye(3)+.5*S+c*S@S
    if th<1e-5:return np.eye(3)-.5*S+S@S/6
    return np.eye(3)-(1-np.cos(th))/th**2*S+(th-np.sin(th))/th**3*S@S

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--launch-vz',type=float,default=3.2);ap.add_argument('--forward-speed',type=float,default=.85)
    ap.add_argument('--height-m',type=float,default=1.78);ap.add_argument('--mass-kg',type=float,default=87.)
    ap.add_argument('--hz',type=int,default=250);ap.add_argument('--ik-hz',type=int,default=125);ap.add_argument('--label',default='forward_v320')
    ap.add_argument('--toe-roll',type=float,default=.5)
    ap.add_argument('--heel-rise-fraction',type=float,default=.2)
    ap.add_argument('--tip-roll',type=float,default=.45)
    ap.add_argument('--tip-rate',type=float,default=4.0);ap.add_argument('--toe-rate',type=float,default=1.5)
    ap.add_argument('--ankle-end-margin',type=float,default=.02)
    ap.add_argument('--momentum-ground',action='store_true')
    ap.add_argument('--dynamics-ground',action='store_true')
    ap.add_argument('--native-ground',action='store_true')
    ap.add_argument('--push-duration',type=float,default=.26)
    ap.add_argument('--balance-weight',type=float,default=1e7)
    ap.add_argument('--posture-acceleration-scale',type=float,default=1.)
    ap.add_argument('--propulsion-profile',choices=['quintic','half-sine'],default='quintic')
    ap.add_argument('--deep-posture',action='store_true')
    ap.add_argument('--arm-counter-swing',action='store_true')
    ap.add_argument('--align-air-feet',action='store_true')
    ap.add_argument('--tuck-fraction',type=float,default=.88)
    ap.add_argument('--free-arms',action='store_true')
    ap.add_argument('--free-tip-pitch',action='store_true')
    ap.add_argument('--crouch-scale',type=float,default=1.)
    ap.add_argument('--forward-lean-m',type=float,default=0.)
    ap.add_argument('--takeoff-drop-m',type=float,default=.05);ap.add_argument('--reuse-ground',action='store_true')
    ap.add_argument('--frozen-ground',action='store_true',help='Replay an unchanged cached ground artifact with its original solver provenance; re-audit all mechanics.')
    a=ap.parse_args();out=ROOT/'results/power_comparison/yun_sungbin'/a.label;out.mkdir(parents=True,exist_ok=True)
    integrated_ground=a.dynamics_ground or a.native_ground
    src=ROOT/'results/power_comparison/cmj_analysis';z=np.load(src/'motion.npz');old=json.loads((src/'setup.json').read_text())
    r=MeasuredJumpRig(a.mass_kg,a.height_m);b,m,d=r.b,r.m,r.d;mass=float(m.body_mass.sum());W=mass*G;scale=a.height_m/1.8
    raw_cs=CubicSpline(z['times'],z['com']*scale);original=CubicSpline(z['times'],z['qpos'][:,r.ids])
    oldrot=RotationSpline(z['times'],Rotation.from_quat(np.roll(z['qpos'][:,3:7],-1,axis=1)))
    bottom=float(old['countermovement_end_s']);old_to=float(old['takeoff_s']);vz=a.launch_vz
    quiet_z=float(np.mean(z['com'][z['times']<=.25,2]))*scale
    def cs(t,derivative=0):
        value=raw_cs(t,derivative)
        if t<=bottom+1e-9:value[2]=a.crouch_scale*value[2]+(1-a.crouch_scale)*quiet_z*(derivative==0)
        duration=bottom-.25;u=np.clip((t-.25)/duration,0,1)
        shift=[u**3*(10-15*u+6*u*u),30*u*u*(1-u)**2/duration,60*u*(1-u)*(1-2*u)/duration**2]
        if derivative<3:value[1]-=a.forward_lean_m*shift[derivative]
        return value
    grid=np.linspace(0,1,1001);shape=np.interp(bottom+(old_to-bottom)*grid,z['times'],z['force_world_N'][:,2]);shape[-1]=0
    end_COM=cs(old_to);end_COM[2]-=a.takeoff_drop_m
    ratio=trapezoid((1-grid)*shape,grid)/trapezoid(shape,grid);stroke=float(end_COM[2]-cs(bottom)[2])
    roots=np.roots([G*(ratio-.5),ratio*vz,-stroke]);T=float(min(x.real for x in roots if abs(x.imag)<1e-8 and x.real>0))
    a0=float(cs(bottom,2)[2]);v0=float(cs(bottom,1)[2])
    if a.propulsion_profile=='half-sine':
        roots=np.roots([(a0+G)/12,.5*(vz+v0),-stroke]);T=float(min(x.real for x in roots if abs(x.imag)<1e-8 and x.real>0))
    if a.native_ground:T=a.push_duration
    pulse=np.pi/2*((vz-v0)/T-(a0-G)/2)
    to=bottom+T;k1=vz/float(cs(old_to,1)[2]);k2=(-G-float(cs(old_to,2)[2])*k1*k1)/float(cs(old_to,1)[2])
    clock=Polynomial(bottom,1,0,old_to,k1,k2,T)
    cp=Polynomial(cs(bottom),cs(bottom,1),cs(bottom,2),end_COM,np.r_[cs(old_to,1)[:2]*k1,vz],[0,0,-G],T)
    def propulsion(t,k=0):
        p=cp.value(t,k)
        if a.propulsion_profile=='half-sine':
            u=t/T
            if k==0:p[2]=cs(bottom)[2]+v0*t+a0*T*T*(u*u/2-u**3/6)+pulse*T*T*(u/np.pi-np.sin(np.pi*u)/np.pi**2)-G*T*T*u**3/6
            elif k==1:p[2]=v0+a0*T*(u-u*u/2)+pulse*T*(1-np.cos(np.pi*u))/np.pi-G*T*u*u/2
            elif k==2:p[2]=a0*(1-u)+pulse*np.sin(np.pi*u)-G*u
        return p
    toe=bottom+a.heel_rise_fraction*T
    tip=toe+.4*(to-toe)
    if a.native_ground:toe=bottom+.5*T;tip=to+1.
    toe_curve=Polynomial(0,0,0,-a.toe_roll,-a.toe_rate,0,to-toe)
    tip_curve=Polynomial(0,0,0,a.tip_roll,a.tip_rate,0,to-tip)
    assert min(float(clock.value(t,1)[0])for t in np.linspace(0,T,100))>0,'Nonmonotone propulsion clock'
    # Horizontal force direction is blended into a constant fraction of vertical
    # GRF. It vanishes with the vertical reaction at takeoff, avoiding an
    # impossible finite horizontal force as the normal force disappears.
    tt=np.linspace(0,T,2001);fz_acc=np.array([propulsion(x,2)[2]+G for x in tt]);blend=np.array([s5(x/min(.04,T/3))for x in tt])
    c0=float(cs(bottom,2)[1]/fz_acc[0]);den=trapezoid(fz_acc*blend,tt)
    c1=(-a.forward_speed-cs(bottom,1)[1]-trapezoid(fz_acc*(1-blend)*c0,tt))/den
    assert max(abs(c0),abs(c1))<.75,('Requested horizontal reaction exceeds friction margin',c0,c1)
    ay=CubicSpline(tt,fz_acc*(c0*(1-blend)+c1*blend));iy=ay.antiderivative();iiy=iy.antiderivative()
    def ground_com(t):
        if t<=bottom:return cs(t)
        u=t-bottom;p=propulsion(u);p[1]=cs(bottom)[1]+cs(bottom,1)[1]*u+iiy(u)-iiy(0)-iy(0)*u;return p
    def ground_velocity(t):
        if t<=bottom:return cs(t,1)
        u=t-bottom;p=propulsion(u,1);p[1]=cs(bottom,1)[1]+iy(u)-iy(0);return p
    def ground_acceleration(t):
        if t<=bottom:return cs(t,2)
        u=t-bottom;p=propulsion(u,2);p[1]=ay(u);return p
    def ground_source_time(t):return t if t<=bottom else float(clock.value(t-bottom)[0])
    def template(t):
        tau=ground_source_time(t);q=b.q0.copy();q[3:7]=np.roll(oldrot(tau).as_quat(),1);q[r.ids]=original(tau)
        if a.deep_posture:
            bend=s5((t-.25)/(bottom-.25))if t<=bottom else 1-s5((tau-bottom)/(old_to-bottom))
            reference_low=original(bottom)
            for side in ['r','l']:
                for stem,angle in [('hip_flexion',1.4),('knee_angle',1.9)]:
                    i=b.ind_names.index(stem+'_'+side);q[r.ids[i]]+=bend*(angle-reference_low[i])
        # Toe roll belongs to the ground phase; the toes stay on the same
        # material support frame while the heel is allowed to rise.
        roll=0. if t<=toe else float(toe_curve.value(t-toe)[0])
        for side in ['r','l']:q[m.jnt_qposadr[b.joints['mtp_angle_'+side]]]=roll
        return b.complete(q)
    q0=template(0);q0[:3]=0;b.pose(q0);q0[:3]=ground_com(0)-m.body_mass@d.xipos/mass;b.pose(q0)
    foot_frames={};points=[]
    for side in ['r','l']:
        heel=m.body('calcn_'+side).id;tb=m.body('toes_'+side).id
        foot_frames[side]=[(bid,d.xpos[bid].copy(),d.xmat[bid].reshape(3,3).copy())for bid in [heel,tb]]
        sole=d.site_xpos[m.site('subject_sole_reference_'+side).id].copy();ball=d.xanchor[b.joints['mtp_angle_'+side]].copy()
        for bid,yy,fore in [(heel,sole[1]+.01*scale,False),(tb,ball[1],True),(tb,sole[1]-.245*scale,True)]:
            for dx in [-.037,.037]:
                p=np.array([sole[0]+dx*scale,yy,0.]);local=d.xmat[bid].reshape(3,3).T@(p-d.xpos[bid]);points.append((bid,local,p,fore,side))
    chosen=[i for i,n in enumerate(b.ind_names)if n in ['flex_extension','lat_bending','axial_rotation']or n.startswith(('hip_','knee_','ankle_','subtalar_'))]
    qids=np.array(r.ids)[chosen];jids=np.array(b.independent)[chosen]
    lo=np.r_[np.full(3,-.8),m.jnt_range[jids,0]+1e-6];hi=np.r_[np.full(3,.8),m.jnt_range[jids,1]-1e-6]
    for j,i in enumerate(chosen):
        if b.ind_names[i].startswith('ankle_angle_'):lo[3+j]+=a.ankle_end_margin
    knots=np.unique(np.round(np.r_[np.arange(0,to,1/a.ik_hz),bottom,toe,tip,to],10));qs=[];fits=[];last=None;jac_error=0.
    cache_args={k:v for k,v in vars(a).items()if k not in ['hz','label','reuse_ground','frozen_ground']}
    dependency='native_jump_ground_path.py'if a.native_ground else'dynamic_ground_path.py'if a.dynamics_ground else'momentum_ground_path.py'
    signature=json.dumps(dict(parameters=cache_args,ground_policy_version=7,solver_sha256=hashlib.sha256((ROOT/'scripts'/dependency).read_bytes()+(ROOT/'scripts/build_contact_jump.py').read_bytes()).hexdigest(),source_motion_sha256=hashlib.sha256((src/'motion.npz').read_bytes()).hexdigest()),sort_keys=True)
    if a.reuse_ground and (out/'ground_kinematics.npz').exists():
        cached=np.load(out/'ground_kinematics.npz');cached_signature=str(cached['signature'])
        if a.frozen_ground:
            prior=json.loads(cached_signature);current=json.loads(signature)
            air_only={'arm_counter_swing','align_air_feet','tuck_fraction'}
            old_inputs={k:v for k,v in prior['parameters'].items()if k not in air_only}
            new_inputs={k:v for k,v in current['parameters'].items()if k not in air_only}
            assert all(new_inputs.get(k)==v for k,v in old_inputs.items()),'Frozen-ground inputs differ'
            assert all(v==ap.get_default(k)for k,v in new_inputs.items()if k not in old_inputs),'New nondefault ground inputs cannot reuse the frozen artifact'
            assert prior['source_motion_sha256']==current['source_motion_sha256'],'Frozen-ground source changed'
            signature=cached_signature
            print('FROZEN_GROUND_REPLAY',hashlib.sha256((out/'ground_kinematics.npz').read_bytes()).hexdigest(),flush=True)
        else:assert cached_signature==signature,'Ground-cache settings differ'
        cached_times,keep=np.unique(np.round(cached['times'],10),return_index=True)
        if a.momentum_ground or integrated_ground:knots=cached_times
        else:assert np.allclose(cached_times,knots)
        qs=list(cached['qpos'][keep]);jac_error=float(cached['jacobian_directional_error'])
    if a.momentum_ground and not len(qs):
        from momentum_ground_path import solve
        knots,qs=solve(r,template,ground_com,ground_velocity,ground_acceleration,foot_frames,points,bottom,toe,tip,to,tip_curve,out,free_arms=a.free_arms,free_tip_pitch=a.free_tip_pitch)
    if a.dynamics_ground and not len(qs):
        from dynamic_ground_path import solve
        knots,qs,dynamic_report=solve(r,template,ground_com,ground_velocity,ground_acceleration,points,bottom,toe,tip,to,out,balance_weight=a.balance_weight,reference_acceleration_scale=a.posture_acceleration_scale)
    if a.native_ground and not len(qs):
        from native_jump_ground_path import solve
        knots,qs,dynamic_report=solve(r,q0,foot_frames,ground_com(0),bottom,toe,to,out)
    if integrated_ground:
        dynamic_report=json.loads((out/'dynamic_ground_summary.json').read_text());to=float(dynamic_report['actual_takeoff_s'])
    work_knots=[]if len(qs)else knots
    for k,t in enumerate(work_knots):
        seed=template(float(t));R0=Rotation.from_quat(np.roll(seed[3:7],-1));prior=seed[qids].copy();C=ground_com(float(t));mode=1 if t>toe else 0
        weights=np.full(len(chosen),.15)
        for j,i in enumerate(chosen):
            if b.ind_names[i]=='flex_extension':
                weights[j]=10.
                if t>bottom:prior[j]=float(original(bottom)[i])
        targets=[foot_frames[s][mode]for s in ['r','l']]
        if t>tip:
            rolled=[];pitch=float(tip_curve.value(t-tip)[0]);A=Rotation.from_rotvec([pitch,0,0]).as_matrix()
            for side,(bid,P,Rf) in zip(['r','l'],targets):
                tip_center=np.mean([p[2]for j,p in enumerate(points)if p[4]==side and j%6>=4],axis=0)
                local=Rf.T@(tip_center-P);RR=A@Rf;rolled.append((bid,tip_center-RR@local,RR))
            targets=rolled
        def pose(x):
            q=seed.copy();q[3:7]=np.roll((R0*Rotation.from_rotvec(x[:3])).as_quat(),1);q[qids]=x[3:];q[:3]=0;q=b.complete(q)
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);q[:3]=C-m.body_mass@d.xipos/mass
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);return q
        def fun(x):
            pose(x);err=[]
            for bid,P,R0f in targets:
                err.extend(1000*(d.xpos[bid]-P));err.extend(50*Rotation.from_matrix(R0f.T@d.xmat[bid].reshape(3,3)).as_rotvec())
            return np.r_[err,weights*(x[3:]-prior),.25*x[:3]]
        def jac(x):
            q=pose(x);B,_=r.coordinates(q,np.zeros(r.n));columns=np.c_[B[:,3:6]@right_jacobian(x[:3]),B[:,6+np.array(chosen)]]
            Jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,Jc,int(m.jnt_bodyid[0]));columns[:3]-=Jc@columns
            rows=[]
            for bid,P,R0f in targets:
                J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jacBody(m,d,J,Jr,bid);R=d.xmat[bid].reshape(3,3)
                phi=Rotation.from_matrix(R0f.T@R).as_rotvec();rows.extend([1000*J@columns,50*right_jacobian(phi,True)@R.T@Jr@columns])
            return np.vstack([*rows,np.c_[np.zeros((len(prior),3)),np.diag(weights)],np.c_[.25*np.eye(3),np.zeros((3,len(prior)))]])
        x0=np.r_[np.zeros(3),prior] if last is None else np.r_[last[:3],last[3:]]
        x0=np.clip(x0,lo+1e-7,hi-1e-7)
        if k==0:
            direction=np.linspace(-.3,.4,len(x0));hh=1e-6;fd=(fun(x0+hh*direction)-fun(x0-hh*direction))/(2*hh)
            jac_error=float(np.max(abs(fd-jac(x0)@direction)));assert jac_error<.001,('IK Jacobian crosscheck',jac_error)
        opt=least_squares(fun,x0,jac=jac,bounds=(lo,hi),max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-9)
        error=fun(opt.x)[:12];q=pose(opt.x);last=opt.x.copy();qs.append(q)
        fit=[float(np.linalg.norm(error[j*6:j*6+3])/1000)for j in range(2)];fits.append(max(fit))
        if max(fit)>.001:
            (out/'IK_failure.json').write_text(json.dumps(dict(time_s=float(t),foot_position_errors_m=fit,optimizer=opt.message,root_rotation_correction_rad=opt.x[:3].tolist(),near_bounds=[dict(coordinate=b.ind_names[chosen[j]],value=float(opt.x[3+j]),lower=float(lo[3+j]),upper=float(hi[3+j]))for j in range(len(chosen))if min(opt.x[3+j]-lo[3+j],hi[3+j]-opt.x[3+j])<.002],parameters=vars(a)),indent=2));raise RuntimeError(('Foot frame IK failed',t,fit))
        if k%25==0:print('FORWARD_GROUND_IK',k,len(knots),round(float(t),3),'foot_mm',round(1000*max(fit),5),flush=True)
    qs=np.array(qs);np.savez_compressed(out/'ground_kinematics.npz',times=knots,qpos=qs,parameters=json.dumps(vars(a)),signature=signature,jacobian_directional_error=jac_error)
    ground_joints=CubicSpline(knots,qs[:,r.ids]);ground_rot=RotationSpline(knots,Rotation.from_quat(np.roll(qs[:,3:7],-1,axis=1)))
    ground_root=CubicSpline(knots,qs[:,:3])
    def ground_q(t):
        q=b.q0.copy();q[r.ids]=ground_joints(t);q[3:7]=np.roll(ground_rot(t).as_quat(),1);q[:3]=0;q=b.complete(q)
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);q[:3]=ground_root(t) if integrated_ground else ground_com(t)-m.body_mass@d.xipos/mass;return q
    eps=5e-5
    def differential(t,config):
        q=config(t);v=np.zeros(m.nv);vm=np.zeros(m.nv);vp=np.zeros(m.nv)
        mujoco.mj_differentiatePos(m,v,2*eps,config(t-eps),config(t+eps))
        mujoco.mj_differentiatePos(m,vm,2*eps,config(t-2*eps),q);mujoco.mj_differentiatePos(m,vp,2*eps,q,config(t+2*eps))
        af=(vp-vm)/(2*eps);u=np.r_[v[:6],v[r.dofs]];ud=np.r_[af[:6],af[r.dofs]];B,curve=r.coordinates(q,u)
        return q,B@u,B@ud+curve,u,ud
    if integrated_ground:
        integrated=np.load(out/'dynamic_ground_integrated.npz');native_u=CubicSpline(integrated['times'],integrated['u']);native_ud=CubicSpline(integrated['times'][:-1],integrated['udot'])
    def ground_state(t):
        if not integrated_ground:return differential(t,ground_q)
        j=int(np.argmin(abs(integrated['times']-t)))
        q=integrated['qpos'][j].copy()if abs(float(integrated['times'][j])-t)<1e-8 else ground_q(t)
        u=native_u(t);ud=native_ud(t);B,c=r.coordinates(q,u)
        return q,B@u,B@ud+c,u,ud
    qto,vto,ato,_,_=ground_state(to);b.pose(qto,vto,ato);mujoco.mj_subtreeVel(m,d)
    Cto=d.subtree_com[1].copy();Vto=d.subtree_linvel[1].copy();Hto=d.subtree_angmom[1].copy();angles=qto[r.ids];rates=vto[r.dofs]
    lower=m.jnt_range[b.independent,0].copy();upper=m.jnt_range[b.independent,1].copy()
    for i,name in enumerate(b.ind_names):
        if name.startswith('shoulder_elv_'):upper[i]=min(upper[i],1.85)
    margin=np.where(rates<0,angles-lower,upper-angles);brake=np.minimum(.06,1.6*np.maximum(margin,1e-5)/np.maximum(abs(rates),1e-6))
    brake=np.maximum(brake,.001);stopped=angles+.5*rates*brake
    assert np.all(stopped>=lower-1e-5)and np.all(stopped<=upper+1e-5),('Takeoff too near joint range',[(b.ind_names[i],angles[i],rates[i])for i in np.where((stopped<lower)|(stopped>upper))[0]])
    target=stopped.copy()
    for side in ['r','l']:
        for stem,value in [('hip_flexion',2.0),('knee_angle',2.05),('ankle_angle',.05),('mtp_angle',0.)]:
            target[b.ind_names.index(stem+'_'+side)]=value
        if a.arm_counter_swing:
            target[b.ind_names.index('shoulder_elv_'+side)]=.12
            target[b.ind_names.index('elbow_flexion_'+side)]=.4
    target[b.ind_names.index('flex_extension')]=-.15
    if a.align_air_feet:
        fits=[]
        for side in ['r','l']:
            names=['hip_adduction_'+side,'hip_rotation_'+side,'subtalar_angle_'+side];columns=[b.ind_names.index(n)for n in names]
            base=target.copy();bid=m.body('calcn_'+side).id;desired_R=foot_frames[side][0][2]
            def orientation_fit(x):
                joints=base.copy();joints[columns]=x;q=qto.copy();q[r.ids]=joints;q=b.complete(q);b.pose(q)
                err=Rotation.from_matrix(desired_R@d.xmat[bid].reshape(3,3).T).as_rotvec()
                return np.r_[20*err[1:],.05*(x-base[columns])]
            fit=least_squares(orientation_fit,np.clip(base[columns],lower[columns]+1e-6,upper[columns]-1e-6),bounds=(lower[columns]+1e-6,upper[columns]-1e-6),max_nfev=150)
            target[columns]=fit.x;fits.append(dict(side=side,coordinates=names,target_rad=fit.x.tolist(),roll_yaw_residual_rad=(orientation_fit(fit.x)[:2]/20).tolist()))
        (out/'air_pose_design.json').write_text(json.dumps(dict(method='Internal hip/subtalar coordination to align foot roll and yaw; no world-frame foot force or imposed pelvis attitude.',fits=fits),indent=2))
    tuck_T=max(float(Vto[2]/G)*a.tuck_fraction,.18);end=to+float(Vto[2]/G)+.18
    def air_angles(tau):
        u=tau/brake;first=angles+rates*brake*(u-2.5*u**4+3*u**5-u**6)
        p=np.clip((tau-brake)/(tuck_T-brake),0,1);second=stopped+(target-stopped)*(p**3*(10-15*p+6*p*p))
        return np.where(tau<brake,first,second)
    def air_angle_rate(tau):
        u=tau/brake;first=rates*(1-10*u**3+15*u**4-6*u**5)
        p=np.clip((tau-brake)/(tuck_T-brake),0,1);second=(target-stopped)/(tuck_T-brake)*(30*p*p*(1-p)**2)
        return np.where(tau<brake,first,second)
    def omega(t,quat):
        q=b.q0.copy();q[:3]=0;q[3:7]=quat/np.linalg.norm(quat);q[r.ids]=air_angles(t-to);q=b.complete(q)
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        u=np.r_[np.zeros(6),air_angle_rate(t-to)];B,_=r.coordinates(q,u);A=np.zeros((3,m.nv));mujoco.mj_angmomMat(m,d,A,1)
        return np.linalg.solve(A[:,3:6],Hto-A@(B@u))
    def qdot(t,q):
        w=omega(t,q);return .5*np.r_[-q[1:]@w,q[0]*w+np.cross(q[1:],w)]
    def integrate_step(t,q,h):
        k1=qdot(t,q);k2=qdot(t+h/2,q+h*k1/2);k3=qdot(t+h/2,q+h*k2/2);k4=qdot(t+h,q+h*k3)
        result=q+h*(k1+2*k2+2*k3+k4)/6;return result/np.linalg.norm(result)
    dense=np.linspace(to,end,int(np.ceil((end-to)*6000))+1);quats=[qto[3:7].copy()]
    for t,tn in zip(dense[:-1],dense[1:]):quats.append(integrate_step(t,quats[-1],tn-t))
    past=[];qq=qto[3:7].copy()
    for k in range(1,7):qq=integrate_step(to-(k-1)*.001,qq,-.001);past.append(qq.copy())
    rot=RotationSpline(np.r_[to-np.arange(6,0,-1)*.001,dense],Rotation.from_quat(np.roll(np.array(past[::-1]+quats),-1,axis=1)))
    def air_q(t):
        tau=t-to;q=b.q0.copy();q[r.ids]=air_angles(tau);q[3:7]=np.roll(rot(t).as_quat(),1);q[:3]=0;q=b.complete(q)
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);q[:3]=Cto+Vto*tau+.5*tau*tau*np.array([0,0,-G])-m.body_mass@d.xipos/mass;return q
    def air_velocity(t):
        q=air_q(t);u=np.r_[np.zeros(6),air_angle_rate(t-to)];B,_=r.coordinates(q,u);b.pose(q)
        A=np.zeros((3,m.nv));J=np.zeros_like(A);mujoco.mj_angmomMat(m,d,A,1);mujoco.mj_jacSubtreeCom(m,d,J,1)
        u[3:6]=np.linalg.solve(A[:,3:6],Hto-A@(B@u))
        u[:3]=Vto+np.array([0,0,-G])*(t-to)-J@(B@u)
        return q,u
    def air_state(t):
        q,u=air_velocity(t);hh=5e-6;ud=(air_velocity(t+hh)[1]-air_velocity(t-hh)[1])/(2*hh);B,c=r.coordinates(q,u)
        return q,B@u,B@ud+c,u,ud
    times=np.unique(np.round(np.r_[np.arange(0,end,1/a.hz),bottom,toe,to,end],10));arrays={k:[]for k in ['times','qpos','qvel','qacc','u','udot','body_xpos','body_xmat','com','velocity','angular_momentum','mode','contact_positions_m','material_contact_positions_m','contact_forces_N','force_world_N','root_residual','foot_contact_position_error_m','foot_contact_speed_m_s','COM_power_W','COM_power_total_W','segment_energy_rate_W']}
    fore=np.array([p[3]for p in points]);cross=0.;audit=set(np.linspace(0,len(times)-1,25).astype(int));contact_qp_exact=0
    for k,t in enumerate(times):
        flying=t>=to-1e-9;q,v,acc,u,ud=air_state(float(t))if flying else ground_state(float(t));b.pose(q,v,acc);mujoco.mj_subtreeVel(m,d)
        actual=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@loc for bid,loc,*_ in points]);P=actual.copy();P[:,2]=0.;F=np.zeros_like(P);ext=np.zeros(m.nv)
        own=np.zeros(m.nv);mujoco.mj_rne(m,d,1,own)
        mask=(np.arange(len(points))%6>=4)if t>tip else fore if t>toe else np.ones(len(points),bool)
        if integrated_ground:
            mask=fore.copy() if t>toe else np.ones(len(points),bool)
            for event in dynamic_report['release_events']:
                if t>=event['time_s']:mask[event['point_indices']]=False
        active=np.array([],int)if flying else np.flatnonzero(mask);jacs=[]
        if integrated_ground and 'contact_forces_N'in integrated and not flying:
            j=int(np.argmin(abs(integrated['times'][:-1]-t)));assert abs(float(integrated['times'][j])-t)<1e-8,'Contact replay needs a native integration sample'
            F=integrated['contact_forces_N'][j].copy();P=actual.copy()
            for j,point in enumerate(points):
                J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,P[j],point[0]);ext+=J.T@F[j]
            contact_qp_exact+=1
        elif len(active):
            for j in active:
                J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,P[j],points[j][0]);jacs.append(J)
            E=np.concatenate([J[:,:6].T for J in jacs],axis=1);rhs=own[:6]/W;es=np.maximum(np.linalg.norm(E,axis=1),.01);E/=es[:,None];rhs/=es
            inequalities=[]
            for j in range(len(active)):
                for x,y,zz in [(0,0,1)]+[(np.cos(p),np.sin(p),.8*np.cos(np.pi/16))for p in np.linspace(0,2*np.pi,16,endpoint=False)]:
                    row=np.zeros(3*len(active));row[j*3:j*3+3]=[x,y,zz];inequalities.append(row)
            try:
                sol=quadprog.solve_qp(np.eye(E.shape[1]),np.zeros(E.shape[1]),np.vstack([E,inequalities]).T,np.r_[rhs,np.zeros(len(inequalities))],meq=6)[0];contact_qp_exact+=1
            except ValueError:
                try:sol=quadprog.solve_qp(E[3:].T@E[3:]+1e-8*np.eye(E.shape[1]),E[3:].T@rhs[3:],np.vstack([E[:3],inequalities]).T,np.r_[rhs[:3],np.zeros(len(inequalities))],meq=3)[0]
                except ValueError:
                    (out/'contact_failure.json').write_text(json.dumps(dict(time_s=float(t),root_demand=own[:6].tolist(),parameters=vars(a)),indent=2));raise
            F[active]=sol.reshape(-1,3)*W
            for j,J in zip(active,jacs):ext+=J.T@F[j]
        residual=own[:6]-ext[:6];err=max((np.linalg.norm(actual[j]-points[j][2])for j in active),default=0.)
        speed=[]
        for j in active:
            J,_=b.jacobian(points[j][0],points[j][1]);speed.append(np.linalg.norm(J@v))
        contacts=[(p[0],P[j],F[j],np.zeros(3))for j,p in enumerate(points)]
        if k in audit:cross=max(cross,sample(r,q,v,acc,contacts)['crosscheck_max'])
        vals=dict(times=t,qpos=q,qvel=v,qacc=acc,u=u,udot=ud,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),mode=2 if flying else 1 if t>toe else 0,contact_positions_m=P,material_contact_positions_m=actual,contact_forces_N=F,force_world_N=F.sum(0),root_residual=residual,foot_contact_position_error_m=err,foot_contact_speed_m_s=max(speed,default=0.),COM_power_W=F[:,2].sum()*d.subtree_linvel[1,2],COM_power_total_W=F.sum(0)@d.subtree_linvel[1],segment_energy_rate_W=own@v)
        for key,value in vals.items():arrays[key].append(value)
        if k%125==0:print('FORWARD_DYNAMICS',k,len(times),round(float(t),3),'GRF',round(F[:,2].sum()),'root',round(float(abs(residual).max()),5),flush=True)
    data={k:np.array(v)for k,v in arrays.items()};np.savez_compressed(out/'motion.npz',**data);digest=hashlib.sha256((out/'motion.npz').read_bytes()).hexdigest()
    stance=data['mode']<2;air=~stance;prop=(times>=bottom)&stance;rf=np.linalg.norm(data['root_residual'][stance,:3],axis=1);rm=np.linalg.norm(data['root_residual'][stance,3:],axis=1)
    H=data['angular_momentum'][air];ta=times[air]-to;C=data['com'][air];expected=Cto+ta[:,None]*Vto+.5*ta[:,None]**2*np.array([0,0,-G])
    rom=max(0.,float(np.maximum(lower-data['qpos'][:,r.ids],data['qpos'][:,r.ids]-upper).max()));normal=data['contact_forces_N'][:,:,2]
    checks=dict(independent_NE_max_error_SI=cross,stance_force_RMS_BW=float(np.sqrt(np.mean(rf*rf))/W),stance_moment_RMS_BW_height=float(np.sqrt(np.mean(rm*rm))/(W*a.height_m)),stance_moment_max_BW_height=float(rm.max()/(W*a.height_m)),stance_moment_RMS_Nm=float(np.sqrt(np.mean(rm*rm))),stance_moment_max_Nm=float(rm.max()),foot_material_position_max_m=float(data['foot_contact_position_error_m'].max()),foot_material_speed_max_m_s=float(data['foot_contact_speed_m_s'].max()),ballistic_COM_max_error_m=float(np.linalg.norm(C-expected,axis=1).max()),flight_H_max_drift=float(np.linalg.norm(H-Hto,axis=1).max()),flight_root_max_SI=float(abs(data['root_residual'][air]).max()),ROM_excess_rad=rom,minimum_normal_N=float(normal.min()),friction_excess_N=float(np.max(np.linalg.norm(data['contact_forces_N'][:,:,:2],axis=2)-.8*normal)),foot_min_z_first_20ms_m=float(data['material_contact_positions_m'][(times>=to)&(times<=to+.02),:,2].min()),peak_GRF_N=float(data['force_world_N'][:,2].max()),peak_propulsive_COM_vertical_power_W=float(data['COM_power_W'][prop].max()),peak_propulsive_segment_energy_rate_W=float(data['segment_energy_rate_W'][prop].max()),actual_takeoff_velocity_m_s=Vto.tolist(),airborne_COM_rise_m=float(C[:,2].max()-Cto[2]),max_minimum_sole_height_m=float(data['material_contact_positions_m'][air,:,2].min(1).max()),exact_contact_wrench_sample_count=contact_qp_exact,stance_sample_count=int(stance.sum()))
    Rair=Rotation.from_quat(np.roll(data['qpos'][air,3:7],-1,axis=1));Rto=Rotation.from_quat(np.roll(qto[3:7],-1))
    checks['maximum_flight_pelvis_rotation_deg']=float(np.rad2deg((Rto.inv()*Rair).magnitude()).max())
    gates=dict(independent_newton_euler=cross<1e-5,force_balance=checks['stance_force_RMS_BW']<.01,rotation_RMS=checks['stance_moment_RMS_BW_height']<.05,rotation_peak=checks['stance_moment_max_BW_height']<.1,foot_position=checks['foot_material_position_max_m']<.002,foot_speed=checks['foot_material_speed_max_m_s']<.02,ballistic=checks['ballistic_COM_max_error_m']<.001,angular_momentum=checks['flight_H_max_drift']<.1,flight_root=checks['flight_root_max_SI']<.1,ROM=rom<.002,unilateral=checks['minimum_normal_N']>-.001,friction=checks['friction_excess_N']<.001,separating_feet=checks['foot_min_z_first_20ms_m']>-.002,no_somersault=checks['maximum_flight_pelvis_rotation_deg']<90)
    limitations=['Prescribed COM and generic posture family, not a prediction of maximum athlete capability.','Residual rotational imbalance is disclosed; no external support torque is applied.','Foot frames are fitted during stance; sampled inverse-dynamics checks are not a full compliant-contact landing simulation.','No landing impact, stable box support or human muscle-force validation is claimed.']
    report=dict(status='prescribed_projected_contact_diagnostic',motion_sha256=digest,gates=gates,checks=checks,parameters=vars(a),ground_source_signature=json.loads(signature),derivatives='Integrated ground velocities/accelerations; exact ballistic and centroidal flight velocity, differentiated velocity for flight acceleration.'if integrated_ground else'Pose differentiation on ground; ballistic/centroidal flight derivatives.',limitations=limitations,exact_sticking_contact=False,force_residual_applied_as_external_support=False)
    setup=dict(profile=r.profile,rig_type='measured_cmj_standard',body_names=[m.body(i).name for i in range(m.nbody)],coordinate_names=b.coordinate_names,mass_kg=mass,height_m=a.height_m,physics_hz=a.hz,takeoff_s=to,countermovement_end_s=bottom,heel_rise_start_s=toe,tip_roll_start_s=tip,movement_onset_s=.348,ground_reaction_and_COP_are_inputs=False,human_motion_validated=False,scope='Small-forward jump with source-inspired arms, fitted flat-foot / forefoot / tip-edge support, prescribed propulsion and momentum-conserving tuck; no target box and no landing.')
    (out/'setup.json').write_text(json.dumps(setup,indent=2));(out/'verification.json').write_text(json.dumps(report,indent=2));(out/'motion_quality.json').write_text(json.dumps(dict(status='passed'if all(gates.values())else'failed',motion_sha256=digest,task='forward_jump_no_target_box',source_human_motion=False),indent=2))
    (out/'contact_definition.json').write_text(json.dumps(dict(points=[dict(body_id=p[0],body=m.body(p[0]).name,local_m=p[1].tolist(),world_reference_m=p[2].tolist(),forefoot=p[3],side=p[4])for p in points],mass_kg=mass,gravity_m_s2=G),indent=2))
    b.pose(b.q0);np.savez_compressed(out/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
    print('FORWARD_JUMP_AUDIT',json.dumps(report),flush=True)
    if not all(gates.values()):raise RuntimeError(('Forward jump audit rejected',gates))

if __name__=='__main__':main()
