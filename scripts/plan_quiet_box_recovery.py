"""Define a quiet supported recovery, then verify its inverse contact mechanics.

COM height is an output of the coordinated posture, not an incompatible second
controller. Feet and upper-body paths are continuous from the actual arrival.
No root force or reaction clipping rescues a failed contact-wrench solution.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco,quadprog
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from scipy.linalg import qr
from retarget_measured_jump import MeasuredJumpRig
from high_flex_profile import apply
from build_contact_jump import contact_geometry
from centroidal_jump import Polynomial
from box_geometry_audit import segments,minimum_clearance
from build_forward_jump_posthoc import right_jacobian
ROOT=Path(__file__).resolve().parents[1];G=9.80665


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='quiet_recovery_v1');ap.add_argument('--hz',type=int,default=1000);ap.add_argument('--duration',type=float,default=2.2);ap.add_argument('--omega',type=float,default=3.);ap.add_argument('--lean',type=float,default=.25);args=ap.parse_args()
    folder=ROOT/'results/athletic_box_jump/yun_160_refined_flight_v1';out=folder/args.label;out.mkdir(exist_ok=True);air=dict(np.load(folder/'flight_motion.npz'));fr=json.loads((folder/'resolved_audit.json').read_text());ground=dict(np.load(ROOT/fr['source_ground']/'dynamic_ground_integrated.npz'));P=air['contact_positions_m'];front=fr['box_front_y_m'];H=1.6;inside=(P[:,:,1]<front)&(P[:,:,1]>front-.8)&(abs(P[:,:,0]+.025)<.5);ids=np.flatnonzero(inside.all(1)&(P[:,:6,2].min(1)<=H+.003)&(P[:,6:,2].min(1)<=H+.003)&(P[:,:,2].min(1)>H-.003));ci=int(ids[0]);q0=air['qpos'][ci].copy();u0=air['u'][ci].copy();t0=float(air['times'][ci]);r=MeasuredJumpRig(87,1.78);apply(r);b,m,d=r.b,r.m,r.d;n=r.n;W=m.body_mass.sum()*G;r.contact_stand_override=ground['qpos'][0];geom=contact_geometry(r);points=geom['points'];b.pose(ground['qpos'][0]);Rquiet=d.xmat[m.body('torso').id].reshape(3,3).copy();Rfeet={s:d.xmat[m.body('calcn_'+s).id].reshape(3,3).copy() for s in ['r','l']};stand=ground['qpos'][0,r.ids];B,_=r.set_state(q0,u0);mujoco.mj_subtreeVel(m,d);C0=d.subtree_com[1].copy();V0=d.subtree_linvel[1].copy();capture=C0[:2]+V0[:2]/np.sqrt(G/max(.35,C0[2]-H));capture[1]=np.clip(capture[1],P[ci,:,1].min()+.04,P[ci,:,1].max()-.04);A=C0[:2]-capture;D=V0[:2]+args.omega*A
    def xy(t):
        e=np.exp(-args.omega*t);return capture+(A+D*t)*e,(D-args.omega*(A+D*t))*e
    upper=[i for i,name in enumerate(b.ind_names) if name.startswith(('elv_angle_','shoulder_','elbow_','pro_sup_','deviation_','flexion_')) or name in ['lat_bending','axial_rotation']];paths=[Polynomial(q0[adr],u0[6+i],0,stand[i],0,0,1.10) for i,adr in enumerate(r.ids)];feet=[]
    def rot_path(bid,Rgoal,T,via=None):
        jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);R0=d.xmat[bid].reshape(3,3).copy();w=R0.T@(jr@d.qvel);end=Rotation.from_matrix(R0.T@Rgoal).as_rotvec()
        if via is None:polys=[(0.,T,Polynomial(np.zeros(3),w,0,end,0,0,T))]
        else:middle=np.array([args.lean,0.,0.]);polys=[(0.,.18,Polynomial(np.zeros(3),w,0,middle,0,0,.18)),(.18,T,Polynomial(middle,0,0,end,0,0,T-.18))]
        return R0,polys
    def orient(path,t):
        R0,polys=path;start,end,poly=next((entry for entry in polys if t<entry[1]),polys[-1]);s=np.clip(t-start,0,end-start);phi=poly.value(s);rate=poly.value(s,1) if t<polys[-1][1] else np.zeros(3);Rt=R0@Rotation.from_rotvec(phi).as_matrix();return Rt,Rt@right_jacobian(phi)@rate
    r.set_state(q0,u0)
    for side in ['r','l']:
        bid=m.body('calcn_'+side).id;local=m.site_pos[m.site('subject_sole_reference_'+side).id].copy();J,p=b.jacobian(bid,local);v=J@d.qvel;end=p.copy();end[:2]+=.020*v[:2];end[2]=H;feet.append((bid,local,Polynomial(p,v,0,end,0,0,.060),rot_path(bid,Rfeet[side],.060)))
    torso=m.body('torso').id;torso_path=rot_path(torso,Rotation.from_rotvec([.15,0,0]).as_matrix()@Rquiet,1.10,via=True);lo=m.jnt_range[b.independent,0];hi=m.jnt_range[b.independent,1].copy()
    for i,name in enumerate(b.ind_names):
        if name.startswith('shoulder_elv'):hi[i]=min(hi[i],1.84)
    def velocity(q,t,preferred):
        B,_=r.set_state(q,np.zeros(n));E=[];rhs=[];Jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,Jc,1);pos,rate=xy(t);E.append((Jc@B)[:2]);rhs.append(rate+30*(pos-d.subtree_com[1,:2]))
        for bid,local,path,rotation in feet:
            J,p=b.jacobian(bid,local);tt=min(t,.060);E.append(J@B);rhs.append((path.value(tt,1) if t<.060 else np.zeros(3))+30*(path.value(tt)-p));jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);Rt,wr=orient(rotation,t);err=Rotation.from_matrix(Rt@d.xmat[bid].reshape(3,3).T).as_rotvec();E.append(jr@B);rhs.append(wr+30*err)
        for i,name in enumerate(b.ind_names):
            if i in upper or name.startswith('mtp_angle'):
                target=paths[i].value(min(t,1.10))[0];rate=paths[i].value(t,1)[0] if t<1.10 else 0.;row=np.zeros((1,n));row[0,6+i]=1.;E.append(row);rhs.append(np.array([rate+20*(target-q[r.ids[i]])]))
        ref=np.array([poly.value(min(t,1.10))[0] for poly in paths]);vr=np.array([poly.value(t,1)[0] if t<1.10 else 0. for poly in paths]);goal=np.r_[np.zeros(6),vr+10*(ref-q[r.ids])];weights=np.r_[np.full(6,.01),np.ones(n-6)];Q=np.diag(weights)+np.eye(n)*50.;linear=weights*goal+50*preferred;jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,torso);Rt,wr=orient(torso_path,t);err=Rotation.from_matrix(Rt@d.xmat[torso].reshape(3,3).T).as_rotvec();L=jr@B;Q+=100*L.T@L;linear+=100*L.T@(wr+12*err);E=np.vstack(E);rhs=np.concatenate(rhs);sc=np.maximum(np.linalg.norm(E,axis=1),1e-6);E/=sc[:,None];rhs/=sc;_,T,piv=qr(E.T,pivoting=True,mode='economic');rank=int((abs(np.diag(T))>1e-8).sum());keep=piv[:rank];limits=[];bounds=[]
        for i in range(n-6):
            row=np.zeros(n);row[6+i]=1.;limits.extend([row,-row]);bounds.extend([100*(lo[i]+1e-7-q[r.ids[i]]),100*(q[r.ids[i]]-hi[i]+1e-7)])
        u=quadprog.solve_qp((Q+Q.T)/2,linear,np.vstack([E[keep],limits]).T,np.r_[rhs[keep],bounds],meq=rank)[0]
        if abs(E@u-rhs).max()>1e-5:raise RuntimeError('Kinematic task equality failed')
        return u,B
    qs=[];us=[];ts=[];q=q0.copy();previous=u0.copy();failure=None
    for k in range(round(args.duration*args.hz)+1):
        t=k/args.hz
        try:
            u,B=(u0.copy(),r.set_state(q,u0)[0]) if k==0 else velocity(q,t,previous);qs.append(q.copy());us.append(u);ts.append(t0+t)
            if k%200==0:print('QUIET_RECOVERY_POSE',round(t,3),flush=True)
            if k==round(args.duration*args.hz):break
            dt=1/args.hz;mid=q.copy();mujoco.mj_integratePos(m,mid,B@u,dt/2);mid=b.complete(mid);um,Bm=velocity(mid,t+dt/2,u);nq=q.copy();mujoco.mj_integratePos(m,nq,Bm@um,dt);q=b.complete(nq);previous=um
        except (ValueError,RuntimeError) as error:failure=str(error);break
    qs=np.array(qs);us=np.array(us);ts=np.array(ts);np.savez_compressed(out/'kinematic_candidate.npz',times=ts,qpos=qs,u=us);spline=CubicSpline(ts,us,axis=0);arrays={k:[] for k in ['times','qpos','u','body_xpos','body_xmat','com','velocity','angular_momentum','contact_positions_m','contact_forces_N']};clear=np.inf;root_error=0.;previousF=np.zeros(36);wrench_failure=None
    for k,(time,q,u) in enumerate(zip(ts,qs,us)):
        B,c=r.set_state(q,u);a=B@spline(time,1)+c;M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);target=(M@a+d.qfrc_bias-d.qfrc_passive)[:6]/W;J=[];positions=[]
        for bid,local,*_ in points:jac,p=b.jacobian(bid,local);J.append(jac.T[:6]);positions.append(p)
        E=np.concatenate(J,axis=1);sc=np.maximum(np.linalg.norm(E,axis=1),.001);ineq=[]
        for j in range(12):
            for x,y,z in [(0,0,1)]+[(-np.cos(phi),-np.sin(phi),.8*np.cos(np.pi/16)) for phi in np.linspace(0,2*np.pi,16,endpoint=False)]:row=np.zeros(36);row[3*j:3*j+3]=[x,y,z];ineq.append(row)
        try:f=quadprog.solve_qp(np.eye(36)*1.1,.1*previousF,np.vstack([E/sc[:,None],ineq]).T,np.r_[target/sc,np.zeros(len(ineq))],meq=6)[0];previousF=f
        except ValueError as error:wrench_failure=dict(time_s=float(time),reason=str(error),required_root_wrench_SI=(target*W).tolist());break
        F=f.reshape(-1,3)*W;root_error=max(root_error,float(abs(E@f-target).max()*W));caps,radii=segments(m,d);clear=min(clear,minimum_clearance(caps,radii,front,H));mujoco.mj_subtreeVel(m,d);vals=dict(times=time,qpos=q,u=u,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),contact_positions_m=positions,contact_forces_N=F)
        for key,value in vals.items():arrays[key].append(value)
    data={key:np.array(value) for key,value in arrays.items()};np.savez_compressed(out/'supported_motion.npz',**data);speed=float(np.linalg.norm(data['velocity'][-1])) if len(data['times']) else None;peak=float(np.linalg.norm(data['contact_forces_N'].sum(1),axis=1).max()) if len(data['times']) else None;success=bool(failure is None and wrench_failure is None and clear>=0 and speed is not None and speed<.1)
    report=dict(status='supported_quiet_recovery_candidate' if success else 'rejected_quiet_recovery',completed_supported_landing=success,failure=failure,contact_wrench_failure=wrench_failure,contact_time_s=t0,contact_index=ci,duration_s=float(ts[-1]-t0),root_balance_error_max_SI=root_error,body_capsule_clearance_m=float(clear),final_COM_speed_m_s=speed,peak_landing_GRF_N=peak,contact_tolerance_m=.003,maximum_sole_penetration_m=max(0.,H-float(data['contact_positions_m'][:,:,2].min())) if len(data['times']) else None,root_actuator=False,force_clipping=False,pose_reset=False,velocity_reset=False,motion_design='Quiet upper-body + material sole paths and horizontal COM; height emerges from joint coordination; inverse contact forces must be feasible.',quiet_upper_body=True,coordinated_trunk=True,upper_joint_peak_speed_deg_s={b.ind_names[i]:float(np.rad2deg(abs(us[:,6+i]).max())) for i in upper},integration_hz=args.hz,muscle_force_analysis_allowed=False)
    (out/'landing_audit.json').write_text(json.dumps(report,indent=2));print('QUIET_RECOVERY_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
