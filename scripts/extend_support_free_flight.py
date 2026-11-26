"""Integrate the unactuated root in flight; no airborne pose/COM correction.

Internal joint coordination is prescribed without a strength ceiling. The six
root accelerations follow the reduced Newton--Euler equations with zero contact.
COM ballistics and momentum are independent checks of the resulting motion.
Landing is excluded and must never be implied by the renderer metadata.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco,quadprog
from scipy.integrate import solve_ivp
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from centroidal_jump import Polynomial
from mechanics_engine_adapter import sample
from audit_jump_coordination import audit
ROOT=Path(__file__).resolve().parents[1];G=9.80665


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder');ap.add_argument('--output-dir');ap.add_argument('--duration',type=float,default=.60);ap.add_argument('--reach-time',type=float,default=.48);ap.add_argument('--landing-hip',type=float,default=.35);ap.add_argument('--landing-knee',type=float,default=.45);ap.add_argument('--landing-shoulder',type=float,default=.8);ap.add_argument('--hz',type=int,default=125);args=ap.parse_args();source=ROOT/args.folder;folder=ROOT/args.output_dir if args.output_dir else source;folder.mkdir(parents=True,exist_ok=True)
    z=np.load(source/'dynamic_ground_integrated.npz');r=MeasuredJumpRig(87,1.78);b,m,d=r.b,r.m,r.d;ids=np.array(r.ids);names=b.ind_names
    release=np.flatnonzero((z['times'][:-1]>.85)&(z['contact_forces_N'][:,:,2].sum(1)<1e-4));takeoff_index=int(release[0]) if len(release) else len(z['times'])-1
    q0=z['qpos'][takeoff_index].copy();u0=z['u'][takeoff_index].copy();to=float(z['times'][takeoff_index]);r.set_state(q0,u0);mujoco.mj_subtreeVel(m,d);C0=d.subtree_com[1].copy();V0=d.subtree_linvel[1].copy();H0=d.subtree_angmom[1].copy()
    if V0[2]<.5:raise RuntimeError('No upward takeoff to extend')
    r.contact_stand_override=z['qpos'][0].copy();points=contact_geometry(r)['points'];r.set_state(q0,u0)
    lower=m.jnt_range[b.independent,0];upper=m.jnt_range[b.independent,1];angle=q0[ids];rate=u0[6:];brake=.04
    for a,v,lo,hi in zip(angle,rate,lower,upper):
        if abs(v)>1e-5:brake=min(brake,1.5*((hi-a) if v>0 else (a-lo))/abs(v))
    if brake<.005:raise RuntimeError(('Insufficient ROM for smooth release',brake))
    stop=angle+.5*brake*rate
    def root_acceleration(q,u,joint_acc):
        B,curve=r.set_state(q,u);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);K=M@B
        root=-np.linalg.solve(K[:6,:6],(d.qfrc_bias+M@curve)[:6]+K[:6,6:]@joint_acc)
        return root,B,curve
    desired=-rate/brake;root,B,c=root_acceleration(q0,u0,desired);base=B@np.r_[root,desired]+c
    chosen=[i for i,n in enumerate(names) if n.startswith(('hip_flexion_','knee_angle_','ankle_angle_','mtp_angle_'))]
    M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);K=M@B;D=np.zeros((r.n,len(chosen)));D[6+np.array(chosen),np.arange(len(chosen))]=1;D[:6]=-np.linalg.solve(K[:6,:6],K[:6,6+np.array(chosen)])
    rows=[];limits=[]
    for bid,local,p0,fore in points:
        J,p=b.jacobian(bid,local)
        if p[2]>.003:continue
        Jd=J.copy();Jrd=J.copy();mujoco.mj_jacDot(m,d,Jd,Jrd,p,bid)
        rows.append((J@B@D)[2]);limits.append(.5-float((J@base+Jd@d.qvel)[2]))
    if rows:desired[chosen]+=quadprog.solve_qp(np.eye(len(chosen)),np.zeros(len(chosen)),np.array(rows).T,np.array(limits))[0]
    braking=Polynomial(angle,rate,desired,stop,0,0,brake);tuck=stop.copy();landing=stop.copy()
    initial=z['qpos'][0,ids]
    for i,n in enumerate(names):
        if n.startswith(('hip_rotation_','subtalar_')):tuck[i]=landing[i]=initial[i]
        if n.startswith('hip_adduction_'):tuck[i]=landing[i]=-.10
    for side in ['r','l']:
        for n,v,w in [('hip_flexion',1.40,args.landing_hip),('knee_angle',1.8,args.landing_knee),('ankle_angle',0.,.05),('mtp_angle',0.,0.),('shoulder_elv',.8,args.landing_shoulder),('elbow_flexion',.4,.4)]:
            i=names.index(n+'_'+side);tuck[i]=v;landing[i]=w
    mid=max(brake+.10,min(.23,float(V0[2]/G)));end=min(args.duration,args.reach_time)
    if end<=mid+.05:raise ValueError('Flight duration too short for coordinated return')
    fold=Polynomial(stop,0,0,tuck,0,0,mid-brake);reach=Polynomial(tuck,0,0,landing,0,0,end-mid)
    def joints(t,k=0):
        return braking.value(t,k) if t<brake else fold.value(t-brake,k) if t<mid else reach.value(t-mid,k) if t<end else landing.copy() if k==0 else np.zeros_like(landing)
    def state(t,y):
        q=q0.copy();q[:3]=y[:3];q[3:7]=y[3:7]/np.linalg.norm(y[3:7]);q[ids]=joints(t);q=b.complete(q);u=np.r_[y[7:13],joints(t,1)];a,B,c=root_acceleration(q,u,joints(t,2));return q,u,np.r_[a,joints(t,2)],B,c
    def rhs(t,y):
        q,u,a,B,c=state(t,y);quat=q[3:7];omega=u[3:6];qdot=.5*np.r_[-quat[1:]@omega,quat[0]*omega+np.cross(quat[1:],omega)]
        return np.r_[u[:3],qdot,a[:6]]
    y0=np.r_[q0[:7],u0[:6]]
    solution=solve_ivp(rhs,[0,args.duration],y0,method='DOP853',rtol=2e-9,atol=2e-10,max_step=.003,dense_output=True)
    if not solution.success:raise RuntimeError(solution.message)
    keys=['times','qpos','qvel','qacc','body_xpos','body_xmat','com','velocity','angular_momentum','force_world_N','contact_forces_N','contact_positions_m','mode','root_residual']
    arrays={k:[] for k in keys};cross=0.;source_hz=1/float(np.median(np.diff(z['times'])));ground=np.arange(0,takeoff_index,max(1,round(source_hz/args.hz)))
    work=[(float(z['times'][i]),int(i)) for i in ground]+[(to+float(t),None) for t in np.unique(np.r_[np.arange(0,args.duration,1/args.hz),args.duration])]
    for k,(t,i) in enumerate(work):
        air=i is None
        if air:q,u,a,B,c=state(t-to,solution.sol(t-to));forces=np.zeros((len(points),3))
        else:q=z['qpos'][i];u=z['u'][i];a=z['udot'][i];B,c=r.coordinates(q,u);forces=z['contact_forces_N'][i]
        v=B@u;acc=B@a+c;mech=sample(r,q,v,acc);mujoco.mj_subtreeVel(m,d);P=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);external=np.r_[forces.sum(0),np.cross(P,forces).sum(0)]
        values=dict(times=t,qpos=q,qvel=v,qacc=acc,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),force_world_N=forces.sum(0),contact_forces_N=forces,contact_positions_m=P,mode=2 if air else 0,root_residual=mech['body_wrenches'].sum(0)-external)
        for key,value in values.items():arrays[key].append(value)
        cross=max(cross,mech['crosscheck_max'])
        if k%100==0:print('FREE_ROOT_REPLAY',k,len(work),flush=True)
    out={k:np.array(v) for k,v in arrays.items()};np.savez_compressed(folder/'motion.npz',**out);b.pose(b.q0);np.savez_compressed(folder/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
    gap=out['contact_positions_m'][:,:,2].min(1);crossing=np.flatnonzero((out['times'][1:]>to+.05)&(gap[:-1]>=0)&(gap[1:]<0));first_contact=None
    if len(crossing):
        i=int(crossing[0]);first_contact=float(out['times'][i]+(out['times'][i+1]-out['times'][i])*gap[i]/(gap[i]-gap[i+1]))
    meta=dict(status='unvalidated_precontact_dynamics',movement='box_jump_reference_replay',scale=b.report['global_normalization_scale'],body_names=[m.body(i).name for i in range(m.nbody)],phase_times_s=dict(bottom=.85,start_ascent=.85,end_ascent=to,end=float(out['times'][-1])),force_analysis_allowed=False,physical_task=True,landing_simulated=False,display_platform_only=False,first_descending_floor_contact_s=first_contact,post_contact_samples_are_event_search_only=True)
    (folder/'summary.json').write_text(json.dumps(meta,indent=2));geometry=audit(folder);air=out['mode']==2;t=out['times'][air]-to;expected=C0+t[:,None]*V0+.5*t[:,None]**2*np.array([0,0,-G]);ROM=max(0.,float(np.maximum(lower-out['qpos'][:,ids],out['qpos'][:,ids]-upper).max()))
    report=dict(status='free_root_dynamics_review',source_ground=str(source),takeoff_s=to,motion_sha256=hashlib.sha256((folder/'motion.npz').read_bytes()).hexdigest(),takeoff_velocity_m_s=V0.tolist(),takeoff_H_kg_m2_s=H0.tolist(),peak_GRF_N=float(out['force_world_N'][:,2].max()),independent_engine_error_SI=cross,ground_root_residual_max_SI=float(abs(out['root_residual'][~air]).max()),flight_root_residual_max_SI=float(abs(out['root_residual'][air]).max()),ballistic_COM_error_max_m=float(np.linalg.norm(out['com'][air]-expected,axis=1).max()),angular_momentum_drift_max_kg_m2_s=float(np.linalg.norm(out['angular_momentum'][air]-H0,axis=1).max()),ROM_excess_rad=ROM,minimum_flight_sole_height_m=float(out['contact_positions_m'][air,:,2].min()),release_braking_duration_s=brake,coordination_screen=geometry,root_position_correction=False,root_actuator=False,landing_simulated=False,human_motion_validated=False,force_analysis_allowed=False)
    report.update(first_descending_floor_contact_s=first_contact,physical_scope='Takeoff to first surface contact. Later samples bracket a contact event; they are not valid supported motion.')
    (folder/'flight_verification.json').write_text(json.dumps(report,indent=2));print('FREE_ROOT_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
