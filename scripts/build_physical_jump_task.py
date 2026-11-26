"""Task-defined launch and exact centroidal flight, independent of video pixels.

The ground controller defines joint coordination. Ground reactions are checked
against Newton--Euler balance and friction; no root actuator or strength cap.
Flight preserves launch COM velocity and angular momentum. Landing is excluded.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,mujoco,quadprog
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from retarget_measured_jump import MeasuredJumpRig
from native_jump_ground_path import solve as solve_ground
from build_contact_jump import contact_geometry
from build_loaded_squat_power import contact_solver
from centroidal_jump import Polynomial
from mechanics_engine_adapter import sample
from audit_jump_coordination import audit as audit_coordination
ROOT=Path(__file__).resolve().parents[1];G=9.80665

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='physical_jump_task_v1');ap.add_argument('--ground-hz',type=int,default=500);ap.add_argument('--hz',type=int,default=125);ap.add_argument('--reuse-ground',action='store_true');ap.add_argument('--dynamic-ground-reference');ap.add_argument('--push-duration',type=float,default=.38);ap.add_argument('--launch-vz-target',type=float);ap.add_argument('--arm-end',type=float,default=.9);args=ap.parse_args()
 out=ROOT/'results/physical_jump_task'/args.label;out.mkdir(parents=True,exist_ok=True);r=MeasuredJumpRig(87,1.78);b,m,d=r.b,r.m,r.d;mass=m.body_mass.sum();ids=np.array(r.ids);names=b.ind_names
 bottom=.90;to=bottom+args.push_duration;toe=bottom+.5*args.push_duration
 profile=dict(simple_arm_task=True,bottom_angles=dict(hip_flexion=1.05,knee_angle=1.5,ankle_angle=.25,shoulder_elv=.06,elbow_flexion=.18),
  takeoff_angles=dict(hip_flexion=.28,knee_angle=.48,ankle_angle=-.35,shoulder_elv=1.35,elbow_flexion=.35),
  takeoff_rates=dict(hip_flexion=-4.,knee_angle=-8.,ankle_angle=-5.,shoulder_elv=1.,elbow_flexion=0.),
  crouch_pitch_rad=.25,bottom_forward_offset_m=-.025,takeoff_forward_offset_m=-.08,takeoff_forward_speed_m_s=-.5)
 initial,_=r.planted_pose(.1,.15,.07,arms=(np.pi/2,.12,.15),trunk=0.);b.pose(initial);initial[2]-=np.mean([d.site_xpos[m.site('subject_sole_reference_'+s).id,2]for s in ['r','l']]);b.pose(initial)
 r.contact_stand_override=initial;geometry=contact_geometry(r);points=geometry['points'];fore=np.array([p[3]for p in points]);frames={s:[(m.body(stem+'_'+s).id,d.xpos[m.body(stem+'_'+s).id].copy(),d.xmat[m.body(stem+'_'+s).id].reshape(3,3).copy())for stem in ['calcn','toes']]for s in ['r','l']};C0=d.subtree_com[1].copy()
 if not args.reuse_ground:
  if args.dynamic_ground_reference:
   from dynamic_ground_path import solve as solve_dynamic
   seed=np.load(ROOT/args.dynamic_ground_reference/'dynamic_ground_integrated.npz');reference=CubicSpline(seed['times'],seed['qpos'],axis=0);centers=[];seed_end=float(seed['times'][-1])
   for qseed in seed['qpos']:b.pose(qseed);centers.append(d.subtree_com[1].copy())
   center=CubicSpline(seed['times'],np.array(centers),axis=0)
   nominal_center=center
   if args.launch_vz_target is not None:
    yprep=Polynomial(C0[1],0,0,C0[1]-.10,0,0,bottom-.25);ypush=Polynomial(C0[1]-.10,0,0,C0[1]-.20,-.5,0,to-bottom);zlow=float(nominal_center(bottom)[2]);T=to-bottom;pulse=np.pi/(2*T)*(args.launch_vz_target+G*T/2)
    def center(t,k=0):
     if t<=bottom:
      v=nominal_center(t,k);v[0]=C0[0]if k==0 else 0.;v[1]=C0[1]if t<=.25 and k==0 else 0. if t<=.25 else yprep.value(t-.25,k)[0];return v
     s=t-bottom;u=s/T;v=np.zeros(3);v[0]=C0[0]if k==0 else 0.;v[1]=ypush.value(s,k)[0]
     v[2]=zlow+pulse*T*T*(u/np.pi-np.sin(np.pi*u)/np.pi**2)-G*T*T*u**3/6 if k==0 else pulse*T*(1-np.cos(np.pi*u))/np.pi-G*T*u*u/2 if k==1 else pulse*np.sin(np.pi*u)-G*u
     return v
   def template(t):
    source_t=t if t<=bottom else bottom+(t-bottom)*(seed_end-bottom)/(to-bottom)
    q=reference(source_t);q[3:7]/=np.linalg.norm(q[3:7]);return q
   # A bilateral sagittal task must not obtain support/clearance by spinning
   # the femurs or everting the feet. These are task assumptions, not limits
   # on muscle strength. Keep the planted-pose transverse/frontal coordinates.
   movable=('hip_flexion_','knee_angle_','ankle_angle_','mtp_angle_','shoulder_elv_','elbow_flexion_')
   locked={name:float(initial[ids[i]])for i,name in enumerate(names)if not name.startswith(movable)}
   task=dict(locked_joint_targets=locked,posture_weight_by_joint={name:25. for name in names if name.startswith(('shoulder_elv_','elbow_flexion_'))},adaptive_contacts=True,
    com_tracking_weight=20000.,ground_trunk_weight=1000.,latch_propulsion_state=args.launch_vz_target is not None,launch_vz_m_s=args.launch_vz_target)
   solve_dynamic(r,template,lambda t:center(t),lambda t:center(t,1),lambda t:center(t,2),points,bottom,toe,to,to,out,hz=args.ground_hz,task_profile=task)
  else:solve_ground(r,initial,frames,C0,bottom,toe,to,out,hz=args.ground_hz,task_profile=profile)
 ground=np.load(out/'dynamic_ground_integrated.npz');tg=ground['times'];ug=ground['u'];du=CubicSpline(tg,ug);qto=ground['qpos'][-1].copy();uto=ug[-1].copy();B,curve=r.set_state(qto,uto);mujoco.mj_subtreeVel(m,d);Ct=d.subtree_com[1].copy();Vt=d.subtree_linvel[1].copy();Ht=d.subtree_angmom[1].copy();assert Vt[2]>.5,('Insufficient upward launch',Vt)
 to=float(tg[-1])
 angle=qto[ids].copy();rate=uto[6:].copy();brake=.06;lower=m.jnt_range[b.independent,0];upper=m.jnt_range[b.independent,1];stop=angle+.5*brake*rate
 # Select a finite internal acceleration that lets the toes separate without
 # an impulse, an external air force, or a velocity reset.
 fullM=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,fullM);Mr=B.T@fullM@B;bias=B.T@(d.qfrc_bias+fullM@curve);desired=-rate/brake
 free_root=-np.linalg.solve(Mr[:6,:6],bias[:6]+Mr[:6,6:]@desired);baseline=B@np.r_[free_root,desired]+curve
 chosen=[i for i,n in enumerate(names)if n.startswith(('hip_flexion_','knee_angle_','ankle_angle_','mtp_angle_'))];N=len(chosen);control=np.zeros((r.n,N));control[6+np.array(chosen),np.arange(N)]=1.;control[:6]=-np.linalg.solve(Mr[:6,:6],Mr[:6,6+np.array(chosen)])
 rows=[];rhs=[]
 for (bid,local,_,is_fore)in points:
  if not is_fore:continue
  J,p=b.jacobian(bid,local);Jd=np.zeros_like(J);Jrd=Jd.copy();mujoco.mj_jacDot(m,d,Jd,Jrd,p,bid);rows.append((J@B@control)[2]);rhs.append(.5-float((J@baseline+Jd@d.qvel)[2]))
 correction=quadprog.solve_qp(np.eye(N),np.zeros(N),np.array(rows).T,np.array(rhs))[0];a0=desired.copy();a0[chosen]+=correction
 braking=Polynomial(angle,rate,a0,stop,0,0,brake);target=stop.copy()
 for s in ['r','l']:
  for name,value in [('hip_flexion',1.7),('knee_angle',2.0),('ankle_angle',.05),('mtp_angle',0.),('shoulder_elv',args.arm_end),('elbow_flexion',.4)]:target[names.index(name+'_'+s)]=value
 tuck_T=max(.30,float(Vt[2]/G)*1.05);tuck=Polynomial(stop,0,0,target,0,0,tuck_T-brake);duration=max(tuck_T+.04,1.45*float(Vt[2]/G));duration=min(duration,1.8*float(Vt[2]/G))
 def joints(tau,k=0):
  if tau<brake:return braking.value(tau,k)
  if tau<tuck_T:return tuck.value(tau-brake,k)
  return target.copy()if k==0 else np.zeros_like(target)
 def omega(tau,quat):
  q=b.q0.copy();q[:3]=0;q[3:7]=quat/np.linalg.norm(quat);q[ids]=joints(tau);q=b.complete(q);B,_=r.coordinates(q,np.zeros(r.n));b.pose(q);A=np.zeros((3,m.nv));mujoco.mj_angmomMat(m,d,A,1);u=np.r_[np.zeros(6),joints(tau,1)]
  return np.linalg.solve(A[:,3:6],Ht-A@(B@u))
 def dq(tau,quat):
  w=omega(tau,quat);return .5*np.r_[-quat[1:]@w,quat[0]*w+np.cross(quat[1:],w)]
 def step(t,q,h):
  k1=dq(t,q);k2=dq(t+h/2,q+h*k1/2);k3=dq(t+h/2,q+h*k2/2);k4=dq(t+h,q+h*k3);v=q+h*(k1+2*k2+2*k3+k4)/6;return v/np.linalg.norm(v)
 grid=np.linspace(0,duration,int(np.ceil(duration*6000))+1);qs=[qto[3:7].copy()]
 for t,tn in zip(grid[:-1],grid[1:]):qs.append(step(float(t),qs[-1],float(tn-t)))
 past=step(0,qto[3:7],-.001);rot=RotationSpline(np.r_[-.001,grid],Rotation.from_quat(np.roll(np.array([past,*qs]),-1,axis=1)))
 def air_velocity(tau):
  q=b.q0.copy();q[:3]=0;q[3:7]=np.roll(rot(tau).as_quat(),1);q[ids]=joints(tau);q=b.complete(q);b.pose(q);q[:3]=Ct+Vt*tau+.5*np.array([0,0,-G])*tau*tau-d.subtree_com[1];B,_=r.coordinates(q,np.zeros(r.n));b.pose(q);A=np.zeros((3,m.nv));J=A.copy();mujoco.mj_angmomMat(m,d,A,1);mujoco.mj_jacSubtreeCom(m,d,J,1);u=np.r_[np.zeros(6),joints(tau,1)];u[3:6]=np.linalg.solve(A[:,3:6],Ht-A@(B@u));u[:3]=Vt+np.array([0,0,-G])*tau-J@(B@u);return q,u
 selected=np.unique(np.r_[np.arange(0,len(tg)-1,max(1,round(args.ground_hz/args.hz))),np.argmin(abs(tg-bottom)),np.argmin(abs(tg-toe))]);selected=selected[tg[selected]<to-1e-8]
 work=[(float(tg[i]),int(i))for i in selected]+[(float(to+t),None)for t in np.unique(np.r_[np.arange(0,duration,1/args.hz),duration])];solvers={False:contact_solver([p[2]for p in points],mass),True:contact_solver([p[2]for p in points if p[3]],mass)}
 arrays={k:[]for k in ['times','qpos','qvel','qacc','body_xpos','body_xmat','com','velocity','angular_momentum','force_world_N','contact_forces_N','contact_positions_m','mode','root_residual','foot_error_m','foot_speed_m_s']};cross=0.
 for k,(t,index)in enumerate(work):
  flying=index is None
  if flying:
   tau=t-to;q,u=air_velocity(tau);eps=5e-6;ud=(air_velocity(tau+eps)[1]-air_velocity(tau-eps)[1])/(2*eps)
  else:q=ground['qpos'][index].copy();u=ug[index];ud=ground['udot'][index]if args.dynamic_ground_reference else du(t,1)
  B,c=r.coordinates(q,u);v=B@u;acc=B@ud+c;b.pose(q,v,acc);mujoco.mj_subtreeVel(m,d);mech=sample(r,q,v,acc);P=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);forces=np.zeros((len(points),3));active=np.zeros(len(points),bool)if flying else fore if t>toe else np.ones(len(points),bool)
  if not flying:
   try:
    if args.dynamic_ground_reference:forces=ground['contact_forces_N'][index].copy()
    else:forces[active],_=solvers[t>toe](mech['body_wrenches'].sum(0))
   except RuntimeError as e:
    (out/'failure.json').write_text(json.dumps(dict(status='rejected_contact_infeasible',time_s=t,reason=str(e)),indent=2));raise
  if args.dynamic_ground_reference and not flying:
   for event in json.loads((out/'dynamic_ground_summary.json').read_text())['release_events']:
    if t>=event['time_s']:active[event['point_indices']]=False
  applied=np.r_[forces.sum(0),np.cross(P if args.dynamic_ground_reference else np.array([p[2]for p in points]),forces).sum(0)];residual=mech['body_wrenches'].sum(0)-applied
  errors=[];speeds=[]
  for j in np.flatnonzero(active):
   J,actual=b.jacobian(points[j][0],points[j][1]);errors.append(np.linalg.norm(actual-points[j][2]));speeds.append(np.linalg.norm(J@v))
  vals=dict(times=t,qpos=q,qvel=v,qacc=acc,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),force_world_N=forces.sum(0),contact_forces_N=forces,contact_positions_m=P,mode=2 if flying else 1 if t>toe else 0,root_residual=residual,foot_error_m=max(errors,default=0.),foot_speed_m_s=max(speeds,default=0.))
  for key,value in vals.items():arrays[key].append(value)
  if k%125==0:print('PHYSICAL_TASK',k,len(work),'GRF',round(float(forces[:,2].sum())),'root',round(float(abs(residual).max()),5),flush=True)
  if k%20==0:cross=max(cross,mech['crosscheck_max'])
 z={k:np.array(v)for k,v in arrays.items()};np.savez_compressed(out/'motion.npz',**z);air=z['mode']==2;ta=z['times'][air]-to;expected=Ct+ta[:,None]*Vt+.5*ta[:,None]**2*np.array([0,0,-G]);normal=z['contact_forces_N'][:,:,2];R=Rotation.from_quat(np.roll(z['qpos'][air,3:7],-1,axis=1));Rto=Rotation.from_quat(np.roll(qto[3:7],-1));sole=z['contact_positions_m'][air,:,2].min(1);peak=int(np.argmax(sole));suggested=max(0.,float(sole[peak]-.03));feet=z['contact_positions_m'][air][peak];near=float(feet[:,1].max()+.04)
 checks=dict(independent_engine_error_SI=cross,stance_root_max_SI=float(abs(z['root_residual'][~air]).max()),flight_root_max_SI=float(abs(z['root_residual'][air]).max()),ballistic_COM_max_error_m=float(np.linalg.norm(z['com'][air]-expected,axis=1).max()),flight_H_max_drift=float(np.linalg.norm(z['angular_momentum'][air]-Ht,axis=1).max()),foot_position_max_m=float(z['foot_error_m'].max()),foot_speed_max_m_s=float(z['foot_speed_m_s'].max()),minimum_normal_N=float(normal.min()),friction_excess_N=float((np.linalg.norm(z['contact_forces_N'][:,:,:2],axis=2)-.8*normal).max()),ROM_excess_rad=max(0.,float(np.maximum(lower-z['qpos'][:,ids],z['qpos'][:,ids]-upper).max())),flight_pelvis_rotation_max_deg=float(np.degrees((Rto.inv()*R).magnitude()).max()),early_flight_minimum_sole_height_m=float(z['contact_positions_m'][(z['times']>=to)&(z['times']<=to+.02),:,2].min()),peak_GRF_N=float(z['force_world_N'][:,2].max()))
 gates=dict(engine=cross<1e-5,stance_balance=checks['stance_root_max_SI']<1e-4,flight_balance=checks['flight_root_max_SI']<.1,COM=checks['ballistic_COM_max_error_m']<1e-6,H=checks['flight_H_max_drift']<1e-4,feet=checks['foot_position_max_m']<.001,foot_speed=checks['foot_speed_max_m_s']<.01,unilateral=checks['minimum_normal_N']>-.001,friction=checks['friction_excess_N']<.001,ROM=checks['ROM_excess_rad']<.001,separation=checks['early_flight_minimum_sole_height_m']>-.002,no_tumbling=checks['flight_pelvis_rotation_max_deg']<45.)
 report=dict(status='physical_task_precontact_candidate',checks=checks,gates=gates,takeoff_velocity_m_s=Vt.tolist(),takeoff_H_kg_m2_s=Ht.tolist(),maximum_full_sole_height_m=float(sole.max()),illustrative_clearance_height_m=suggested,landing_simulated=False,individual_athlete_prediction=False,video_tracking_in_objective=False,profile=profile,strength_ceiling=False,force_clipping=False,root_support=False,internal_release_acceleration_rad_s2={names[i]:float(a0[i])for i in chosen})
 report.update(task_parameters=vars(args),motion_sha256=hashlib.sha256((out/'motion.npz').read_bytes()).hexdigest(),task_tracking=dict(requested_launch_vz_m_s=args.launch_vz_target,actual_launch_vz_m_s=float(Vt[2]),requested_target_is_a_soft_control_goal=True),flight_tuck_target_rad=dict(hip_flexion=1.7,knee_angle=2.0,shoulder_elv=args.arm_end))
 tracking_error=abs(float(Vt[2])-args.launch_vz_target)/args.launch_vz_target if args.launch_vz_target else None
 report['task_tracking'].update(relative_launch_velocity_error=tracking_error,within_ten_percent=tracking_error is None or tracking_error<=.1)
 report.update(numerical_mechanics_passed=all(gates.values()),human_motion_validated=False,ready_for_comparative_analysis=False)
 (out/'verification.json').write_text(json.dumps(report,indent=2));b.pose(b.q0);np.savez_compressed(out/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
 meta=dict(status='physics_task_review_only',movement='box_jump_reference_replay',scale=b.report['global_normalization_scale'],body_names=[m.body(i).name for i in range(m.nbody)],phase_times_s=dict(bottom=bottom,start_ascent=bottom,end_ascent=to,end=float(z['times'][-1])),force_analysis_allowed=False,display_platform_only=False,box_geometry=dict(top_height_m=suggested,y_bounds_m=[near-.65,near]),physical_task=True,landing_simulated=False)
 (out/'summary.json').write_text(json.dumps(meta,indent=2));coordination=audit_coordination(out);report['coordination_screen']=coordination;(out/'verification.json').write_text(json.dumps(report,indent=2));(out/'motion_quality.json').write_text(json.dumps(dict(status='rejected_foot_crossing' if not coordination['foot_order_preserved'] else 'mechanics_checked_precontact' if all(gates.values()) else 'failed',force_analysis_allowed=False,human_motion_validated=False),indent=2));print('PHYSICAL_TASK_RESULT',json.dumps(report),flush=True)
 if not all(gates.values()):raise RuntimeError(('Physical task gates',gates))
 if not coordination['foot_order_preserved']:raise RuntimeError('Rejected: feet cross despite mechanical balance')
if __name__=='__main__':main()
