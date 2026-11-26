"""Causal posture/timing pilot on inertia-preserving sagittal native skeleton.

Eleven internal joint torques drive three unactuated sagittal base coordinates.
The rigid foot retains heel and forefoot contacts without toe articulation.
Foot contacts, release,
flight and re-contact are native MuJoCo outputs. No launch-state or COM path
targets, no prescribed flight path. Frozen out-of-plane/articular helper
coordinates define a reduced control model, not the muscle-path analysis rig.
"""
from pathlib import Path
import argparse,json,time
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from native_rigid_upper_landing import actual_contact_wrench
from centroidal_jump import Polynomial

ROOT=Path(__file__).resolve().parents[1]
STEMS=['flex_extension','hip_flexion','knee_angle','ankle_angle','mtp_angle','shoulder_elv','elbow_flexion']
DEFAULT=dict(ready_s=.50,crouch_s=.65,push_s=.27,arm_delay_s=0.,ankle_delay=.45,
    hip_bottom=1.55,knee_bottom=1.65,ankle_bottom=.48,trunk_bottom=-.12,
    shoulder_bottom=-.20,shoulder_top=1.50,elbow=.30,
    hip_top=-.05,knee_top=.06,ankle_top=-.65,trunk_top=.05,
    flight_hip=1.1,flight_knee=1.35,flight_ankle=0.,flight_shoulder=.1,flight_trunk=.05,landing_hip=1.3,landing_knee=1.3,prepare_landing_s=.22,prepare_hip=1.2,prepare_knee=.6,prepare_shoulder=-.2,prepare_trunk=-.1,
    kp_scale=1.,leg_kp=500.,ankle_kp=240.,spine_kp=1400.,balance_gain=1.,lean_m=.16,pitch_bottom=.55,pitch_top=.55,pitch_gain=250.,momentum_feedback=True,landing_feedback=True,joint_armature_kg_m2=0.,contact_time_constant_s=.008,landing_contact_time_constant_s=.012,controller='pd',seconds=4.5)

def ease(u):
 if isinstance(u,(float,int,np.floating)):
  u=max(0.,min(1.,float(u)));return u**3*(10-15*u+6*u*u)
 u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)

def build(dt=.001):
 r=MeasuredJumpRig(87,1.78);b,m,d=r.b,r.m,r.d
 q=np.load(ROOT/'results/athletic_box_jump/yun_160_final_cycle_v4/motion.npz')['qpos'][0].copy()
 r.contact_stand_override=q.copy();points=contact_geometry(r)['points'];b.pose(q);X=d.xpos.copy();R=d.xmat.reshape(-1,3,3).copy();anchors=d.xanchor.copy()
 keep={'flex_extension'}|{stem+'_'+s for s in ['r','l'] for stem in STEMS[1:]}
 keep-={'mtp_angle_r','mtp_angle_l'}
 spec=b.spec.copy();spec.compiler.fusestatic=False
 for category in ['actuators','sensors','tendons','keys','pairs','equalities']:
  for item in list(getattr(spec,category)):spec.delete(item)
 for body in spec.bodies:
  bid=m.body(body.name).id
  if bid==0:continue
  if bid!=1:
   parent=int(m.body_parentid[bid]);owner=body.parent
   while not isinstance(owner,mujoco.MjsBody):owner=owner.parent
   frame=owner.add_frame(name='control_frozen_'+body.name,pos=[0,0,0],quat=[1,0,0,0]);body.set_frame(frame)
   body.pos=R[parent].T@(X[bid]-X[parent]);body.quat=np.roll(Rotation.from_matrix(R[parent].T@R[bid]).as_quat(),1)
  else:body.pos=X[bid];body.quat=np.roll(Rotation.from_matrix(R[bid]).as_quat(),1)
  body.explicitinertial=True;body.mass=float(m.body_mass[bid]);body.ipos=m.body_ipos[bid];body.iquat=m.body_iquat[bid];body.fullinertia=[np.nan,0,0,0,0,0];body.inertia=m.body_inertia[bid]
 for j in range(1,m.njnt):
  name=m.joint(j).name;item=spec.joint(name)
  if name not in keep:spec.delete(item);continue
  bid=m.jnt_bodyid[j];item.pos=R[bid].T@(anchors[j]-X[bid]);axis=np.array([1. if name.startswith('knee') else -1.,0.,0.]);item.axis=R[bid].T@axis;item.ref=float(q[m.jnt_qposadr[j]]);item.armature=.01
  # The toe pilot uses a sagittal hinge, and shoulder excursion includes a
  # modest backswing; native multi-coordinate coupling is frozen explicitly.
  if name.startswith('shoulder_elv'):item.range=[-.6,2.2]
  if name.startswith('mtp'):item.range=[-.6,.2]
 # A sagittal first controller: translation and pitch are free, while lateral
 # motion is constrained geometrically. These constraints do no sagittal work.
 spec.delete(spec.joint(m.joint(0).name));rootbody=spec.body(m.body(1).name)
 for name,kind,worldaxis in [('control_root_y',mujoco.mjtJoint.mjJNT_SLIDE,[0,1,0]),('control_root_z',mujoco.mjtJoint.mjJNT_SLIDE,[0,0,1]),('control_root_pitch',mujoco.mjtJoint.mjJNT_HINGE,[1,0,0])]:
  rootbody.add_joint(name=name,type=kind,axis=R[1].T@np.array(worldaxis),limited=False,damping=0,armature=0)
 for geom in spec.geoms:geom.contype=geom.conaffinity=0
 for i,(bid,local,p,fore) in enumerate(points):
  normal=R[bid].T@np.array([0,0,1.]);spec.body(m.body(bid).name).add_geom(name='control_sole_'+str(i),type=mujoco.mjtGeom.mjGEOM_SPHERE,pos=local+.012*normal,size=[.012,0,0],contype=1,conaffinity=1,condim=3,friction=[.8,.005,.0001],solref=[.008,1.],solimp=[.95,.99,.002,.5,2.],density=0.)
 spec.worldbody.add_geom(name='control_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[5,5,.1],contype=1,conaffinity=1,friction=[.8,.005,.0001],solref=[.008,1.])
 native=spec.compile()
 # Restore exactly the already-normalized source body inertias after compile.
 for attr in ['body_mass','body_inertia','body_iquat','body_ipos']:getattr(native,attr)[:]=getattr(m,attr)
 native.opt.gravity[:]=[0,0,-9.80665];native.opt.disableflags=0;native.opt.timestep=dt;native.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;native.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON;native.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC;native.opt.iterations=60;native.opt.tolerance=1e-10
 native.dof_damping[:]=0;native.dof_frictionloss[:]=0;native.jnt_stiffness[:]=0;native.body_gravcomp[:]=0;native.dof_armature[:3]=0
 nd=mujoco.MjData(native);mujoco.mj_setConst(native,nd)
 names=[native.joint(j).name for j in range(native.njnt) if native.joint(j).name in keep];qa=np.array([native.jnt_qposadr[native.joint(n).id] for n in names]);da=np.array([native.jnt_dofadr[native.joint(n).id] for n in names]);nd.qpos[qa]=[q[m.jnt_qposadr[b.joints[n]]] for n in names]
 kd=np.array([55 if n.startswith(('hip_flexion','knee_angle')) else 25 if n.startswith('ankle') else 100 if n.startswith('flex_extension') else 6 if n.startswith('mtp') else 12 for n in names]);native.dof_damping[da]=kd
 mujoco.mj_forward(native,nd);mujoco.mj_subtreeVel(native,nd)
 error=float(abs(nd.xpos-X).max());assert error<1e-6,error
 return r,native,nd,names,qa,da,points,dict(position_registration_error_m=error,mass_kg=float(native.body_mass.sum()),source_body_inertias_preserved=True,numerical_joint_armature_kg_m2=.01,rigid_foot=True,toe_joint_control=False,free_root_in_sagittal_plane=True,lateral_motion_constrained=True,sagittal_axes=True,articular_helper_motion_frozen=True)

def run(p,out=None,dt=.001,prepared=None,flight_cache=None):
 if flight_cache is not None:
  signature=json.dumps(dict(dt=dt,ground={k:v for k,v in p.items()if not k.startswith(('flight_','prepare_','landing_'))and k!='seconds'}),sort_keys=True)
  if 'signature' in flight_cache:assert flight_cache['signature']==signature,'Ground controls changed; clear the flight cache.'
  else:flight_cache['signature']=signature
 start_clock=time.perf_counter();r,m,d,names,qa,da,points,registration=prepared or build(dt);m.opt.timestep=dt;mujoco.mj_resetData(m,d);d.qpos[:]=m.qpos0;mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d)
 methods={'IMPLICITFAST':mujoco.mjtIntegrator.mjINT_IMPLICITFAST,'RK4':mujoco.mjtIntegrator.mjINT_RK4};m.opt.integrator=methods[p.get('integration_method','IMPLICITFAST')];m.jnt_solref[:,0]=p.get('joint_limit_time_constant_s',.02)
 energy_ledger=bool(p.get('record_energy_ledger',False));energy_bit=int(mujoco.mjtEnableBit.mjENBL_ENERGY);m.opt.enableflags=(m.opt.enableflags|energy_bit)if energy_ledger else(m.opt.enableflags&~energy_bit)
 control_geoms=np.array([gi for gi in range(m.ngeom)if m.geom(gi).name.startswith('control_')]);m.geom_solref[control_geoms,0]=p.get('contact_time_constant_s',.008);m.geom_solref[control_geoms,1]=1.;m.dof_armature[da]=p.get('joint_armature_kg_m2',.01);registration=registration.copy();registration['numerical_joint_armature_kg_m2']=p.get('joint_armature_kg_m2',.01)
 initial=d.qpos[qa].copy();C0=d.subtree_com[1].copy();pitch0=float(Rotation.from_matrix(d.xmat[1].reshape(3,3)).as_rotvec()[0]);standR=d.xmat[m.body('torso').id].reshape(3,3).copy();last_land_correction=np.zeros(len(names));raw_land_correction=last_land_correction.copy();push_correction=last_land_correction.copy();raw_push_correction=last_land_correction.copy();air_poly=None;landing_poly=None;predicted_air_duration=None;prepare_start=p['prepare_landing_s']
 foot_reference_R={s:d.xmat[m.body('calcn_'+s).id].reshape(3,3).copy()for s in ['r','l']}
 point_body_ids=np.array([item[0]for item in points],np.intp);point_local_positions=np.array([item[1]for item in points])
 kp=np.array([p['leg_kp'] if n.startswith(('hip_flexion','knee_angle')) else p['ankle_kp'] if n.startswith('ankle') else p['spine_kp'] if n.startswith('flex_extension') else 800 if n.startswith('mtp') else 120 for n in names])*p['kp_scale']
 base_damping=np.array([35 if n.startswith(('hip_flexion','knee_angle')) else 15 if n.startswith('ankle') else 70 if n.startswith('flex_extension') else 6 if n.startswith('mtp') else 10 for n in names],float);m.dof_damping[da]=base_damping
 def pose(**values):return np.array([values.get(n.rsplit('_',1)[0] if n!='flex_extension' else 'trunk',initial[i])for i,n in enumerate(names)])
 low=pose(hip_flexion=p['hip_bottom'],knee_angle=p['knee_bottom'],ankle_angle=p['ankle_bottom'],trunk=p['trunk_bottom'],shoulder_elv=p['shoulder_bottom'],elbow_flexion=p['elbow'],mtp_angle=0.)
 ready=initial.copy()
 for i,n in enumerate(names):
  if n.startswith('shoulder_elv'):ready[i]=p.get('ready_shoulder',initial[i])
 high=pose(hip_flexion=p['hip_top'],knee_angle=p['knee_top'],ankle_angle=p['ankle_top'],trunk=p['trunk_top'],shoulder_elv=p['shoulder_top'],elbow_flexion=p['elbow'],mtp_angle=-.25)
 air=pose(hip_flexion=p['flight_hip'],knee_angle=p['flight_knee'],ankle_angle=p['flight_ankle'],trunk=p['flight_trunk'],shoulder_elv=p['flight_shoulder'],elbow_flexion=p['elbow'],mtp_angle=0.)
 prepare=pose(hip_flexion=p['prepare_hip'],knee_angle=p['prepare_knee'],ankle_angle=.05,trunk=p['prepare_trunk'],shoulder_elv=p['prepare_shoulder'],elbow_flexion=.35,mtp_angle=0.)
 land=pose(hip_flexion=p['landing_hip'],knee_angle=p['landing_knee'],ankle_angle=p.get('landing_ankle',.28),trunk=p.get('landing_trunk',.15),shoulder_elv=p.get('landing_shoulder',.8),elbow_flexion=.35,mtp_angle=0.)
 crouch_end=p['ready_s']+p['crouch_s'];bottom=crouch_end+p.get('prepush_lean_s',0.);push_end=bottom+p['push_s'];takeoff=None;touchdown=None;takeoff_candidate=None;landed=False;arr={k:[]for k in ['times','qpos','qvel','qacc','body_xpos','body_xmat','com','velocity','angular_momentum','force_world_N','contact_positions_m','joint_torque_Nm','joint_power_W','actual_COP_world_m','target_angles_rad','phase']};maxroot=0.;peakF=0.;penetration=0.;fall=False;takeoff_state=None;landing_state=None;positive_work=0.;last_tau=np.zeros(len(names))
 arr['joint_net_torque_Nm']=[];arr['joint_velocity_feedback_Nm']=[];arr['joint_net_power_W']=[];arr['physics_positive_net_joint_power_W']=[];arr['physics_net_joint_power_W']=[]
 if energy_ledger:arr['physics_energy_J']=[];arr['physics_constraint_power_W']=[];arr['physics_joint_net_torque_Nm']=[]
 arr['physics_times_s']=[];arr['physics_force_world_N']=[];start_step=0;nearest_landing_contact=np.inf;initial_toe_y=min(x[2][1]for x in points);previous_target=initial.copy()
 if flight_cache and 'state' in flight_cache:
  state=flight_cache['state'];d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.time=state['time'];d.qacc_warmstart[:]=state['warmstart'];start_step=state['step'];takeoff=state['takeoff'];takeoff_state=state['takeoff_state'];positive_work=state['positive_work'];peakF=state['peakF'];penetration=state['penetration'];arr={k:[v.copy() if isinstance(v,np.ndarray) else v for v in values]for k,values in state['prefix'].items()}
 for step in range(start_step,round(p['seconds']/dt)+1):
  t=step*dt;mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d);F,_,cop=actual_contact_wrench(m,d);C=d.subtree_com[1].copy();V=d.subtree_linvel[1].copy();P=d.xpos[point_body_ids]+np.einsum('nij,nj->ni',d.xmat[point_body_ids].reshape(-1,3,3),point_local_positions);gap=float(P[:,2].min())
  if t>bottom and takeoff is None:
   if F[2]<1 and V[2]>.3 and gap>.003:
    if takeoff_candidate is None:takeoff_candidate=(t,C.copy(),V.copy(),d.subtree_angmom[1].copy())
    if t-takeoff_candidate[0]>=p.get('takeoff_confirmation_s',.012):takeoff=takeoff_candidate[0];takeoff_state=takeoff_candidate
   else:takeoff_candidate=None
  if flight_cache is not None and takeoff is not None and 'state' not in flight_cache:
   flight_cache['state']=dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),time=float(d.time),warmstart=d.qacc_warmstart.copy(),step=step,takeoff=takeoff,takeoff_state=takeoff_state,positive_work=positive_work,peakF=peakF,penetration=penetration,prefix={k:list(values)for k,values in arr.items()})
  if takeoff is not None:
   m.geom_solref[control_geoms,0]=p.get('landing_contact_time_constant_s',.02)
  if takeoff is not None and touchdown is None and t>takeoff+.06 and F[2]>30:
   touchdown=t;landing_state=(C.copy(),V.copy(),P.copy());landing_torso_pitch=float(Rotation.from_matrix(d.xmat[m.body('torso').id].reshape(3,3)@standR.T).as_rotvec()[0]);landing_root_pitch=float(Rotation.from_matrix(d.xmat[1].reshape(3,3)).as_rotvec()[0]);landed=True
  if takeoff is None:
   if t<p['ready_s']:target=initial+ease(t/max(.1,p['ready_s']-.10))*(ready-initial);phase=0
   elif t<bottom:target=ready+ease((t-p['ready_s'])/p['crouch_s'])*(low-ready);phase=1
   else:
    u=(t-bottom)/p['push_s'];target=low+ease(u)*(high-low);phase=2
    for i,n in enumerate(names):
     if n.startswith(('hip_flexion','knee_angle')):
      delay=p.get('hip_delay' if n.startswith('hip_') else 'knee_delay',0.);duration=p.get('hip_extension_duration_fraction'if n.startswith('hip_')else'knee_extension_duration_fraction',1.);target[i]=low[i]+ease((u-delay)/((1-delay)*duration))*(high[i]-low[i])
     if n.startswith(('ankle','mtp')):target[i]=low[i]+ease((u-p['ankle_delay'])/(1-p['ankle_delay']))*(high[i]-low[i])
     if n.startswith(('shoulder','elbow')):target[i]=low[i]+ease((t-bottom-p['arm_delay_s'])/p['push_s'])*(high[i]-low[i])
  elif touchdown is None:
   prepare=pose(hip_flexion=p['prepare_hip'],knee_angle=p['prepare_knee'],ankle_angle=.05,trunk=p['prepare_trunk'],shoulder_elv=p['prepare_shoulder'],elbow_flexion=.35,mtp_angle=0.)
   air_T=p.get('flight_transition_s',.25)
   if air_poly is None:
    air_start=t;air_poly=Polynomial(d.qpos[qa].copy(),d.qvel[da].copy(),0,air,0,0,air_T)
    if p.get('prepare_timing_mode')=='predicted_touchdown':
     z0=takeoff_state[1][2];vz0=takeoff_state[2][2];height=p.get('nominal_landing_COM_height_m',.75);predicted_air_duration=float((vz0+np.sqrt(max(0.,vz0*vz0+2*9.80665*(z0-height))))/9.80665);prepare_start=max(.02,predicted_air_duration-p['prepare_landing_s'])
   target=air_poly.value(min(t-air_start,air_T))
   target=target+ease((t-takeoff-prepare_start)/p.get('prepare_transition_s',.20))*(prepare-target);phase=3
  elif t<touchdown+.35:
   landing_T=p.get('landing_transition_s',.25)
   if landing_poly is None:landing_poly=Polynomial(d.qpos[qa].copy(),d.qvel[da].copy(),0,land,0,0,landing_T)
   target=landing_poly.value(min(t-touchdown,landing_T));phase=4
  else:target=land+ease((t-touchdown-.35)/1.50)*(initial-land);phase=5
  if touchdown is not None and p.get('landing_flat_foot_feedback'):
   blend=float(ease((t-touchdown)/.04))
   for i,n in enumerate(names):
    if n.startswith('ankle_angle'):
     s=n[-1];bid=m.body('calcn_'+s).id;foot_pitch=float(Rotation.from_matrix(d.xmat[bid].reshape(3,3)@foot_reference_R[s].T).as_rotvec()[0]);lo,hi=m.jnt_range[m.joint(n).id];flat=float(np.clip(d.qpos[qa[i]]+foot_pitch,lo,hi));target[i]+=blend*(flat-target[i])
  # Gravity compensation is used only while supported. Contact wrench is an
  # output, never applied externally or used as a prescribed launch impulse.
  gain=kp.copy();m.dof_damping[da]=base_damping
  if touchdown is not None:
   recovery=float(ease((t-touchdown-.35)/1.50))
   for i,n in enumerate(names):
    if n.startswith(('hip_flexion','knee_angle')):
     support=p.get('landing_support_gain_fraction',.15);damping=p.get('landing_support_damping',10.);gain[i]*=support+(1-support)*recovery;m.dof_damping[da[i]]=damping+(55-damping)*recovery
    if n.startswith('ankle'):gain[i]*=.20+.80*recovery;m.dof_damping[da[i]]=8.+17*recovery
  tau=gain*(target-d.qpos[qa])
  # Optional reference-velocity feedforward compensates the existing implicit
  # joint damping during ground preparation/propulsion. It changes internal
  # servo torques only, and defaults to zero to preserve earlier trajectories.
  if takeoff is None:tau+=p.get('target_velocity_feedforward',0.)*base_damping*(target-previous_target)/dt
  previous_target=target.copy()
  if takeoff is None and t>=bottom and p.get('push_torque_boost_Nm'):
   u=(t-bottom)/p['push_s'];pulse=float(ease(u/.15)*(1-ease((u-.70)/.30)))
   for i,n in enumerate(names):
    stem=n.rsplit('_',1)[0]
    if stem in p['push_torque_boost_Nm']:
     boost=p['push_torque_boost_Nm'][stem];jid=m.joint(n).id;lo,hi=m.jnt_range[jid];q=d.qpos[qa[i]]
     if (boost<0 and q>lo+.08)or(boost>0 and q<hi-.08):tau[i]+=pulse*boost
  if F[2]>20:
   tau+=d.qfrc_bias[da]
   # Foot-pressure balance through ankle/hip torques, with an anticipatory
   # body lean during push. All feedback torques are internal.
   centre=P[:,:2].mean(0);lean=float(ease((t-p['ready_s'])/p['crouch_s']));desired_y=C0[1]-p['lean_m']*lean
   if touchdown is not None:desired_y=centre[1]
   pushing=takeoff is None and t>=bottom
   if pushing:desired_y-=p.get('push_lean_extra_m',0.)*float(ease((t-bottom)/p['push_s']))
   velocity_gain=p.get('push_balance_velocity_gain',100.)if pushing else 100.
   balance=p['balance_gain']*(350*(C[1]-desired_y)+velocity_gain*V[1])
   pitch=float(Rotation.from_matrix(d.xmat[1].reshape(3,3)).as_rotvec()[0]);pitch_goal=pitch0+(p['pitch_bottom']-pitch0)*lean
   if t>=bottom:pitch_goal=p['pitch_bottom']+ease((t-bottom)/p['push_s'])*(p['pitch_top']-p['pitch_bottom'])
   if touchdown is not None:pitch_goal=landing_root_pitch+ease((t-touchdown)/.50)*(.10-landing_root_pitch)
   pitch_control=p['pitch_gain']*(pitch_goal-pitch)-(10 if touchdown is not None else 40)*d.qvel[2]
   for i,n in enumerate(names):
    if n.startswith('ankle_angle'):tau[i]+=balance/2
    if n.startswith('hip_flexion'):tau[i]+=pitch_control/2
  if p.get('controller')=='response' and F[2]>20 and takeoff is None:
   if step%4==0:
    pitch=float(Rotation.from_matrix(d.xmat[1].reshape(3,3)).as_rotvec()[0]);lean=float(ease((t-p['ready_s'])/p['crouch_s']));pitch_goal=pitch0+(p['pitch_bottom']-pitch0)*lean
    if t>=bottom:pitch_goal=p['pitch_bottom']+ease((t-bottom)/p['push_s'])*(p['pitch_top']-p['pitch_bottom'])
    desired_y=C0[1]-p['lean_m']*lean
    goal=np.r_[12*(desired_y-C[1])-5*V[1],80*(pitch_goal-pitch)-16*d.qvel[2],120*(target-d.qpos[qa])-20*d.qvel[da]]
    weights=np.r_[35.,8.,np.full(len(names),.35)];warm=d.qacc_warmstart.copy()
    def response(x):
     d.qfrc_applied[:]=0;d.qfrc_applied[da]=x;d.qacc_warmstart[:]=warm;mujoco.mj_forward(m,d);force,_,_=actual_contact_wrench(m,d)
     return weights*np.r_[force[1]/87,d.qacc[2],d.qacc[da]]
    x=last_tau.copy();base=response(x);A=np.column_stack([(response(x+10*np.eye(len(names))[j])-base)/10 for j in range(len(names))]);ridge=.025
    delta=np.linalg.lstsq(np.vstack([A,ridge*np.eye(len(names))]),np.r_[weights*goal-base,-ridge*x],rcond=1e-8)[0];tau=x+delta;last_tau=tau.copy()
   else:tau=last_tau.copy()
  if p.get('landing_feedback') and touchdown is not None and t>touchdown+p.get('landing_feedback_start_s',.12) and F[2]>20 and step%max(1,round(.01/dt))==0:
   # Local contact-response feedback controls braking, support and angular
   # momentum using paired internal torques. It never supplies a root force.
   channels=[]
   for stem in ['hip_flexion','knee_angle','ankle_angle','flex_extension']:
    vec=np.zeros(len(names));vec[[i for i,n in enumerate(names)if n.startswith(stem)]]=1.;channels.append(vec)
   channels=np.array(channels).T;centre=P[:,:2].mean(0);zgoal=.72+ease((t-touchdown-.35)/1.5)*(.99-.72)
   desired=np.r_[15*(centre[1]-C[1])-7*V[1],max(0.,9.80665+50*(zgoal-C[2])-12*V[2]),-8*float(d.subtree_angmom[1,0])/10]
   weights=np.array([8.,8.,3.]);warm=d.qacc_warmstart.copy()
   def contact_response(x):
    d.qfrc_applied[:]=0;d.qfrc_applied[da]=x;d.qacc_warmstart[:]=warm;mujoco.mj_forward(m,d);ff,mm,_=actual_contact_wrench(m,d)
    return weights*np.r_[ff[1]/87,ff[2]/87,(mm-np.cross(C,ff))[0]/10]
   response0=contact_response(tau);A=np.column_stack([(contact_response(tau+10*vec)-response0)/10 for vec in channels.T]);delta=np.linalg.lstsq(np.vstack([A,.06*np.eye(4)]),np.r_[weights*desired-response0,np.zeros(4)],rcond=1e-7)[0]
   # Trust region limits each local linearization step, not muscle capacity.
   trust=p.get('landing_response_trust_Nm',150.);raw_land_correction=channels@np.clip(delta,-trust,trust)
  if touchdown is not None:
   last_land_correction+=dt/(.012+dt)*(raw_land_correction-last_land_correction);tau+=last_land_correction
  if p.get('momentum_feedback') and F[2]>20 and takeoff is None:
   ankle_ids=[i for i,n in enumerate(names)if n.startswith('ankle_angle')]
   f0=F;cop0=cop
   goal=-6*float(d.subtree_angmom[1,0]);desired_cop=C[1]+(goal-C[2]*f0[1])/max(f0[2],20.);desired_cop=float(np.clip(desired_cop,P[:,1].min()+.005,P[:,1].max()-.005))
   if np.isfinite(cop0[1]):tau[ankle_ids]+=.5*(desired_cop-cop0[1])*f0[2]/len(ankle_ids)
  if p.get('propulsion_force_feedback') and takeoff is None and t>=bottom and F[2]>20:
   # A task-space push command is realized solely by paired internal torques.
   # Requested GRF is an objective of the native-contact response solve, never
   # an applied root force, impulse, or prescribed takeoff state.
   if step%max(1,round(.01/dt))==0:
    channels=[]
    for stem in ['hip_flexion','knee_angle','ankle_angle','flex_extension','shoulder_elv']:
     vec=np.zeros(len(names));vec[[i for i,n in enumerate(names)if n.startswith(stem)]]=1.;channels.append(vec)
    channels=np.array(channels).T;u=float(np.clip((t-bottom)/p['push_s'],0,1));ramp=float(ease(u/.18));desired=np.array([-p.get('push_forward_accel_m_s2',8.)*ramp,9.80665+p.get('push_up_accel_m_s2',8.)*ramp,-p.get('push_angular_feedback',6.)*float(d.subtree_angmom[1,0])/10]);weights=np.array([8.,8.,3.]);warm=d.qacc_warmstart.copy()
    def push_response(x):
     d.qfrc_applied[:]=0;d.qfrc_applied[da]=x;d.qacc_warmstart[:]=warm;mujoco.mj_forward(m,d);ff,mm,_=actual_contact_wrench(m,d)
     return weights*np.r_[ff[1]/87,ff[2]/87,(mm-np.cross(C,ff))[0]/10]
    response0=push_response(tau);A=np.column_stack([(push_response(tau+10*vec)-response0)/10 for vec in channels.T]);reg=p.get('push_response_regularization',.035);delta=np.linalg.lstsq(np.vstack([A,reg*np.eye(5)]),np.r_[weights*desired-response0,np.zeros(5)],rcond=1e-7)[0];raw_push_correction=channels@np.clip(delta,-400,400)
   push_correction+=dt/(.008+dt)*(raw_push_correction-push_correction);tau+=push_correction
  if p.get('joint_torque_limits_Nm'):
   limits=np.array([p['joint_torque_limits_Nm'][name]for name in names],float);assert np.all(limits>0)
   if p.get('joint_torque_limit_basis')=='net_virtual_PD':tau=np.clip(tau+d.qfrc_passive[da],-limits,limits)-d.qfrc_passive[da]
   else:tau=np.clip(tau,-limits,limits)
  d.qfrc_applied[:]=0;d.qfrc_applied[da]=tau;maxroot=max(maxroot,float(abs(d.qfrc_applied[:3]).max()));mujoco.mj_forward(m,d);F,_,cop=actual_contact_wrench(m,d);peakF=max(peakF,float(F[2]));positive_work+=float(np.maximum(tau*d.qvel[da],0).sum())*dt
  arr['physics_times_s'].append(t);arr['physics_force_world_N'].append(F.copy())
  # In this virtual PD rig, dof_damping implements the implicit D feedback.
  # Report its signed contribution as well as the explicit P/feedforward part.
  # Net generalized actuator torque is not an individual muscle tension.
  velocity_feedback=d.qfrc_passive[da].copy();net_tau=tau+velocity_feedback;net_power=net_tau*d.qvel[da]
  arr['physics_positive_net_joint_power_W'].append(float(np.maximum(net_power,0).sum()));arr['physics_net_joint_power_W'].append(float(net_power.sum()))
  if energy_ledger:arr['physics_energy_J'].append(d.energy.copy());arr['physics_constraint_power_W'].append(float(d.qfrc_constraint@d.qvel));arr['physics_joint_net_torque_Nm'].append(net_tau.copy())
  for ci in range(d.ncon):
   penetration=max(penetration,max(0.,-d.contact[ci].dist))
   if touchdown is not None:
    cf=np.zeros(6);mujoco.mj_contactForce(m,d,ci,cf)
    if cf[0]>1.:nearest_landing_contact=min(nearest_landing_contact,float(-d.contact[ci].pos[1]+initial_toe_y))
  if step%max(1,round(.008/dt))==0:
   mujoco.mj_subtreeVel(m,d);values=dict(times=t,qpos=d.qpos.copy(),qvel=d.qvel.copy(),qacc=d.qacc.copy(),body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),force_world_N=F.copy(),contact_positions_m=P,joint_torque_Nm=tau.copy(),joint_power_W=tau*d.qvel[da],actual_COP_world_m=cop,target_angles_rad=target.copy(),phase=phase)
   values.update(joint_net_torque_Nm=net_tau.copy(),joint_velocity_feedback_Nm=velocity_feedback.copy(),joint_net_power_W=net_power.copy())
   for k,v in values.items():arr[k].append(v)
  # Launch-only diagnostic stops after native release; it supplies no impulse
  # and is never eligible for a stable landing or performance record.
  if p.get('stop_after_takeoff_s') is not None and takeoff is not None and t>=takeoff+p['stop_after_takeoff_s']:break
  if C[2]<.30 or np.max(abs(d.qvel))>150 or not np.all(np.isfinite(d.qpos)):fall=True;break
  mujoco.mj_step(m,d)
 arr={k:np.asarray(v)for k,v in arr.items()};clearances=[]
 for body,radius in [('head',.10),('torso',.14),('pelvis',.10)]:
  bid=m.body(body).id;centres=arr['body_xpos'][:,bid]+np.einsum('nij,j->ni',arr['body_xmat'][:,bid].reshape(-1,3,3),m.body_ipos[bid]);clearances.append(float((centres[:,2]-radius).min()))
 nonfoot_clearance=min(clearances);stable=bool(landed and not fall and nonfoot_clearance>.02 and np.linalg.norm(arr['velocity'][-1])<.2 and 400<arr['force_world_N'][-1,2]<1400);report=dict(status='stable_landing' if stable else 'landing_unsettled' if landed and not fall else 'airborne_then_fall' if takeoff is not None and fall else 'airborne' if takeoff else 'no_takeoff',stable_landing=stable,minimum_head_torso_pelvis_clearance_proxy_m=nonfoot_clearance,parameters=p,registration=registration,physics_dt_s=dt,compute_seconds=time.perf_counter()-start_clock,takeoff_s=takeoff,touchdown_s=touchdown,peak_vertical_GRF_N=peakF,peak_GRF_BW=peakF/(87*9.80665),positive_internal_joint_work_J=positive_work,maximum_root_applied_force_or_torque=maxroot,maximum_contact_penetration_m=float(penetration),fell=fall,final_COM_velocity_m_s=arr['velocity'][-1].tolist(),input_launch_velocity=False,input_COM_trajectory=False,input_ground_reaction=False,forces_are_native_contact_outputs=True,muscle_tensions_computed=False)
 if takeoff_state is not None:
  ii=int(np.argmin(abs(arr['times']-takeoff)));root_pitch=Rotation.from_matrix(arr['body_xmat'][ii,1].reshape(3,3)).as_rotvec()[0];report.update(takeoff_root_pitch_rad=float(root_pitch),takeoff_COM_m=takeoff_state[1].tolist(),takeoff_velocity_m_s=takeoff_state[2].tolist(),takeoff_H_kg_m2_s=takeoff_state[3].tolist())
 if landing_state is not None:
  heel=[i for i,x in enumerate(points) if not x[3]];toe0=max(-arr['contact_positions_m'][0,:,1]);heel_land=min(-landing_state[2][heel,1]);height=max(.3,landing_state[0][2]);capture=landing_state[0][1]+landing_state[1][1]/np.sqrt(9.80665/height);ys=landing_state[2][:,1];capture_error=max(0.,ys.min()-capture,capture-ys.max());report.update(first_contact_heel_distance_m=float(heel_land-toe0),valid_distance_m=float(heel_land-toe0) if stable else None,flight_COM_distance_m=float(takeoff_state[1][1]-landing_state[0][1]),landing_torso_pitch_rad=landing_torso_pitch,landing_root_pitch_rad=landing_root_pitch,landing_capture_error_m=float(capture_error),COM_path_is_kinematically_prescribed=False,COM_balance_reference_used=True,contact_point_metric='Forward is negative world Y; nearer heel at first contact, relative to initial foremost sole point. Stable record acceptance is a separate gate.')
  report['nearest_landing_foot_contact_distance_m']=float(nearest_landing_contact)if np.isfinite(nearest_landing_contact)else None;report['valid_contact_distance_m']=float(nearest_landing_contact)if stable and np.isfinite(nearest_landing_contact)else None;report['contact_distance_scope']='Nearest native sole-ground contact after touchdown through recovery; whole-body contacts are not simulated.'
 if out:
  out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'control_motion.npz',**arr);(out/'control_summary.json').write_text(json.dumps(report,indent=2));(out/'joint_names.json').write_text(json.dumps(names));mujoco.mj_saveModel(m,str(out/'control_model.mjb'));(out/'body_names.json').write_text(json.dumps([m.body(i).name for i in range(m.nbody)]))
 report['net_actuator_work_J']={}
 for label,mask in [('ground',arr['physics_times_s']<(takeoff if takeoff is not None else p['seconds']+1)),('flight',(arr['physics_times_s']>=(takeoff if takeoff is not None else np.inf))&(arr['physics_times_s']<(touchdown if touchdown is not None else np.inf))),('landing',arr['physics_times_s']>=(touchdown if touchdown is not None else np.inf))]:
  report['net_actuator_work_J'][label]=dict(positive=float(np.trapezoid(arr['physics_positive_net_joint_power_W'][mask],arr['physics_times_s'][mask]))if mask.sum()>1 else 0.,signed=float(np.trapezoid(arr['physics_net_joint_power_W'][mask],arr['physics_times_s'][mask]))if mask.sum()>1 else 0.)
 report['torque_accounting']='Explicit P/feedforward torque and implicit velocity-feedback torque are stored separately; their signed sum is virtual net joint actuation. No anatomical passive joint damping is inferred.'
 report['prepare_timing_mode']=p.get('prepare_timing_mode','elapsed_after_takeoff');report['predicted_air_duration_s']=predicted_air_duration;report['prepare_start_after_takeoff_s']=prepare_start
 report['integration_method']=p.get('integration_method','IMPLICITFAST');report['joint_limit_time_constant_s']=p.get('joint_limit_time_constant_s',.02)
 report['positive_explicit_applied_joint_work_J']=report['positive_internal_joint_work_J'];report['positive_internal_joint_work_J']=sum(w['positive']for w in report['net_actuator_work_J'].values());report['work_accounting_version']=2
 if out:(out/'control_summary.json').write_text(json.dumps(report,indent=2))
 return report,arr

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='pilot_v1');ap.add_argument('--parameters');ap.add_argument('--parameters-file');ap.add_argument('--base');ap.add_argument('--seconds',type=float);ap.add_argument('--joint-armature',type=float);ap.add_argument('--dt',type=float,default=.0005);a=ap.parse_args();p=DEFAULT.copy()
 preset=ROOT/'scenarios/controlled_jump_pilot.json'
 if not a.base and preset.exists():p.update(json.loads(preset.read_text())['parameters'])
 if a.base:p.update(json.loads(Path(a.base).read_text())['parameters'])
 if a.parameters_file:
  data=json.loads(Path(a.parameters_file).read_text());p.update(data.get('parameters',data))
 if a.parameters:p.update(json.loads(a.parameters))
 if a.joint_armature is not None:p['joint_armature_kg_m2']=a.joint_armature
 if a.seconds is not None:p['seconds']=a.seconds
 report,_=run(p,ROOT/'results/controlled_broad_jump'/a.label,dt=a.dt);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
