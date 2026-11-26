"""Prescribed full-body OHP inverse dynamics, explicit foot reactions and muscle demand.

Native MyoSim 416-actuator geometry. Equality-linked coordinates are reduced.
Nonnegative tension + friction-limited foot forces are solved simultaneously.
Reserve generalized forces are exported, never hidden or called muscle tension.
"""
from pathlib import Path
import json,math,os,warnings
import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline,BSpline
import mujoco,myo_sim,osqp
from bar_path import forward_offset,forward_motion

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('EXERCISE_OUTPUT',str(ROOT/'results/ohp_fullbody')))
CONFIG_PATH=Path(os.environ.get('EXERCISE_SCENARIO',str(ROOT/'scenarios/ohp_fullbody.json')))
CONFIG=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
if CONFIG.get('movement','ohp')!='ohp':
 raise ValueError('This solver is for OHP. Squat comparison files are specifications; their squat-specific contact/trajectory solver is not implemented yet.')
NATIVE_HEIGHT_M=1.788053
ARM=['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion']
FINGER=['cmc_','mp_flexion','ip_flexion','mcp','pm2','md2','pm3','md3','pm4','md4','pm5','md5']

def is_finger_joint(name):
 # Substring matching wrongly treats hip_flexion as ip_flexion.
 return name.startswith(tuple(FINGER))


def profile(t,T=CONFIG['ascent_s'],ramp=CONFIG['ramp_s'],rise=1.):
 v=rise/(T-ramp)
 if t<=0:return 0.,0.,0.
 if t>=T:return rise,0.,0.
 if t>T-ramp:
  x,vy,a=profile(T-t,T,ramp,rise);return rise-x,vy,-a
 if t>=ramp:return v*(t-ramp/2),v,0.
 u=t/ramp;return v*ramp*(21*u**6-60*u**7+67.5*u**8-35*u**9+7*u**10),v*(126*u**5-420*u**6+540*u**7-315*u**8+70*u**9),v/ramp*630*u**4*(1-u)**4


class Body:
 def __init__(self):
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');spec=myo_sim.load_spec('myofullbody')
  spec.memory=32*2**20
  self.m=spec.compile();m=self.m;self.d=mujoco.MjData(m);self.q0=m.qpos0.copy()
  for side in ['r','l']:
   m.jnt_range[m.joint('elbow_flexion_'+side).id,1]=np.deg2rad(CONFIG.get('elbow_flexion_max_deg',130.))
  mass_scale=CONFIG['subject']['mass_kg']/m.body_mass.sum();m.body_mass[:]*=mass_scale;m.body_inertia[:]*=mass_scale
  self.mass_scale=mass_scale
  # Uniform similarity transformation, evaluated in native generalized
  # coordinates. Divide physical virtual-work equations by length_scale:
  # tau_equivalent = tau_gravity_native + length_scale*tau_inertia_native.
  # All muscle/external/contact moment arms share the cancelled length factor.
  self.length_scale=CONFIG['subject']['height_m']/NATIVE_HEIGHT_M
  # No native contact/equality solver forces: reactions are solved explicitly below.
  m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
  self.joints={m.joint(i).name:i for i in range(m.njnt)}
  self.couplers=[(int(m.eq_obj1id[i]),int(m.eq_obj2id[i]),m.eq_data[i,:5].copy()) for i in range(m.neq)]
  self.dependent={a for a,b,c in self.couplers}
  self.independent=[i for i in range(1,m.njnt) if i not in self.dependent and not is_finger_joint(m.joint(i).name)]
  self.ind_names=[m.joint(i).name for i in self.independent]
  self.grip_body={s:m.body('lunate_'+s).id for s in ['r','l']}
  self.grip_local=np.array([0,-.065,0])
  self.f0=m.actuator_gainprm[:,2].copy()
  auto=self.f0<=0;self.f0[auto]=m.actuator_gainprm[auto,3]/m.actuator_acc0[auto]
  self.f0=np.maximum(self.f0,1.)*self.length_scale**2
  self.names=[m.actuator(i).name for i in range(m.nu)]
  self.pose(self.q0)
  self.rest_xpos=self.d.xpos.copy();self.rest_xmat=self.d.xmat.copy()
  self.body_names=[m.body(i).name for i in range(m.nbody)]
  self.base=self.q0.copy()
  # Modest symmetric stance and a prescribed closed hand around the bar.
  for name,jid in self.joints.items():
   address=m.jnt_qposadr[jid]
   if name.startswith('hip_adduction_'):self.base[address]=-.08
   if any(k in name for k in ['mcp2_flexion','mcp3_flexion','mcp4_flexion','mcp5_flexion']):self.base[address]=.95
   if any(k in name for k in ['pm2_flexion','pm3_flexion','pm4_flexion','pm5_flexion']):self.base[address]=1.1
   if any(k in name for k in ['md2_flexion','md3_flexion','md4_flexion','md5_flexion']):self.base[address]=.8
   if name.startswith('cmc_flexion'):self.base[address]=-.35
   if name.startswith('cmc_abduction'):self.base[address]=.25
   if name.startswith(('mp_flexion','ip_flexion')):self.base[address]=-.45
  self.base[2]-=.003
  # A small fixed standing lean places both feet behind the pelvis. This is
  # static throughout the repetition, not leg drive or an enlarged support box.
  posterior=CONFIG.get('stance_posterior_shift_m',0.)
  if posterior:
   self.pose(self.base)
   for side in ['r','l']:
    foot=m.body('calcn_'+side).id;target=self.d.xpos[foot].copy();target[1]+=posterior;up=self.d.xmat[foot].reshape(3,3)[:,2].copy()
    addresses=[m.jnt_qposadr[self.joints[name+'_'+side]] for name in ['hip_flexion','ankle_angle']]
    def stance_residual(values):
     q=self.base.copy();q[addresses]=values;self.d.qpos[:]=self.complete(q);mujoco.mj_kinematics(m,self.d)
     return np.r_[(self.d.xpos[foot,1]-target[1])*50,(self.d.xmat[foot].reshape(3,3)[:,2]-up)*10]
    stance=least_squares(stance_residual,self.base[addresses],bounds=(-.2,.2),ftol=1e-11,xtol=1e-11,gtol=1e-11)
    self.base[addresses]=stance.x;self.pose(self.base)
  self.pose(self.base)
  self.neutral_head_pos=m.body_pos[m.body('head').id].copy();self.neutral_head_quat=m.body_quat[m.body('head').id].copy()
  self.head_pose_transform=np.eye(4)
  extension=CONFIG.get('head_extension_deg',0.)
  if extension:
   hid=m.body('head').id;parent=int(m.body_parentid[hid]);head_before=np.eye(4);head_before[:3,:3]=self.d.xmat[hid].reshape(3,3);head_before[:3,3]=self.d.xpos[hid]
   theta=-math.radians(extension);rotation=np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
   pivot=self.d.xpos[hid].copy();pivot[2]=CONFIG['bar_start_height_m']/self.length_scale-.01
   new_position=pivot+rotation@(self.d.xpos[hid]-pivot);new_rotation=rotation@head_before[:3,:3];parent_rotation=self.d.xmat[parent].reshape(3,3)
   m.body_pos[hid]=parent_rotation.T@(new_position-self.d.xpos[parent]);quat=np.zeros(4);mujoco.mju_mat2Quat(quat,np.ascontiguousarray((parent_rotation.T@new_rotation).ravel()));m.body_quat[hid]=quat
   self.pose(self.base);head_after=np.eye(4);head_after[:3,:3]=self.d.xmat[hid].reshape(3,3);head_after[:3,3]=self.d.xpos[hid];self.head_pose_transform=head_after@np.linalg.inv(head_before)
  self.sole=[]
  for side in ['r','l']:
   bid=m.body('calcn_'+side).id
   x=self.d.xpos[bid,0];y=self.d.xpos[bid,1]-.085
   # Include the toe support area: native toe bones extend to Y ~= -0.04 m.
   for dx,dy in [(-.037,-.16),(.037,-.16),(-.037,.095),(.037,.095)]:
    world=np.array([x+dx,y+dy,.025])
    local=self.d.xmat[bid].reshape(3,3).T@(world-self.d.xpos[bid])
    self.sole.append((bid,local))

 def complete(self,q):
  q=q.copy()
  for _ in range(3):
   for dep,ind,coef in self.couplers:
    x=q[self.m.jnt_qposadr[ind]] if ind>=0 else 0.
    q[self.m.jnt_qposadr[dep]]=sum(coef[k]*x**k for k in range(5))
  return q

 def pose(self,q):
  self.d.qpos[:]=self.complete(q);self.d.qvel[:]=0;self.d.qacc[:]=0
  mujoco.mj_forward(self.m,self.d)

 def arm(self,vals,side='r'):
  q=self.base.copy()
  for n,v in zip(ARM,vals):q[self.m.jnt_qposadr[self.joints[n+'_'+side]]]=v
  self.d.qpos[:]=self.complete(q);self.d.qvel[:]=0;self.d.qacc[:]=0
  # IK only needs body transforms; muscle wrapping is evaluated in dynamics.
  mujoco.mj_kinematics(self.m,self.d)

 def grip(self,side='r'):
  b=self.grip_body[side];return self.d.xpos[b]+self.d.xmat[b].reshape(3,3)@self.grip_local

 def axis(self,side='r'):
  return self.d.xmat[self.grip_body[side]].reshape(3,3)[:,2]

 def reduction(self,q):
  n=6+len(self.independent);B=np.zeros((self.m.nv,n));B[:6,:6]=np.eye(6)
  eps=1e-5
  for j,jid in enumerate(self.independent):
   plus=q.copy();minus=q.copy();address=self.m.jnt_qposadr[jid]
   plus[address]+=eps;minus[address]-=eps
   vector=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,vector,2*eps,self.complete(minus),self.complete(plus));B[:,6+j]=vector
  return B

 def jacobian(self,bid,local):
  point=self.d.xpos[bid]+self.d.xmat[bid].reshape(3,3)@local
  J=np.zeros((3,self.m.nv));Jr=np.zeros_like(J);mujoco.mj_jac(self.m,self.d,J,Jr,point,bid)
  return J,point

 def moments(self):
  d=self.d;m=self.m;R=np.zeros((m.nv,m.nu))
  for i in range(m.nu):
   start=d.moment_rowadr[i];count=d.moment_rownnz[i]
   R[d.moment_colind[start:start+count],i]=-d.actuator_moment[start:start+count]
  return R


def motion(body):
 cache=OUT/'fullbody_ik.npz'
 if cache.exists():
  c=np.load(cache)
  body.pose(c['base']);old_torso=body.d.xpos[body.m.body('torso').id].copy();old_rotation=body.d.xmat[body.m.body('torso').id].copy();body.pose(body.base)
  torso_same=np.allclose(old_torso,body.d.xpos[body.m.body('torso').id],atol=1e-12,rtol=0) and np.allclose(old_rotation,body.d.xmat[body.m.body('torso').id],atol=1e-12,rtol=0)
  if 'parameters' in c and len(c['parameters'])==7 and np.allclose(c['parameters'],[CONFIG['grip_width_m'],CONFIG['bar_start_height_m'],CONFIG['bar_start_forward_m'],CONFIG.get('bar_face_clearance_bump_m',0),body.length_scale,CONFIG.get('elbow_flexion_max_deg',130.),6]) and torso_same:
   return CubicSpline(c['rvalues'],c['poses']),c['start'],c['end'],float(c['error'])
 lower=np.array([-1.65,.01,-1.56,0,-1.56,-.174,-.78]);upper=np.array([2.26,3.13,2.09,np.deg2rad(CONFIG.get('elbow_flexion_max_deg',130.)),1.56,.436,.78])
 x=-.025-CONFIG['grip_width_m']/body.length_scale/2;y0=.199545
 def end_fun(v):
  q=np.array([v[0],v[1],v[2],.002,v[3],v[4],v[5]]);body.arm(q)
  g=body.grip();a=body.axis();return np.r_[(g[:2]-[x,y0])*6,a[1:]*.25,v[4:]*.0001]
 ix=[0,1,2,4,5,6]
 end=least_squares(end_fun,[1.4,2.9,0,-.7,0,.1],bounds=(lower[ix],upper[ix]),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
 qend=np.array([end.x[0],end.x[1],end.x[2],.002,end.x[3],end.x[4],end.x[5]])
 body.arm(qend);target_end=body.grip().copy();target_start=np.array([x,y0-CONFIG['bar_start_forward_m']/body.length_scale,CONFIG['bar_start_height_m']/body.length_scale])
 def residual(vals,target,reference,progress):
  body.arm(vals)
  elbow=body.d.xanchor[body.joints['elbow_flexion_r']]
  # Align elbow under the hand in the frontal plane at the bottom, fading out
  # before lockout where a straight arm necessarily has a different elbow X.
  alignment=(elbow[0]-target[0])*4*(1-progress)**2
  return np.r_[(body.grip()-target)*(1000 if progress<1e-9 else 60),body.axis()[1:],alignment,max(0,elbow[2]-target[2]+.05)*50,
   max(0,elbow[1]-y0+.08)*(1-progress)**2*100,(vals-reference)*.02]
 rvalues=np.linspace(0,1,161);poses=[];err=[]
 rack=OUT/'rack_start.json';guess=np.array(json.loads(rack.read_text())['initial_arm_coordinates_rad']) if rack.exists() else qend.copy()
 for r in rvalues[::-1]:
  t=CONFIG['ascent_s']*(1-r);s=profile(t)[0]
  target=np.array([x,y0-forward_offset(s,CONFIG,t)/body.length_scale,target_start[2]+s*(target_end[2]-target_start[2])])
  sol=least_squares(residual,guess,args=(target,guess.copy(),s),bounds=(lower,upper),max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10);q=sol.x;guess=q
  body.arm(q);err.append(np.linalg.norm(body.grip()-target));poses.append(q)
 poses=poses[::-1]
 assert max(err)<.003,max(err)
 print('Framewise IK ready',max(err),flush=True)
 # Continuation from a calibrated rack avoids the old lockout branch jump.
 # Accept this direct spline only after auditing the actual 30 Hz trajectory.
 candidate=CubicSpline(rvalues,np.array(poses));test_times=np.arange(76)/CONFIG['fps'];test_q=[];hand_error=[]
 for t in test_times:
  p,_,_=profile(t);r=max(0,1-t/CONFIG['ascent_s']);angles=candidate(r);q=body.base.copy()
  for side in ['r','l']:
   for name,value in zip(ARM,angles):q[body.m.jnt_qposadr[body.joints[name+'_'+side]]]=value
  test_q.append(body.complete(q));body.arm(angles)
  target=np.array([x,y0-forward_offset(p,CONFIG,t)/body.length_scale,target_start[2]+p*(target_end[2]-target_start[2])])
  hand_error.append(np.linalg.norm(body.grip()-target))
 test_vel=np.zeros((76,body.m.nv))
 for i in range(1,75):mujoco.mj_differentiatePos(body.m,test_vel[i],2/CONFIG['fps'],test_q[i-1],test_q[i+1])
 test_acc=np.gradient(test_vel,1/CONFIG['fps'],axis=0);test_vel[test_times>=CONFIG['ascent_s']]=0;test_acc[0]=0;test_acc[test_times>=CONFIG['ascent_s']]=0
 peak_acc=float(abs(test_acc).max());print('Continuation trajectory audit',peak_acc,max(hand_error),flush=True)
 if peak_acc<30 and max(hand_error)<.003:
  np.savez(cache,base=body.base,parameters=[CONFIG['grip_width_m'],CONFIG['bar_start_height_m'],CONFIG['bar_start_forward_m'],CONFIG.get('bar_face_clearance_bump_m',0),body.length_scale,CONFIG.get('elbow_flexion_max_deg',130.),6],rvalues=rvalues,poses=np.array(poses),start=target_start,end=target_end,error=float(max(max(err),max(hand_error))),solver_method='rack-seeded time-parameterized continuation IK with cubic interpolation; acceleration and hand-fit audit')
  return candidate,target_start,target_end,float(max(max(err),max(hand_error)))
 # Resolve redundant joint angles over the whole trajectory, rather than
 # accepting framewise branch/twist changes that create fictitious accelerations.
 nctrl=24;knots=np.r_[np.zeros(4),np.linspace(0,1,nctrl-2)[1:-1],np.ones(4)]
 basis_spline=BSpline(knots,np.eye(nctrl),3);basis=basis_spline(rvalues)
 controls=np.linalg.lstsq(basis,np.array(poses),rcond=None)[0];controls=np.clip(controls,lower+1e-7,upper-1e-7)
 controls[0]=qend;controls[-1]=poses[-1]
 samples=np.linspace(0,1,81);B=basis_spline(samples);D1=basis_spline.derivative(1)(samples);D2=basis_spline.derivative(2)(samples)
 sample_times=CONFIG['ascent_s']*(1-samples);progress=np.array([profile(t)[0] for t in sample_times])
 targets=np.c_[np.full(len(samples),x),y0-np.array([forward_offset(p,CONFIG,t) for p,t in zip(progress,sample_times)])/body.length_scale,target_start[2]+progress*(target_end[2]-target_start[2])]
 addresses=[body.m.jnt_qposadr[body.joints[n+'_r']] for n in ARM]
 def global_residual(flat):
  c=controls.copy();c[1:-1]=flat.reshape(-1,7);angles=B@c
  q=np.tile(body.base,(len(samples),1));q[:,addresses]=angles
  for _ in range(3):
   for dep,ind,coef in body.couplers:
    value=q[:,body.m.jnt_qposadr[ind]] if ind>=0 else np.zeros(len(samples))
    q[:,body.m.jnt_qposadr[dep]]=sum(coef[k]*value**k for k in range(5))
  errors=[]
  for k,pose in enumerate(q):
   body.d.qpos[:]=pose;mujoco.mj_kinematics(body.m,body.d)
   elbow=body.d.xanchor[body.joints['elbow_flexion_r']]
   errors.extend((body.grip()-targets[k])*5000)
   errors.extend(body.axis()[1:]*3)
   errors.append((elbow[0]-targets[k,0])*(1-progress[k])**2)
   errors.append(max(0,elbow[2]-targets[k,2]+.05)*50)
   errors.append(max(0,elbow[1]-y0+.08)*(1-progress[k])**2*100)
  errors.extend((D2@c*.01).ravel());errors.extend((D1@c*.01).ravel())
  errors.extend(np.minimum((D1@c)[:,3],0)*.2)
  return np.array(errors)
 # Local B-spline support makes the Jacobian sparse. Coloring avoids evaluating
 # every control separately and LSMR avoids oversized dense factorizations.
 monotonic_pattern=np.zeros((len(samples),(nctrl-2)*7));monotonic_pattern[:,3::7]=abs(D1[:,1:-1])>1e-14
 pattern=sp.vstack([sp.csr_matrix(np.kron(abs(B[:,1:-1])>1e-14,np.ones((8,7)))),
  sp.csr_matrix(np.kron(abs(D2[:,1:-1])>1e-14,np.eye(7))),
  sp.csr_matrix(np.kron(abs(D1[:,1:-1])>1e-14,np.eye(7))),sp.csr_matrix(monotonic_pattern)],format='csr')
 solution=least_squares(global_residual,controls[1:-1].ravel(),bounds=(np.tile(lower,nctrl-2),np.tile(upper,nctrl-2)),
  jac_sparsity=pattern,tr_solver='lsmr',x_scale='jac',max_nfev=250,ftol=2e-7,xtol=2e-7,gtol=2e-7)
 controls[1:-1]=solution.x.reshape(-1,7);poses=basis@controls;err=[]
 for r,pose in zip(rvalues,poses):
  t=CONFIG['ascent_s']*(1-r);progress=profile(t)[0];target=np.array([x,y0-forward_offset(progress,CONFIG,t)/body.length_scale,target_start[2]+progress*(target_end[2]-target_start[2])])
  body.arm(pose);err.append(np.linalg.norm(body.grip()-target))
 print('Global IK candidate',solution.nfev,solution.status,max(err),'at r',rvalues[np.argmax(err)],flush=True)
 assert max(err)<.003,('Smooth trajectory IK',max(err))
 print('Globally smooth IK',solution.nfev,'evaluations, max hand error',max(err),flush=True)
 np.savez(cache,base=body.base,parameters=[CONFIG['grip_width_m'],CONFIG['bar_start_height_m'],CONFIG['bar_start_forward_m'],CONFIG.get('bar_face_clearance_bump_m',0),body.length_scale,CONFIG.get('elbow_flexion_max_deg',130.),6],rvalues=rvalues,poses=np.array(poses),start=target_start,end=target_end,error=float(max(err)))
 return CubicSpline(rvalues,np.array(poses)),target_start,target_end,float(max(err))


def allocate(Am,C,target,scale,f0):
 nm=Am.shape[1];nc=C.shape[1];nr=len(target);n=nm+nc+nr
 rowscale=np.maximum(np.linalg.norm(np.c_[Am,C],axis=1),20.)
 E=np.c_[Am,C,np.eye(nr)]/rowscale[:,None]
 bound=np.zeros((nm+8+6,n));bound[:nm,:nm]=np.eye(nm)
 for k in range(8):bound[nm+k,nm+3*k+2]=1
 for j in range(6):bound[nm+8+j,nm+nc+j]=1
 friction=np.zeros((32,n))
 for k in range(8):
  for j in range(4):
   friction[k*4+j,nm+k*3+(j//2)]=1 if j%2==0 else -1
   friction[k*4+j,nm+k*3+2]=-CONFIG['ground_friction']
 matrix=sp.csc_matrix(np.r_[E,bound,friction])
 lo=np.r_[target/rowscale,np.zeros(nm+8+6),np.full(32,-np.inf)]
 hi=np.r_[target/rowscale,np.full(nm+8,np.inf),np.zeros(6),np.zeros(32)]
 penalty=np.r_[np.ones(nm),np.full(nc,.002),1e7/rowscale**2]
 solver=osqp.OSQP();solver.setup(P=sp.diags(penalty,format='csc'),q=np.zeros(n),A=matrix,l=lo,u=hi,
   verbose=False,eps_abs=2e-7,eps_rel=2e-7,max_iter=30000,polishing=True)
 result=solver.solve(raise_error=False)
 if result.x is None or not result.info.status.startswith('solved'):raise RuntimeError('Full-body support optimization: '+result.info.status)
 a=np.maximum(result.x[:nm],0);contacts=result.x[nm:nm+nc].reshape(8,3)*1000;reserve=result.x[-nr:]
 error=Am@a+C@(contacts.ravel()/1000)+reserve-target
 if max(abs(error))>.01:raise RuntimeError(f'Force balance failed: {max(abs(error))}')
 return a*f0,a,contacts,reserve,float(max(abs(error))),result.info.status


def export_anatomy_landmarks(body,qs):
 ids=[[body.m.site(f'LAT{k}_LAT{k}-P1_{side}').id for k in [1,2,3]] for side in ['r','l']]
 body.pose(body.q0);rest=np.array([body.d.site_xpos[group].mean(0) for group in ids])
 poses=[]
 for q in qs:
  body.pose(q);poses.append([body.d.site_xpos[group].mean(0).tolist() for group in ids])
 (OUT/'anatomy_landmarks.json').write_text(json.dumps({'units':'native metres','sides':['right','left'],
  'lat_insertion_rest':rest.tolist(),'lat_insertion_frames':poses,
  'reference':'mean of LAT1/LAT2/LAT3 P1 sites on proximal humerus; not elbow or forearm'},indent=2),encoding='utf-8')


def main():
 OUT.mkdir(exist_ok=True,parents=True);body=Body();m,d=body.m,body.d
 curve,start,end,ik_error=motion(body);print('Full-body IK',start,end,'err',ik_error,flush=True)
 fps=CONFIG['fps'];times=np.arange(round((CONFIG['ascent_s']+CONFIG['hold_s'])*fps)+1)/fps
 qs=[]
 for t in times:
  s,v,a=profile(t);r=max(0,1-t/CONFIG['ascent_s']);arm=curve(r);q=body.base.copy()
  for side in ['r','l']:
   for name,value in zip(ARM,arm):q[m.jnt_qposadr[body.joints[name+'_'+side]]]=value
  qs.append(body.complete(q))
 qs=np.array(qs);vel=np.zeros((len(times),m.nv));acc=np.zeros_like(vel)
 for i in range(1,len(times)-1):mujoco.mj_differentiatePos(m,vel[i],2/fps,qs[i-1],qs[i+1])
 acc=np.gradient(vel,1/fps,axis=0);vel[0]=0;vel[times>=CONFIG['ascent_s']]=0;acc[0]=0;acc[times>=CONFIG['ascent_s']]=0
 all_forces=[];all_ratios=[];all_contacts=[];all_reserves=[];xpos=[];xmat=[];grips=[];errors=[];maxreserve=[];all_targets=[];elbows=[]
 for i,t in enumerate(times):
  body.pose(qs[i]);d.qvel[:]=vel[i];mujoco.mj_forward(m,d);d.qacc[:]=acc[i]
  tau=np.zeros(m.nv);mujoco.mj_rne(m,d,1,tau)
  gravity=m.opt.gravity.copy();m.opt.gravity[:]=0
  inertial=np.zeros(m.nv);mujoco.mj_rne(m,d,1,inertial);m.opt.gravity[:]=gravity
  tau+=(body.length_scale-1)*inertial
  progress,vy,az=profile(t);rise=(end[2]-start[2])*body.length_scale
  _,_,ddf=forward_motion(t,progress,vy,az,CONFIG);bar_az=az*rise;bar_ay=-ddf
  # Forward direction is -Y. Left/right hands each carry half the bar load.
  hand_force=np.array([0,-CONFIG['bar_mass_kg']*bar_ay/2,-CONFIG['bar_mass_kg']*(9.80665+bar_az)/2])
  for side in ['r','l']:
   J,p=body.jacobian(body.grip_body[side],body.grip_local);tau-=J.T@hand_force
  B=body.reduction(qs[i]);R=body.moments();Am=B.T@R*body.f0[None,:]
  contact_matrices=[]
  for bid,local in body.sole:
   J,p=body.jacobian(bid,local);contact_matrices.append(B.T@J.T*1000)
  C=np.concatenate(contact_matrices,axis=1);target=B.T@tau
  all_targets.append(target)
  elbows.append([d.xanchor[body.joints['elbow_flexion_'+side]].copy()*body.length_scale for side in ['r','l']])
  try:force,ratio,contacts,reserve,error,status=allocate(Am,C,target,None,body.f0)
  except RuntimeError:
   zc=.025;root=qs[i,:3]
   cop=[root[0]+((zc-root[2])*target[0]-target[4])/target[2],root[1]+(target[3]+(zc-root[2])*target[1])/target[2]]
   print('SUPPORT_DIAGNOSTIC',i,float(t),'root target',target[:6].tolist(),'required COP',cop,flush=True)
   raise
  all_forces.append(force);all_ratios.append(ratio);all_contacts.append(contacts);all_reserves.append(reserve);errors.append(error)
  maxreserve.append(float(max(abs(reserve[6:]))));xpos.append(d.xpos.copy());xmat.append(d.xmat.copy());grips.append(np.array([body.grip(s) for s in ['r','l']]))
  if i%15==0:print(f'Frame {i}/75 | demand max {max(ratio):.2f} | reserve {maxreserve[-1]:.2f} Nm | {status}',flush=True)
 force=np.array(all_forces);ratio=np.array(all_ratios)
 impulse=np.zeros_like(force);impulse[1:]=np.cumsum((force[1:]+force[:-1])/2*np.diff(times)[:,None],axis=0)
 np.savez_compressed(OUT/'fullbody_motion.npz',times=times,qpos=qs,body_xpos=xpos,body_xmat=xmat,grips=grips,
  force_N=force,reference_ratio=ratio,impulse_Ns=impulse,contacts_N=all_contacts,reserve=all_reserves,
  physical_grips=np.array(grips)*body.length_scale,
  qvel_native=vel,qacc_native=acc,reduced_required_generalized_force=all_targets,physical_elbow_centres=elbows,
  rest_body_xpos=body.rest_xpos,rest_body_xmat=body.rest_xmat)
 hand_bodies=[]
 for bid in range(m.nbody):
  ancestor=bid
  while ancestor:
   if body.body_names[ancestor] in ['lunate_r','lunate_l']:hand_bodies.append(body.body_names[bid]);break
   ancestor=int(m.body_parentid[ancestor])
 metadata={'body_names':body.body_names,'hand_display_body_names':hand_bodies,'muscle_names':body.names,'reference_force_N':body.f0.tolist(),
  'subject_mass_kg':CONFIG['subject']['mass_kg'],'native_mass_scale':body.mass_scale,'target_height_m':CONFIG['subject']['height_m'],'native_visual_height_m':NATIVE_HEIGHT_M,
  'physics_length_scale':body.length_scale,'scaling_method':'uniform similarity via virtual work; gravity native, inertial terms multiplied by length scale after common moment-arm factor cancellation',
  'kinematics_method':str(np.load(OUT/'fullbody_ik.npz').get('solver_method','global cubic B-spline IK, 24 control points; forces not post-smoothed')),
  'elbow_flexion_max_deg':CONFIG.get('elbow_flexion_max_deg',130.),'bar_face_clearance_bump_m':CONFIG.get('bar_face_clearance_bump_m',0.),
  'stance_posterior_shift_m':CONFIG.get('stance_posterior_shift_m',0.),
  'head_extension_deg':CONFIG.get('head_extension_deg',0.),'head_pose_transform_native':body.head_pose_transform.tolist(),
  'bar_vertical_rise_m':float((end[2]-start[2])*body.length_scale),
  'position_arrays':'body_xpos, rest_body_xpos and grips are native metres for the render rig; physical_grips are scaled metres',
  'reference_strength_scaling':'native reference force multiplied by length_scale squared; subject mass alone does not alter reference strength',
  'model':'MyoSim 0.2.5 native MyoFullBody, 416 musculotendon actuators',
  'method':'prescribed whole-body inverse dynamics; nonnegative tension, 8 friction-constrained foot support points; equality-coordinate reduction; explicit reserve forces',
  'duration_s':float(times[-1]),'lift_s':CONFIG['ascent_s'],'ramp_s':CONFIG['ramp_s'],'bar_mass_kg':CONFIG['bar_mass_kg'],'grip_width_m':CONFIG['grip_width_m'],'max_IK_error_m':ik_error,
  'max_balance_error':max(errors),'max_nonroot_reserve':max(maxreserve),
  'limitations':['Fixed reference-Fmax allocation; no force-length-velocity, passive co-contraction, fatigue or damage prediction.',
    'Prescribed successful kinematics; demands may exceed original reference strengths.',
    'Shoulder-girdle and spinal coupling follow the native model, not subject-specific measurements.',
    'Reserve generalized forces are explicit mathematical support and must not be presented as muscle forces.',
    'Atlas muscle surfaces are semantically mapped to actuator groups; unmapped structures are grey.'],
  'independent_coordinates':['root_x','root_y','root_z','root_rx','root_ry','root_rz']+body.ind_names,
  'color_metric':'force divided by original model reference force; group force-weighted mean ratio'}
 (OUT/'fullbody_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
 # Export the native bone meshes in world rest coordinates, with body ownership.
 hid=m.body('head').id;posed_head_pos=m.body_pos[hid].copy();posed_head_quat=m.body_quat[hid].copy()
 m.body_pos[hid]=body.neutral_head_pos;m.body_quat[hid]=body.neutral_head_quat;body.pose(body.q0);meshes=[]
 for gid in range(m.ngeom):
  if m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:continue
  mid=int(m.geom_dataid[gid]);adr=m.mesh_vertadr[mid];nv=m.mesh_vertnum[mid];fa=m.mesh_faceadr[mid];nf=m.mesh_facenum[mid]
  vertices=m.mesh_vert[adr:adr+nv]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
  meshes.append({'name':m.geom(gid).name,'body':body.body_names[m.geom_bodyid[gid]],'vertices':vertices.round(6).tolist(),'faces':m.mesh_face[fa:fa+nf].tolist()})
 (OUT/'bone_meshes.json').write_text(json.dumps(meshes,separators=(',',':')),encoding='utf-8')
 m.body_pos[hid]=posed_head_pos;m.body_quat[hid]=posed_head_quat
 export_anatomy_landmarks(body,qs)
 model_dir=OUT;os.chdir(model_dir);mujoco.mj_saveModel(m,'fullbody_model.mjb',None)
 print('Saved full-body data and bone meshes',flush=True)


if __name__=='__main__':main()
