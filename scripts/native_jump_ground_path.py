"""Native coordinated jump kinematics with exact foot and centroidal velocity constraints.

COM is an output. A soft horizontal task supplies a small forward approach.
Ground reaction feasibility is independently audited by the caller.
"""
import json
import numpy as np,mujoco,quadprog
from scipy.linalg import qr
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from centroidal_jump import Polynomial,s5

def solve(r,initial,foot_frames,com0,bottom,toe,to,folder,hz=1000,task_profile=None):
 b,m,d=r.b,r.m,r.d;n=r.n;ids=np.array(r.ids);names=b.ind_names
 profile=task_profile or{}
 start=initial[ids].copy();low=start.copy();finish=start.copy();end_rate=np.zeros(len(ids))
 for side in ['r','l']:
  for name,value in dict(hip_flexion=1.4,knee_angle=1.9,ankle_angle=.3,shoulder_elv=.12,elbow_flexion=.35).items():low[names.index(name+'_'+side)]=profile.get('bottom_angles',{}).get(name,value)
  for name,value,rate in [('hip_flexion',.2,-8),('knee_angle',.4,-16),('ankle_angle',-.4,-10),('shoulder_elv',1.5,2.5),('elbow_flexion',.25,0)]:
   i=names.index(name+'_'+side);finish[i]=profile.get('takeoff_angles',{}).get(name,value);end_rate[i]=profile.get('takeoff_rates',{}).get(name,rate)
 preparation=Polynomial(start,np.zeros(len(ids)),np.zeros(len(ids)),low,np.zeros(len(ids)),np.zeros(len(ids)),bottom-.25)
 propulsion=Polynomial(low,np.zeros(len(ids)),np.zeros(len(ids)),finish,end_rate,np.zeros(len(ids)),to-bottom)
 toe_poly=Polynomial(0,0,0,-.4,-2,0,to-toe)
 lean=profile.get('crouch_pitch_rad',.3);rot0=Rotation.from_quat(np.roll(initial[3:7],-1));rotlow=rot0*Rotation.from_rotvec([lean,0,0])
 forward_low=Polynomial(com0[1],0,0,com0[1]+profile.get('bottom_forward_offset_m',-.03),0,0,bottom-.25)
 forward_up=Polynomial(com0[1]+profile.get('bottom_forward_offset_m',-.03),0,0,com0[1]+profile.get('takeoff_forward_offset_m',-.12),profile.get('takeoff_forward_speed_m_s',-1.2),0,to-bottom)
 def reference(t):
  if t<=.25:return start.copy(),np.zeros(len(ids)),rot0,0.,com0[1],0.
  if t<=bottom:
   u=(t-.25)/(bottom-.25);j=preparation.value(t-.25);v=preparation.value(t-.25,1);rot=rot0*Rotation.from_rotvec([lean*s5(u),0,0]);omega=lean*30*u*u*(1-u)**2/(bottom-.25);cy=forward_low.value(t-.25)[0];vy=forward_low.value(t-.25,1)[0]
  else:
   u=(t-bottom)/(to-bottom);j=propulsion.value(t-bottom);v=propulsion.value(t-bottom,1);rot=rotlow*Rotation.from_rotvec([-lean*s5(u),0,0]);omega=-lean*30*u*u*(1-u)**2/(to-bottom);cy=forward_up.value(t-bottom)[0];vy=forward_up.value(t-bottom,1)[0]
  for side in ['r','l']:
   i=names.index('mtp_angle_'+side);j[i]=0 if t<=toe else toe_poly.value(t-toe)[0];v[i]=0 if t<=toe else toe_poly.value(t-toe,1)[0]
  return j,v,rot,omega,cy,vy
 lo=m.jnt_range[b.independent,0].copy();hi=m.jnt_range[b.independent,1].copy()
 for i,name in enumerate(names):
  if name.startswith('shoulder_elv_'):hi[i]=min(hi[i],1.85)
 weights=np.r_[np.full(3,.01),np.full(3,5.),np.ones(n-6)]
 mtp=[names.index('mtp_angle_'+s)for s in ['r','l']]
 frozen=[i for i,name in enumerate(names)if name.startswith(('pro_sup_','deviation_','flexion_'))]+mtp
 if profile.get('simple_arm_task'):
  movable=('hip_','knee_','ankle_','subtalar_','shoulder_elv_','elbow_flexion_')
  frozen=[i for i,name in enumerate(names)if not name.startswith(movable)]+mtp
 def flow(q,t):
  B,_=r.coordinates(q,np.zeros(n));b.pose(q);Jc=np.zeros((3,m.nv));AH=Jc.copy();mujoco.mj_jacSubtreeCom(m,d,Jc,1);mujoco.mj_angmomMat(m,d,AH,1)
  j,v,rot,omega,cy,vy=reference(t);R=Rotation.from_quat(np.roll(q[3:7],-1));goal=np.r_[np.zeros(3),np.array([omega,0,0])+10*(R.inv()*rot).as_rotvec(),v+25*(j-q[ids])]
  cost=np.diag(weights);objective=cost@goal
  row=(Jc@B)[1];value=vy+20*(cy-d.subtree_com[1,1]);cost+=30*np.outer(row,row);objective+=30*row*value
  E=[AH@B];rhs=[np.zeros(3)]
  for side in ['r','l']:
   bid,p,Rt=foot_frames[side][int(t>toe)];J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jacBody(m,d,J,Jr,bid);err=Rotation.from_matrix(Rt@d.xmat[bid].reshape(3,3).T).as_rotvec()
   E.extend([J@B,Jr@B]);rhs.extend([60*(p-d.xpos[bid]),60*err])
  for i in frozen:
   row=np.zeros((1,n));row[0,6+i]=1.;E.append(row);rhs.append(np.array([v[i]+60*(j[i]-q[ids[i]])]))
  E=np.vstack(E);rhs=np.concatenate(rhs);sc=np.maximum(np.linalg.norm(E,axis=1),1e-6);En=E/sc[:,None];rn=rhs/sc
  _,tri,piv=qr(En.T,mode='economic',pivoting=True);rank=int((abs(np.diag(tri))>1e-9).sum());keep=piv[:rank]
  I=np.c_[np.zeros((n-6,6)),np.eye(n-6)];C=np.vstack([En[keep],I,-I]);limit=np.r_[rn[keep],60*(lo-q[ids]),-60*(hi-q[ids])]
  u=quadprog.solve_qp(cost,objective,C.T,limit,meq=rank)[0];assert abs(E@u-rhs).max()<1e-5
  return B@u,u
 times=np.unique(np.round(np.r_[np.arange(0,to,1/hz),.25,bottom,toe,to],10));q=initial.copy();states=[q.copy()];speeds=[]
 for k,(t,tn)in enumerate(zip(times[:-1],times[1:])):
  try:
   dt=float(tn-t);v,u=flow(q,float(t));speeds.append(u.copy());mid=q.copy();mujoco.mj_integratePos(m,mid,v,dt/2);mid=b.complete(mid);vm,_=flow(mid,float((t+tn)/2));nq=q.copy();mujoco.mj_integratePos(m,nq,vm,dt);q=b.complete(nq);states.append(q.copy())
  except Exception as e:
   (folder/'native_ground_failure.json').write_text(json.dumps(dict(time_s=float(t),reason=str(e)),indent=2));raise
  if k%200==0:print('NATIVE_GROUND',k,len(times),round(float(tn),3),flush=True)
 _,u=flow(q,float(to));speeds.append(u.copy());speeds=np.array(speeds);ud=CubicSpline(times,speeds)(times[:-1],1)
 np.savez_compressed(folder/'dynamic_ground_integrated.npz',times=times,qpos=np.array(states),u=speeds,udot=ud)
 report=dict(status='native_joint_coordination_centroidal_velocity_constraints',actual_takeoff_s=to,release_events=[],COM_is_output=True,H_target_kg_m2_s=[0,0,0],ground_reaction_feasibility_audit_pending=True,task_profile=profile)
 (folder/'dynamic_ground_summary.json').write_text(json.dumps(report,indent=2));return times,np.array(states),report
