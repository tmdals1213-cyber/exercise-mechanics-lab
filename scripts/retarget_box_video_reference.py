"""Kinematic-only native replay of a real box-jump demonstration.

Monocular pose directions guide IK; orthographic sagittal root translation is
an explicit display assumption. No force, GRF or ballistic validity is claimed.
"""
from pathlib import Path
import json,hashlib
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator,CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation,RotationSpline
from retarget_measured_jump import MeasuredJumpRig
from build_forward_jump_posthoc import right_jacobian

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'results/reference_box_jump/catalyst_tracking';OUT=ROOT/'results/reference_box_jump/catalyst_native_replay'
def main():
 OUT.mkdir(exist_ok=True);z=np.load(SOURCE/'landmarks.npz');all_t=z['source_times_s'];take=(all_t>=11.5)&(all_t<=14.1);t=all_t[take];image=z['image_landmarks_xy_z_visibility_presence'][take];raw=z['monocular_estimated_world_landmarks'][take]
 assert np.isfinite(raw).all();smooth=savgol_filter(raw,7,2,axis=0);C=np.array([[1.,0,0],[0,0,1],[0,-1,0]]);world=smooth@C.T
 right=np.median(world[:8,24]-world[:8,23],axis=0);right[2]=0;forward=np.cross([0,0,1.],right);forward/=np.linalg.norm(forward)
 yaw=-np.pi/2-np.arctan2(forward[1],forward[0]);A=Rotation.from_rotvec([0,0,yaw]).as_matrix();world=world@A.T
 rig=MeasuredJumpRig(87,1.78);b,m,d=rig.b,rig.m,rig.d;g=b.report['global_normalization_scale'];b.pose(b.q0)
 markers={};source_ids={}
 for side,offset in [('l',0),('r',1)]:
  for key,joint,source_i in [('hip','hip_flexion',23),('knee','knee_angle',25),('ankle','ankle_angle',27),('shoulder','shoulder_elv',11),('elbow','elbow_flexion',13),('wrist','flexion',15)]:
   jid=b.joints[joint+'_'+side];bid=int(m.jnt_bodyid[jid]);p=d.xanchor[jid].copy();local=d.xmat[bid].reshape(3,3).T@(p-d.xpos[bid]);markers[key+'_'+side]=(bid,local);source_ids[key+'_'+side]=source_i+offset
  sole=d.site_xpos[m.site('subject_sole_reference_'+side).id].copy()
  for key,body,dy,source_i in [('heel','calcn',.01,29),('toe','toes',-.245,31)]:
   bid=m.body(body+'_'+side).id;p=sole+np.array([0,dy*g,0]);local=d.xmat[bid].reshape(3,3).T@(p-d.xpos[bid]);markers[key+'_'+side]=(bid,local);source_ids[key+'_'+side]=source_i+offset
 edges=[(a+'_'+s,c+'_'+s)for s in ['l','r']for a,c in [('hip','knee'),('knee','ankle'),('shoulder','elbow'),('elbow','wrist'),('ankle','toe')]]+[('hip_l','hip_r'),('shoulder_l','shoulder_r'),('hip_l','shoulder_l'),('hip_r','shoulder_r')]
 selected=[i for i,n in enumerate(b.ind_names)if n in ['flex_extension','lat_bending','axial_rotation']or n.startswith(('hip_','knee_','ankle_','subtalar_','mtp_','elv_angle_','shoulder_','elbow_'))]
 qids=np.array(rig.ids)[selected];jids=np.array(b.independent)[selected];lo=np.r_[np.full(3,-.7),m.jnt_range[jids,0]+1e-6];hi=np.r_[np.full(3,.7),m.jnt_range[jids,1]-1e-6]
 for k,i in enumerate(selected):
  if b.ind_names[i].startswith('shoulder_elv_'):hi[3+k]=min(hi[3+k],1.85)
 qprevious=b.q0.copy();solutions=[];errors=[];previous_R=Rotation.from_quat(np.roll(qprevious[3:7],-1))
 for count,index in enumerate(range(0,len(t),3)):
  targets=[];weights=[]
  for a,c in edges:
   v=world[index,source_ids[c]]-world[index,source_ids[a]];targets.append(v/np.linalg.norm(v));weights.append(max(.2,float(image[index,[source_ids[a],source_ids[c]],3].min())))
  targets=np.array(targets);weights=np.array(weights);R0=previous_R;prior=qprevious[qids].copy();cache={}
  def evaluate(x):
   key=x.tobytes()
   if cache.get('key')==key:return cache['value']
   q=qprevious.copy();q[:3]=0;q[3:7]=np.roll((R0*Rotation.from_rotvec(x[:3])).as_quat(),1);q[qids]=x[3:];q=b.complete(q);B,_=rig.coordinates(q,np.zeros(rig.n));b.pose(q)
   columns=np.c_[B[:,3:6]@right_jacobian(x[:3]),B[:,6+np.array(selected)]];positions={};jacobians={}
   for name,(bid,local)in markers.items():
    J,p=b.jacobian(bid,local);positions[name]=p.copy();jacobians[name]=J@columns
   residual=[];rows=[];angles=[]
   for k,(a,c)in enumerate(edges):
    v=positions[c]-positions[a];length=np.linalg.norm(v);u=v/length;residual.extend(weights[k]*(u-targets[k]));rows.append(weights[k]*(np.eye(3)-np.outer(u,u))/length@(jacobians[c]-jacobians[a]));angles.append(np.degrees(np.arccos(np.clip(u@targets[k],-1,1))))
   residual=np.r_[residual,.004*(x[3:]-prior),.003*x[:3]];jac=np.vstack([*rows,np.c_[np.zeros((len(prior),3)),.004*np.eye(len(prior))],np.c_[.003*np.eye(3),np.zeros((3,len(prior)))]])
   cache['key']=key;cache['value']=(residual,jac,q.copy(),angles);return cache['value']
  x0=np.clip(np.r_[np.zeros(3),prior],lo+1e-7,hi-1e-7);opt=least_squares(lambda x:evaluate(x)[0],x0,jac=lambda x:evaluate(x)[1],bounds=(lo,hi),max_nfev=80,ftol=1e-7,xtol=1e-7,gtol=1e-7)
  _,_,q,angles=evaluate(opt.x);solutions.append(q);errors.append(angles);qprevious=q.copy();previous_R=Rotation.from_quat(np.roll(q[3:7],-1))
  if count%5==0:print('REFERENCE_IK',count,round(float(t[index]),2),'direction_RMS_deg',round(float(np.sqrt(np.mean(np.array(angles)**2))),2),flush=True)
 indices=np.arange(0,len(t),3);st=t[indices];solutions=np.array(solutions);times=t-t[0];joints=PchipInterpolator(st,solutions[:,rig.ids]);rots=RotationSpline(st,Rotation.from_quat(np.roll(solutions[:,3:7],-1,axis=1)))
 b.pose(b.q0);hip0=np.mean([d.xanchor[b.joints['hip_flexion_'+s]]for s in ['l','r']],axis=0);ankle0=np.mean([d.xanchor[b.joints['ankle_angle_'+s]]for s in ['l','r']],axis=0)
 observed_hip=image[:,[23,24],:2].mean(1);observed_ankle=image[:,[27,28],:2].mean(1);scale=float((hip0[2]-ankle0[2])/np.median(observed_ankle[:8,1]-observed_hip[:8,1]));delta=savgol_filter(observed_hip-observed_hip[0],7,2,axis=0);forward_x=float(forward[0]);assert abs(forward_x)>.25
 positions=[];matrices=[];qs=[];contact_points=[];vertical_shift=None
 for k,source_t in enumerate(t):
  q=b.q0.copy();q[3:7]=np.roll(rots(source_t).as_quat(),1);q[rig.ids]=joints(source_t);q[:3]=0;q=b.complete(q);b.pose(q)
  native_hip=np.mean([d.xanchor[b.joints['hip_flexion_'+s]]for s in ['l','r']],axis=0);wanted=hip0+np.array([0,-delta[k,0]*scale/forward_x,-delta[k,1]*scale]);q[:3]+=wanted-native_hip;b.pose(q)
  feet=np.array([d.xpos[markers[n][0]]+d.xmat[markers[n][0]].reshape(3,3)@markers[n][1]for n in ['heel_l','toe_l','heel_r','toe_r']])
  if vertical_shift is None:vertical_shift=-float(feet[:,2].min())
  q[2]+=vertical_shift;b.pose(q);positions.append(d.xpos.copy());matrices.append(d.xmat.copy());qs.append(q.copy());contact_points.append(feet+np.array([0,0,vertical_shift]))
 np.savez_compressed(OUT/'motion.npz',times=times,source_times_s=t,qpos=np.array(qs),body_xpos=np.array(positions),body_xmat=np.array(matrices),display_foot_points_m=np.array(contact_points))
 b.pose(b.q0);np.savez_compressed(OUT/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
 report=dict(status='kinematic_reference_review_only',source_landmarks_sha256=hashlib.sha256((SOURCE/'landmarks.npz').read_bytes()).hexdigest(),source_interval_s=[float(t[0]),float(t[-1])],source_video='https://www.youtube.com/watch?v=d2z2_rRkpAo',source_pose_is_monocular_estimate=True,physical_timing_verified=False,camera_calibrated=False,force_analysis_allowed=False,native_bone_lengths_preserved=True,source_direction_RMS_deg=float(np.sqrt(np.mean(np.array(errors)**2))),projection_assumptions='Orthographic sagittal root translation; standing hip/ankle distance gives display scale. Body direction comes from monocular estimated 3D.',joint_directions=list(edges),per_fit_direction_errors_deg=errors)
 report['error_interpretation']='Direction RMS measures agreement with monocular estimated bone directions, not accuracy against measured anatomical joint angles.'
 (OUT/'reference_retarget_report.json').write_text(json.dumps(report,indent=2));(OUT/'motion_quality.json').write_text(json.dumps(dict(status='kinematic_review_only',force_analysis_allowed=False),indent=2))
 feet=np.array(contact_points);resting=feet[-8:];platform_height=float(np.median(resting[:,:,2].mean(1)));near=float(np.median(resting[:,:,1].max(1)))+.05
 meta=dict(status='kinematic_reference_review_only',movement='box_jump_reference_replay',scale=g,body_names=[m.body(i).name for i in range(m.nbody)],phase_times_s=dict(end=float(times[-1])),source_phases_reviewed=False,force_analysis_allowed=False,display_platform_only=True,box_geometry=dict(top_height_m=platform_height,y_bounds_m=[near-.65,near]),platform_height_is_measured=False,platform_note='Illustrative platform placed from replayed final foot heights. Not calibrated source geometry and not a simulated contact.')
 (OUT/'summary.json').write_text(json.dumps(meta,indent=2));print('REFERENCE_NATIVE_REPLAY',json.dumps({k:v for k,v in report.items()if k not in ['joint_directions','per_fit_direction_errors_deg']}),flush=True)
if __name__=='__main__':main()
