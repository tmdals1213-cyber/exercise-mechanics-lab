"""Fit a complete reference sequence to image observations, not inferred 3D poses.

One shared perspective camera, fixed native bones, a sagittal root path,
temporal regularization and observed support intervals. Kinematics review only.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy import sparse
from retarget_measured_jump import MeasuredJumpRig

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'results/reference_box_jump/catalyst_tracking';OLD=ROOT/'results/reference_box_jump/catalyst_native_replay';OUT=ROOT/'results/reference_box_jump/temporal_image_fit'
def main():
 global OUT,OLD
 ap=argparse.ArgumentParser();ap.add_argument('--initial-folder');ap.add_argument('--output-folder');ap.add_argument('--reference-config');ap.add_argument('--frontal-prior',action='store_true');ap.add_argument('--max-nfev',type=int,default=65);ap.add_argument('--check-jacobian',action='store_true');args=ap.parse_args()
 cfg=json.loads((ROOT/args.reference_config).read_text())if args.reference_config else {}
 if args.initial_folder:OLD=ROOT/args.initial_folder
 if args.output_folder:OUT=ROOT/args.output_folder
 tracking_file=ROOT/cfg['tracking_file']if cfg else SOURCE/'landmarks.npz'
 OUT.mkdir(exist_ok=True);data=np.load(tracking_file);start,end=cfg.get('interval_s',[11.5,14.1]);use=(data['source_times_s']>=start)&(data['source_times_s']<=end);ts=data['source_times_s'][use];observations=data['image_landmarks_xy_z_visibility_presence'][use].copy();N=len(ts);dt=float(np.median(np.diff(ts)))
 observed_valid=np.isfinite(observations[:,:,:2]).all(-1)
 for j in range(observations.shape[1]):
  valid=observed_valid[:,j]
  if valid.sum()<2:raise RuntimeError('Insufficient reference observations for marker '+str(j))
  for axis in [0,1]:observations[:,j,axis]=np.interp(ts,ts[valid],observations[valid,j,axis])
 observations[:,:,3:]=np.nan_to_num(observations[:,:,3:]);ground_flat,ground_toe,box_toe,box_flat=cfg.get('support_boundaries_s',[12.7,12.933,13.3,13.5])
 rig=MeasuredJumpRig(87,1.78);b,m,d=rig.b,rig.m,rig.d;g=b.report['global_normalization_scale'];b.pose(b.q0);markers=[];labels=[];source_indices=[]
 for side,offset in [('l',0),('r',1)]:
  for key,joint,source_i in [('hip','hip_flexion',23),('knee','knee_angle',25),('ankle','ankle_angle',27),('shoulder','shoulder_elv',11),('elbow','elbow_flexion',13),('wrist','flexion',15)]:
   jid=b.joints[joint+'_'+side];bid=int(m.jnt_bodyid[jid]);p=d.xanchor[jid].copy();local=d.xmat[bid].reshape(3,3).T@(p-d.xpos[bid]);markers.append((bid,local));labels.append(key+'_'+side);source_indices.append(source_i+offset)
  sole=d.site_xpos[m.site('subject_sole_reference_'+side).id].copy()
  for key,body,dy,source_i in [('heel','calcn',.01,29),('toe','toes',-.245,31)]:
   bid=m.body(body+'_'+side).id;p=sole+[0,dy*g,0];local=d.xmat[bid].reshape(3,3).T@(p-d.xpos[bid]);markers.append((bid,local));labels.append(key+'_'+side);source_indices.append(source_i+offset)
 frozen=cfg.get('frozen_coordinates',{})
 selected=[i for i,n in enumerate(b.ind_names)if(n in ['flex_extension','lat_bending','axial_rotation']or n.startswith(('hip_','knee_','ankle_','subtalar_','mtp_','elv_angle_','shoulder_','elbow_')))and n not in frozen]
 qids=np.array(rig.ids)[selected];jids=np.array(b.independent)[selected];K=3+len(selected);M=len(markers);foot_ids=[i for i,n in enumerate(labels)if n.startswith(('heel_','toe_'))]
 y=observations[:,source_indices,:2];confidence=np.minimum(observations[:,source_indices,3],observations[:,source_indices,4]);weights=np.sqrt(np.maximum(confidence,.05))/4.;weights[~observed_valid[:,source_indices]]=0.
 for j,n in enumerate(labels):
  if n.startswith(('heel_','toe_')):weights[(ts>=box_toe-.1)&(ts<=box_flat),j]*=.65
 prior=np.load(OLD/'motion.npz')
 if cfg:
  warp=np.array(cfg['initial_time_warp']);prior_ts=np.interp(ts,warp[:,0],warp[:,1]);initial_q=np.array([np.interp(prior_ts,prior['source_times_s'],prior['qpos'][:,j])for j in range(prior['qpos'].shape[1])]).T;initial_q[:,3:7]/=np.linalg.norm(initial_q[:,3:7],axis=1)[:,None]
 else:
  assert np.allclose(prior['source_times_s'],ts);initial_q=prior['qpos']
 initial=np.zeros((N,K));initial[:,0]=-(initial_q[:,1]-b.q0[1]);initial[:,1]=initial_q[:,2]-b.q0[2];initial[:,2]=Rotation.from_quat(np.roll(initial_q[:,3:7],-1,axis=1)).as_rotvec()[:,0];initial[:,3:]=initial_q[:,qids]
 lower=np.r_[[-.35,-.7,-.4],m.jnt_range[jids,0]+1e-6];upper=np.r_[[1.4,1.3,1.2],m.jnt_range[jids,1]-1e-6]
 if 'root_translation_upper_m'in cfg:upper[:2]=cfg['root_translation_upper_m']
 for j,i in enumerate(selected):
  if b.ind_names[i].startswith('shoulder_elv_'):upper[3+j]=min(upper[3+j],1.85)
  limits=cfg.get('trunk_rotation_bounds_deg',{})
  if b.ind_names[i]in limits:
   bound=np.deg2rad(limits[b.ind_names[i]]);lower[3+j]=max(lower[3+j],-bound);upper[3+j]=min(upper[3+j],bound)
 initial=np.clip(initial,lower+1e-5,upper-1e-5);cache=[None]*N
 def pose(i,x):
  key=x.tobytes()
  if cache[i]is not None and cache[i][0]==key:return cache[i][1:]
  q=b.q0.copy();q[:3]+=np.array([0,-x[0],x[1]]);q[3:7]=np.roll((Rotation.from_quat(np.roll(b.q0[3:7],-1))*Rotation.from_rotvec([x[2],0,0])).as_quat(),1);q[qids]=x[3:]
  for name,value in frozen.items():q[m.jnt_qposadr[b.joints[name]]]=value
  q=b.complete(q);B,_=rig.coordinates(q,np.zeros(rig.n));b.pose(q);columns=np.c_[-B[:,1],B[:,2],B[:,3],B[:,6+np.array(selected)]];P=[];J=[]
  for bid,local in markers:
   jac,p=b.jacobian(bid,local);P.append(p.copy());J.append(jac@columns)
  comJ=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,comJ,1)
  result=(np.array(P),np.array(J),q.copy(),d.subtree_com[1].copy(),comJ@columns);cache[i]=(key,*result);return result
 def project(P,c):
  phi,elev,D,f,cx,cy=c[:6];right=np.array([np.cos(phi),np.sin(phi),0.]);up=np.array([-np.sin(phi)*np.sin(elev),np.cos(phi)*np.sin(elev),np.cos(elev)]);depth=-np.cross(right,up);delta=P-[0,0,1.];xx=delta@right;yy=delta@up;zz=D+delta@depth;assert np.min(zz)>1
  result=np.stack([cx+f*xx/zz,cy-f*yy/zz],axis=-1);jac=np.stack([f/zz[...,None]*right-f*xx[...,None]/zz[...,None]**2*depth,-f/zz[...,None]*up+f*yy[...,None]/zz[...,None]**2*depth],axis=-2);return result,jac
 width,height=cfg.get('image_size',[854,480]);camlo=np.r_[np.deg2rad([-165,0]),3.,300.,0.,0.,.15];camhi=np.r_[np.deg2rad([-15,35]),15.,3000.,width,height,cfg.get('box_height_fit_upper_m',1.05)];camera=np.array([np.deg2rad(-50),np.deg2rad(10),6.,1000.,.45*width,.58*height,.55])
 P0=np.array([pose(i,x)[0]for i,x in enumerate(initial)])
 candidates=[]
 for yaw in [-50.,-130.]:
  seed=camera.copy();seed[0]=np.deg2rad(yaw);fit=least_squares(lambda c:((project(P0,c)[0]-y)*weights[...,None]).ravel(),seed,bounds=(camlo,camhi),max_nfev=100);candidates.append(fit)
 fit=min(candidates,key=lambda f:f.cost);camera=fit.x
 if 'camera_parameters'in prior and not cfg:camera=prior['camera_parameters'].copy()
 pairs=[]
 for j,i in enumerate(selected):
  name=b.ind_names[i]
  if name.endswith('_l')and name[:-2]+'_r'in b.ind_names:
   other=b.ind_names.index(name[:-2]+'_r')
   if other in selected:pairs.append((3+j,3+selected.index(other)))
 size=N*K+7;last={};counter=0
 def evaluate(x):
  nonlocal counter
  key=x.tobytes()
  if last.get('key')==key:return last['residual'],last['jacobian']
  poses=x[:N*K].reshape(N,K);cam=x[N*K:];P=[];J=[];C=[];CJ=[]
  for i,p in enumerate(poses):
   pi,ji,_,ci,cji=pose(i,p);P.append(pi);J.append(ji);C.append(ci);CJ.append(cji)
  P=np.array(P);J=np.array(J);C=np.array(C);CJ=np.array(CJ);screen,projJ=project(P,cam);frameJ=np.einsum('nmab,nmbk->nmak',projJ,J);cameraJ=[]
  for k in range(6):
   step=1e-5 if k<3 else 1e-3;cp=cam.copy();cm=cam.copy();cp[k]+=step;cm[k]-=step;cameraJ.append((project(P,cp)[0]-project(P,cm)[0])/(2*step))
  cameraJ=np.stack(cameraJ,axis=-1);res=[];rr=[];cc=[];vv=[];row=0
  def block(value,columns,jac):
   nonlocal row
   value=np.atleast_1d(value).ravel();jac=np.asarray(jac).reshape(len(value),len(columns));res.extend(value);rindex=np.repeat(np.arange(row,row+len(value)),len(columns));cindex=np.tile(columns,len(value));rr.extend(rindex);cc.extend(cindex);vv.extend(jac.ravel());row+=len(value)
  for i in range(N):
   cols=np.r_[np.arange(i*K,(i+1)*K),np.arange(N*K,N*K+6)]
   for j in range(M):block((screen[i,j]-y[i,j])*weights[i,j],cols,np.c_[frameJ[i,j],cameraJ[i,j]]*weights[i,j])
   for j in foot_ids:
    # Support intervals are visually annotated; transition/air frames stay free.
    ground=ts[i]<=ground_flat or(ts[i]<=ground_toe and labels[j].startswith('toe_'))
    box=ts[i]>=box_flat or(ts[i]>=box_toe and labels[j].startswith('toe_'))
    if ground or box:
     target=cam[6]if box else 0.;columns=np.r_[np.arange(i*K,(i+1)*K),N*K+6];sigma_z=cfg.get('contact_z_sigma_m',.015);block([(P[i,j,2]-target)/sigma_z],columns,np.r_[J[i,j,2],-1. if box else 0.][None]/sigma_z)
   if cfg.get('arm_sagittal_sigma_m'):
    sl,sr=labels.index('shoulder_l'),labels.index('shoulder_r');delta=P[i,sl]-P[i,sr];length=np.linalg.norm(delta);normal=delta/length;normal_J=(np.eye(3)-np.outer(normal,normal))@(J[i,sl]-J[i,sr])/length
    for side in ['l','r']:
     shoulder=labels.index('shoulder_'+side)
     for key in ['elbow_','wrist_']:
      j=labels.index(key+side);v=P[i,j]-P[i,shoulder];jac=normal@(J[i,j]-J[i,shoulder])+v@normal_J;sigma=cfg['arm_sagittal_sigma_m'];block([v@normal/sigma],np.arange(i*K,(i+1)*K),jac[None]/sigma)
   for left,right in pairs:
    # Moderate bilateral regularization removes unsupported depth/asymmetry.
    sigma=.07 if args.frontal_prior else .35
    block([(poses[i,left]-poses[i,right])/sigma],[i*K+left,i*K+right],[[1/sigma,-1/sigma]])
   if args.frontal_prior:
    # Explicit near-sagittal reference assumption, not inferred 3D truth.
    # Single-view reprojection alone allowed unsupported trunk twists and
    # contralateral hip rotations that looked plausible only from its camera.
    for j,idx in enumerate(selected):
     name=b.ind_names[idx];sigma=.07 if name in ['lat_bending','axial_rotation'] else .12 if name.startswith(('hip_rotation_','subtalar_')) else .18 if name.startswith('hip_adduction_') else None
     if sigma is not None:block([poses[i,3+j]/sigma],[i*K+3+j],[[1/sigma]])
   for j in range(3,K):block([.06*(poses[i,j]-initial[i,j])],[i*K+j],[[.06]])
  accel_weights=np.r_[[.018,.018,.006],np.full(K-3,.006)]/dt**2
  if cfg.get('arm_cartesian_acceleration_weight'):
   for j,idx in enumerate(selected):
    if b.ind_names[idx].startswith('elv_angle_'):accel_weights[3+j]=0.
  for i in range(1,N-1):
   for j in range(K):block([(poses[i-1,j]-2*poses[i,j]+poses[i+1,j])*accel_weights[j]],[(i-1)*K+j,i*K+j,(i+1)*K+j],[[accel_weights[j],-2*accel_weights[j],accel_weights[j]]])
   if cfg.get('arm_cartesian_acceleration_weight'):
    for side in ['l','r']:
     shoulder=labels.index('shoulder_'+side)
     for key in ['elbow_','wrist_']:
      j=labels.index(key+side);factor=cfg['arm_cartesian_acceleration_weight']/dt**2
      v=(P[i-1,j]-P[i-1,shoulder])-2*(P[i,j]-P[i,shoulder])+(P[i+1,j]-P[i+1,shoulder]);jac=np.concatenate([J[i-1,j]-J[i-1,shoulder],-2*(J[i,j]-J[i,shoulder]),J[i+1,j]-J[i+1,shoulder]],axis=1)
      block(v*factor,np.arange((i-1)*K,(i+2)*K),jac*factor)
   if cfg.get('flight_COM_interval_s'):
    lo_f,hi_f=cfg['flight_COM_interval_s']
    if ts[i-1]>=lo_f and ts[i+1]<=hi_f:
     sigma=np.array(cfg['flight_COM_acceleration_sigma_m_s2']);acc=(C[i-1]-2*C[i]+C[i+1])/dt**2;target=np.asarray(m.opt.gravity);jac=np.concatenate([CJ[i-1],-2*CJ[i],CJ[i+1]],axis=1)/dt**2
     block((acc-target)/sigma,np.arange((i-1)*K,(i+2)*K),jac/sigma[:,None])
  if cfg.get('contact_anchor_sigma_m'):
   ground_anchor=int(np.argmin(abs(ts-cfg['ground_anchor_start_s'])));box_anchor=int(np.searchsorted(ts,box_flat));sigma=cfg['contact_anchor_sigma_m']
   for i in range(N):
    for j in foot_ids:
     anchor=ground_anchor if i>=ground_anchor and ts[i]<=(ground_toe if labels[j].startswith('toe_')else ground_flat)else box_anchor if i>=box_anchor else None
     if anchor is not None and i!=anchor:
      block((P[i,j,:2]-P[anchor,j,:2])/sigma,np.r_[np.arange(anchor*K,(anchor+1)*K),np.arange(i*K,(i+1)*K)],np.concatenate([-J[anchor,j,:2],J[i,j,:2]],axis=1)/sigma)
  for i in range(1,N):
   for j in foot_ids:
    both_ground=ts[i]<=ground_flat;both_box=ts[i-1]>=box_flat
    if(both_ground or both_box)and not cfg.get('contact_anchor_sigma_m'):
     for axis in [0,1]:block([(P[i,j,axis]-P[i-1,j,axis])/.008],np.r_[np.arange((i-1)*K,i*K),np.arange(i*K,(i+1)*K)],np.r_[-J[i-1,j,axis],J[i,j,axis]][None]/.008)
  residual=np.array(res);jac=sparse.coo_matrix((vv,(rr,cc)),shape=(row,size)).tocsr();counter+=1
  if counter%5==1:print('TEMPORAL_IMAGE_FIT',counter,'pixel_RMS',round(float(np.sqrt(np.mean((screen-y)**2))),3),'cost',round(float(np.mean(residual**2)),3),flush=True)
  last.update(key=key,residual=residual,jacobian=jac,P=P,screen=screen,poses=poses.copy(),camera=cam.copy());return residual,jac
 x0=np.r_[initial.ravel(),camera];lo=np.r_[np.tile(lower,N),camlo];hi=np.r_[np.tile(upper,N),camhi]
 if args.check_jacobian:
  _,jac=evaluate(x0);checks=[]
  probes=[K*5,K*10+1,K*(N//2)+2,K*(N//2)+3,N*K,N*K+4]
  probes.extend(K*(N//2)+3+j for j,idx in enumerate(selected)if b.ind_names[idx]in ['shoulder_elv_r','elv_angle_l','elbow_flexion_r','hip_flexion_r'])
  for col in probes:
   step=1e-6;xp=x0.copy();xm=x0.copy();xp[col]+=step;xm[col]-=step;num=(evaluate(xp)[0]-evaluate(xm)[0])/(2*step);analytic=jac[:,col].toarray().ravel();error=num-analytic;checks.append(dict(column=int(col),max_abs_error=float(abs(error).max()),relative_norm=float(np.linalg.norm(error)/max(np.linalg.norm(num),1e-8))))
  report=dict(checks=checks,passed=all(x['relative_norm']<1e-4 and x['max_abs_error']<.02 for x in checks));(OUT/'jacobian_check.json').write_text(json.dumps(report,indent=2));print('JACOBIAN_CHECK',json.dumps(report),flush=True);assert report['passed'];return
 # The optimized states are one sequence. No independent frame solutions or
 # post-hoc coordinate smoothing are substituted after this fit.
 opt=least_squares(lambda x:evaluate(x)[0],x0,jac=lambda x:evaluate(x)[1],bounds=(lo,hi),tr_solver='lsmr',x_scale='jac',loss='soft_l1',f_scale=1.,max_nfev=args.max_nfev,ftol=1e-5,xtol=1e-6,gtol=1e-5)
 evaluate(opt.x);poses=opt.x[:N*K].reshape(N,K);P=last['P'];screen=last['screen'];cam=last['camera'];qs=[];bodyp=[];bodyR=[]
 for i,p in enumerate(poses):
  _,_,q,_,_=pose(i,p);b.pose(q);qs.append(q);bodyp.append(d.xpos.copy());bodyR.append(d.xmat.copy())
 extra=dict(image_to_reference_H=data['image_to_reference_H'][use])if'image_to_reference_H'in data else{}
 np.savez_compressed(OUT/'motion.npz',times=ts-ts[0],source_times_s=ts,qpos=np.array(qs),body_xpos=np.array(bodyp),body_xmat=np.array(bodyR),fit_image_points_px=screen,reference_image_points_px=y,observed_marker_mask=observed_valid[:,source_indices],native_marker_positions_m=P,pose_parameters=poses,camera_parameters=cam,marker_names=labels,**extra)
 b.pose(b.q0);np.savez_compressed(OUT/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
 report=dict(status='temporal_image_tracking_review_pending',objective='Observed 2D marker reprojection + temporal acceleration + bilateral regularization + observed foot support; monocular 3D used only for initialization.',source='https://www.youtube.com/watch?v=d2z2_rRkpAo',source_interval_s=[float(ts[0]),float(ts[-1])],pixel_component_RMS=float(np.sqrt(np.mean((screen-y)**2))),pixel_point_RMS=float(np.sqrt(np.mean(np.sum((screen-y)**2,axis=-1)))),solver_success=bool(opt.success),solver_message=opt.message,nfev=int(opt.nfev),camera_parameters=cam.tolist(),fitted_box_height_m=float(cam[6]),box_height_independently_measured=False,physical_time_verified=False,force_analysis_allowed=False,native_bone_lengths_preserved=True,previous_replay_rejected_by_visual_review=True,source_landmarks_sha256=hashlib.sha256((SOURCE/'landmarks.npz').read_bytes()).hexdigest())
 report['near_sagittal_prior']=args.frontal_prior
 if cfg:
  report.update(source=cfg['source'],source_landmarks_sha256=hashlib.sha256(tracking_file.read_bytes()).hexdigest(),manual_support_annotation=True,reference_configuration=args.reference_config)
 report.update(frozen_coordinates=frozen,camera_motion_removed=cfg.get('camera_motion_removed',False),flight_COM_constraint=cfg.get('flight_COM_interval_s'),assumptions=cfg.get('assumption_notes',[]))
 (OUT/'tracking_report.json').write_text(json.dumps(report,indent=2));(OUT/'motion_quality.json').write_text(json.dumps(dict(status='kinematic_review_only',force_analysis_allowed=False),indent=2))
 feet=P[-8:,foot_ids];near=float(np.median(feet[:,:,1].max(1)))+.05;meta=dict(status='kinematic_reference_review_only',movement='box_jump_reference_replay',scale=g,body_names=[m.body(i).name for i in range(m.nbody)],phase_times_s=dict(end=float(ts[-1]-ts[0])),force_analysis_allowed=False,display_platform_only=True,box_geometry=dict(top_height_m=float(cam[6]),y_bounds_m=[near-.65,near]),platform_height_is_measured=False)
 (OUT/'summary.json').write_text(json.dumps(meta,indent=2));print('TEMPORAL_REFERENCE_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
