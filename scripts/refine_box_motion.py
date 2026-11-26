"""Smooth full-body landing motion, then re-solve and audit contact wrenches.

Motion-first refinement. Hand motion remains free; no strength/force cap.
Any root residual is reported and never applied as a support actuator.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco,quadprog
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from retarget_measured_jump import MeasuredJumpRig
from high_flex_profile import apply
from build_contact_jump import contact_geometry
from box_geometry_audit import segments,minimum_clearance
from centroidal_jump import s5,Polynomial
ROOT=Path(__file__).resolve().parents[1];G=9.80665

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sigma',type=float,default=.10);ap.add_argument('--label',default='yun_160_smooth_cycle_v4');args=ap.parse_args();src=ROOT/'results/athletic_box_jump/yun_160_supported_cycle_v2';z=dict(np.load(src/'motion.npz'));meta=json.loads((src/'summary.json').read_text());audit=json.loads((src/'cycle_audit.json').read_text());contact=audit['contact_s'];out=ROOT/'results/athletic_box_jump'/args.label;out.mkdir(exist_ok=True);r=MeasuredJumpRig(87,1.78);apply(r);b,m,d=r.b,r.m,r.d;n=r.n;W=87*G;r.contact_stand_override=z['qpos'][0];points=contact_geometry(r)['points'];land=np.flatnonzero(z['times']>=contact-1e-8);t=z['times'][land];rawq=z['qpos'][land].copy();dt=float(np.median(np.diff(t)));rootR=Rotation.from_quat(np.roll(rawq[:,3:7],-1,axis=1));R0=rootR[0];rv=(R0.inv()*rootR).as_rotvec();smoothed=rawq.copy();smoothed[:,:3]=gaussian_filter1d(rawq[:,:3],args.sigma/dt,axis=0,mode='nearest');smoothed[:,r.ids]=gaussian_filter1d(rawq[:,r.ids],args.sigma/dt,axis=0,mode='nearest');sr=gaussian_filter1d(rv,args.sigma/dt,axis=0,mode='nearest');smoothed[:,3:7]=np.roll((R0*Rotation.from_rotvec(sr)).as_quat(),1,axis=1)
 from build_forward_jump_posthoc import right_jacobian
 c0i=land[0];airacc=z['qacc'][c0i-1].copy();v0=z['qvel'][c0i].copy();u0=z['u'][c0i].copy();basep=CubicSpline(t,smoothed[:,:3],axis=0);basej=CubicSpline(t,smoothed[:,r.ids],axis=0);baser=CubicSpline(t,sr,axis=0);join=contact+.23;pstart=Polynomial(rawq[0,:3],v0[:3],airacc[:3],basep(join),basep(join,1),basep(join,2),.23);jstart=Polynomial(rawq[0,r.ids],v0[r.dofs],airacc[r.dofs],basej(join),basej(join,1),basej(join,2),.23);rstart=Polynomial(np.zeros(3),v0[3:6],airacc[3:6],baser(join),baser(join,1),baser(join,2),.23)
 ccurve=CubicSpline(z['times'],z['com'],axis=0);cvcurve=CubicSpline(z['times'],z['velocity'],axis=0);cjoin=contact+.20;cstart=Polynomial(z['com'][c0i],z['velocity'][c0i],[0,0,-G],ccurve(cjoin),cvcurve(cjoin),cvcurve(cjoin,1),.20)
 b.pose(z['qpos'][0]);quiet_feet={side:d.xmat[m.body('calcn_'+side).id].reshape(3,3).copy() for side in ['r','l']};r.set_state(rawq[0],u0);foot_paths={};brake=.08
 for side in ['r','l']:
  bid=m.body('calcn_'+side).id;local=m.site_pos[m.site('subject_sole_reference_'+side).id].copy();jp=np.zeros((3,m.nv));jr=jp.copy();jd=jp.copy();jrd=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);mujoco.mj_jacDot(m,d,jd,jrd,d.xpos[bid],bid);P0=d.xpos[bid].copy();Rf=d.xmat[bid].reshape(3,3).copy();Vf=jp@d.qvel;Af=jp@airacc+jd@d.qvel;_,sole=b.jacobian(bid,local);sv=b.jacobian(bid,local)[0]@d.qvel;ends=sole.copy();ends[:2]+=.5*brake*sv[:2];ends[2]=1.6;Rend=quiet_feet[side];endbody=ends-Rend@local;phiend=Rotation.from_matrix(Rf.T@Rend).as_rotvec();foot_paths[bid]=(Polynomial(P0,Vf,Af,endbody,0,0,brake),Rf,Polynomial(np.zeros(3),Rf.T@(jr@d.qvel),Rf.T@(jr@airacc+jrd@d.qvel),phiend,0,0,brake))
 def foot_target(bid,time):
  path,Rf,turn=foot_paths[bid];tt=min(time-contact,brake);return path.value(tt),Rf@Rotation.from_rotvec(turn.value(tt)).as_matrix()
 def com_target(time):return cstart.value(time-contact) if time<cjoin else ccurve(time)

 lower=[i for i,name in enumerate(b.ind_names) if name.startswith(('hip_','knee_','ankle_','subtalar_','shoulder_elv_','elbow_flexion_')) or name in ['flex_extension','lat_bending','axial_rotation']];cols=np.r_[np.arange(6),6+np.array(lower)];weights=np.r_[np.ones(3),np.full(3,4.),np.array([4. if b.ind_names[i] in ['flex_extension','lat_bending','axial_rotation'] else 2. if b.ind_names[i].startswith(('shoulder_elv_','elbow_flexion_')) else 1. for i in lower])];inv=1/weights;footids=[m.body('calcn_'+s).id for s in ['r','l']];lo=m.jnt_range[b.independent,0];hi=m.jnt_range[b.independent,1];chosen=np.unique(np.r_[np.arange(min(70,len(t))),np.arange(70,len(t),10),len(t)-1]);qs=[];ts=[];pose_error=0.;projection_failure=None;previous_base=None;previous_pose=None
 for count,k in enumerate(chosen):
  qold=rawq[k];time=t[k];targets=[foot_target(bid,time) for bid in footids];C=com_target(time);q=smoothed[k].copy();q[:3]=pstart.value(time-contact) if time<join else basep(time);q[r.ids]=jstart.value(time-contact) if time<join else basej(time);phi=rstart.value(time-contact) if time<join else baser(time);q[3:7]=np.roll((R0*Rotation.from_rotvec(phi)).as_quat(),1);q=b.complete(q);baseq=q.copy()
  if previous_base is not None:
   increment=np.zeros(m.nv);mujoco.mj_differentiatePos(m,increment,1.,previous_base,baseq);q=previous_pose.copy();mujoco.mj_integratePos(m,q,increment,1.);q=b.complete(q)
  for iteration in range(20):
   B,_=r.set_state(q,np.zeros(n));rows=[];errors=[]
   for bid,(p0,Rt) in zip(footids,targets):
    jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);rows.extend([jp@B,jr@B]);errors.extend([p0-d.xpos[bid],Rotation.from_matrix(Rt@d.xmat[bid].reshape(3,3).T).as_rotvec()])
   J=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,J,1);rows.append(J@B);errors.append(C-d.subtree_com[1]);A=np.vstack(rows)[:,cols];e=np.concatenate(errors)
   if abs(e).max()<1e-8:break
   from scipy.linalg import qr
   _,tri,piv=qr(A.T,pivoting=True,mode='economic');rank=int((abs(np.diag(tri))>1e-9).sum());keep=piv[:rank];limits=[];bounds=[]
   for j,col in enumerate(cols):
    if col<6:continue
    ii=int(col-6);row=np.zeros(len(cols));row[j]=1;limits.extend([row,-row]);bounds.extend([lo[ii]+1e-7-q[r.ids[ii]],q[r.ids[ii]]-hi[ii]+1e-7])
   try:delta=quadprog.solve_qp(np.diag(weights**2),np.zeros(len(cols)),np.vstack([A[keep],limits]).T,np.r_[e[keep],bounds],meq=rank)[0]
   except ValueError:break
   full=np.zeros(n);full[cols]=delta;mujoco.mj_integratePos(m,q,B@full,1.);q=b.complete(q)
  error=float(abs(e).max());pose_error=max(pose_error,error)
  if error>.002:projection_failure=dict(time_s=float(t[k]),maximum_pose_error=error);break
  qs.append(q.copy());ts.append(float(t[k]));previous_base=baseq.copy();previous_pose=q.copy();
  if count%30==0:print('SMOOTH_BOX_POSE',count,len(chosen),round(t[k],3),flush=True)
 if projection_failure:
  (out/'refinement_audit.json').write_text(json.dumps(dict(status='rejected_pose_projection',failure=projection_failure),indent=2));print('SMOOTH_BOX_REJECTED',projection_failure);return
 qs=np.array(qs);ts=np.array(ts);jcurve=CubicSpline(ts,qs[:,r.ids],axis=0,bc_type=((1,z['u'][land[0],6:]),(1,z['u'][land[-1],6:])));pcurve=CubicSpline(ts,qs[:,:3],axis=0,bc_type=((1,z['qvel'][land[0],:3]),(1,z['qvel'][land[-1],:3])));rotation_knots=Rotation.from_quat(np.roll(qs[:,3:7],-1,axis=1));rotation_base=rotation_knots[0];rotation_vectors=(rotation_base.inv()*rotation_knots).as_rotvec();from build_forward_jump_posthoc import right_jacobian
 rcurve=CubicSpline(ts,rotation_vectors,axis=0,bc_type=((1,z['u'][land[0],3:6]),(1,np.linalg.solve(right_jacobian(rotation_vectors[-1]),z['u'][land[-1],3:6]))))
 boundary=contact;blend_end=boundary+.010;boundary_i=int(abs(z['times']-boundary).argmin());qb=z['qpos'][boundary_i];vb=z['qvel'][boundary_i];ab=airacc;phi0=(rotation_base.inv()*Rotation.from_quat(np.roll(qb[3:7],-1))).as_rotvec();pd0=np.linalg.solve(right_jacobian(phi0),vb[3:6]);jrd0=(right_jacobian(phi0+1e-5*pd0)-right_jacobian(phi0-1e-5*pd0))/(2e-5);pdd0=np.linalg.solve(right_jacobian(phi0),ab[3:6]-jrd0@pd0);pblend=Polynomial(qb[:3],vb[:3],ab[:3],pcurve(blend_end),pcurve(blend_end,1),pcurve(blend_end,2),blend_end-boundary);jblend=Polynomial(qb[r.ids],vb[r.dofs],ab[r.dofs],jcurve(blend_end),jcurve(blend_end,1),jcurve(blend_end,2),blend_end-boundary);rblend=Polynomial(phi0,pd0,pdd0,rcurve(blend_end),rcurve(blend_end,1),rcurve(blend_end,2),blend_end-boundary)
 def state(time):
  if time<=boundary+1e-9:
   i=int(abs(z['times']-time).argmin());r.set_state(z['qpos'][i],z['u'][i]);return z['qpos'][i].copy(),z['u'][i].copy(),z['qvel'][i].copy(),airacc.copy()
  pc,jc,rc=(pblend,jblend,rblend) if time<blend_end else (pcurve,jcurve,rcurve);tt=time-boundary if time<blend_end else time
  value=lambda curve,order=0:curve.value(tt,order) if time<blend_end else curve(tt,order)
  q=qs[0].copy();q[:3]=value(pc);phi=value(rc);rate=value(rc,1);Jr=right_jacobian(phi);Jrdot=(right_jacobian(phi+1e-5*rate)-right_jacobian(phi-1e-5*rate))/(2e-5);q[3:7]=np.roll((rotation_base*Rotation.from_rotvec(phi)).as_quat(),1);q[r.ids]=value(jc);q=b.complete(q);u=np.r_[value(pc,1),Jr@rate,value(jc,1)];ud=np.r_[value(pc,2),Jr@value(rc,2)+Jrdot@rate,value(jc,2)];B,c=r.set_state(q,u);return q,u,B@u,B@ud+c
 arrays={key:[] for key in ['times','qpos','qvel','qacc','u','body_xpos','body_xmat','com','velocity','angular_momentum','contact_positions_m','contact_forces_N','force_world_N','mode','root_residual']};errors=[];clear=np.inf;maxgap=0.;prev=np.zeros(36);old_speed={};new_speed={};handids=[m.body('lunate_'+s).id for s in ['r','l']]
 for bid in handids:
  old_speed[m.body(bid).name]=float(np.linalg.norm(CubicSpline(z['times'][land],z['body_xpos'][land,bid],axis=0)(z['times'][land],1),axis=1).max())
 for k,time in enumerate(ts):
  q,u,v,acc=state(time);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);demand=(M@acc+d.qfrc_bias-d.qfrc_passive)[:6];E=[];positions=[]
  for bid,local,*_ in points:J,p=b.jacobian(bid,local);E.append(J.T[:6]);positions.append(p)
  E=np.concatenate(E,axis=1);target=demand/W;constraints=[]
  for j in range(len(points)):
   for x,y,zz in [(0,0,1)]+[(-np.cos(phi),-np.sin(phi),.8*np.cos(np.pi/16)) for phi in np.linspace(0,2*np.pi,16,endpoint=False)]:row=np.zeros(36);row[3*j:3*j+3]=[x,y,zz];constraints.append(row)
  C=np.array(constraints);sc=np.maximum(np.linalg.norm(E,axis=1),.001)
  try:f=quadprog.solve_qp(np.eye(36)*1.001,prev*.001,np.vstack([E/sc[:,None],C]).T,np.r_[target/sc,np.zeros(len(C))],meq=6)[0]
  except ValueError:
   # Kinematic differentiation can leave a small floating-root residual.
   # Choose the closest feasible frictional wrench, record every residual,
   # and accept only the declared 10% numerical tolerance. No root support.
   A=np.vstack([E[:3],E[3:]/1.78]);y=np.r_[target[:3],target[3:]/1.78];Q=np.eye(36)*.001+1e6*A.T@A;lin=1e6*A.T@y
   f=quadprog.solve_qp(Q,lin,C.T,np.zeros(len(C)))[0]
  prev=f;F=f.reshape(-1,3)*W;res=E@f*W-demand;errors.append(res);caps,rad=segments(m,d);clear=min(clear,minimum_clearance(caps,rad,-.45,1.6));maxgap=max(maxgap,max(0,1.6-float(np.min(np.array(positions)[:,2]))));mujoco.mj_subtreeVel(m,d);vals=dict(times=time,qpos=q,qvel=v,qacc=acc,u=u,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),contact_positions_m=np.array(positions),contact_forces_N=F,force_world_N=F.sum(0),mode=2,root_residual=res)
  for key,val in vals.items():arrays[key].append(val)
  if k%30==0:print('SMOOTH_BOX_CONTACT',k,len(ts),'res',round(float(abs(res).max()),2),flush=True)
 landout={key:np.array(vals) for key,vals in arrays.items()}
 for bid in handids:new_speed[m.body(bid).name]=float(np.linalg.norm(CubicSpline(ts,landout['body_xpos'][:,bid],axis=0)(ts,1),axis=1).max())
 error=np.array(errors);rforce=float(abs(error[:,:3]).max());rmoment=float(abs(error[:,3:]).max());improvement=1-max(new_speed.values())/max(old_speed.values());passed=bool(clear>=0 and maxgap<.003 and rforce<.10*W and rmoment<.10*W*1.78 and improvement>.25);report=dict(status='accepted_smooth_motion_diagnostic' if passed else 'rejected_smooth_motion_diagnostic',passed=passed,sigma_s=args.sigma,contact_sliding_brake_s=brake,contact_model='Finite 80 ms material-sole settling, Coulomb cone; no instantaneous foot pin.',pose_projection_error_max=pose_error,root_force_residual_max_N=rforce,root_moment_residual_max_Nm=rmoment,root_residual_applied_as_support=False,force_clipping=False,body_capsule_clearance_m=clear,sole_penetration_m=maxgap,old_peak_world_hand_speed_m_s=old_speed,new_peak_world_hand_speed_m_s=new_speed,maximum_hand_speed_reduction=improvement,scope='Motion-first smoothing followed by inverse contact mechanics; residuals are diagnostic, not hidden actuators.')
 before=z['times']<contact-1e-8;data={}
 for key,arr in landout.items():
  prior=z[key][before] if key in z else np.zeros((int(before.sum()),*arr.shape[1:]));data[key]=np.concatenate([prior,arr])
 np.savez_compressed(out/'motion.npz',**data);(out/'reference_pose.npz').write_bytes((src/'reference_pose.npz').read_bytes());meta['phase_times_s']['end']=float(ts[-1]);meta['refined_motion']=report;meta['force_analysis_allowed']=False;(out/'summary.json').write_text(json.dumps(meta,indent=2));(out/'refinement_audit.json').write_text(json.dumps(report,indent=2));(out/'cycle_audit.json').write_text(json.dumps(dict(audit,refinement=report),indent=2));print('SMOOTH_BOX_RESULT',json.dumps(report),flush=True)
if __name__=='__main__':main()
