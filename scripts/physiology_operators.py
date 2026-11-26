"""Geometry/physiology operators for active+passive, GH-stability optimization.

No force history is silently modified. Glenoid orientation is estimated from
the imported scapular bone mesh and saved explicitly for independent review.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from solve_fullbody import Body,OUT,CONFIG,profile
from analyze_bar_and_joints import subtree


def spatial_force_matrices(body):
 m,d=body.m,body.d;A=np.zeros((m.nbody,3,m.nu));points=d.wrap_xpos.reshape(-1,3);objects=d.wrap_obj.ravel()
 for actuator in range(m.nu):
  tid=int(m.actuator_trnid[actuator,0]);p=int(d.ten_wrapadr[tid]);end=p+int(d.ten_wrapnum[tid]);previous=None
  for j in range(int(m.tendon_adr[tid]),int(m.tendon_adr[tid]+m.tendon_num[tid])):
   kind=int(m.wrap_type[j]);obj=int(m.wrap_objid[j])
   if kind==int(mujoco.mjtWrap.mjWRAP_SITE):count=1;bid=int(m.site_bodyid[obj])
   elif kind in [int(mujoco.mjtWrap.mjWRAP_SPHERE),int(mujoco.mjtWrap.mjWRAP_CYLINDER)]:count=2 if p<end and objects[p]==obj else 0;bid=int(m.geom_bodyid[obj])
   else:raise ValueError('Unsupported spatial tendon type')
   for _ in range(count):
    point=points[p]
    if previous is not None and previous[0]!=bid:
     other,prior=previous;delta=point-prior;length=np.linalg.norm(delta)
     if length>1e-12:
      direction=m.actuator_gear[actuator,0]*delta/length;A[other,:,actuator]+=direction;A[bid,:,actuator]-=direction
    previous=(bid,point);p+=1
  assert p==end
 return A


def estimate_glenoid_geometry(body,data):
 m=body.m;meshes=json.loads((OUT/'bone_meshes.json').read_text('utf-8'));result=[]
 body.pose(body.q0)
 for side in ['r','l']:
  bid=m.body('scapula_'+side).id;hid=m.body('humerus_'+side).id;head=data['rest_body_xpos'][hid]
  # All three native GH rotational coordinates share this actual joint anchor.
  for joint in ['elv_angle_','shoulder_elv_','shoulder_rot_']:
   assert np.linalg.norm(body.d.xanchor[m.joint(joint+side).id]-head)<1e-8
  vertices=[];faces=[]
  for item in meshes:
   if item['body']=='scapula_'+side:
    offset=len(vertices);vertices.extend(item['vertices']);faces.extend((np.array(item['faces'])+offset).tolist())
  v=np.array(vertices);f=np.array(faces);a,b,c=v[f[:,0]],v[f[:,1]],v[f[:,2]];cross=np.cross(b-a,c-a);area=np.linalg.norm(cross,axis=1);normals=cross/area[:,None];centres=(a+b+c)/3
  lateral=np.array([-1.,0,0]) if side=='r' else np.array([1.,0,0])
  distance=np.linalg.norm(centres-head,axis=1)
  mask=(distance<.04)&((normals@lateral)>.45)&(abs(centres[:,2]-head[2])<.025)
  assert mask.sum()>=12,('Glenoid patch not identified',side,int(mask.sum()))
  centre=np.average(centres[mask],axis=0,weights=area[mask]);normal=np.average(normals[mask],axis=0,weights=area[mask]);normal/=np.linalg.norm(normal)
  axis=head-centre;gap=np.linalg.norm(axis);axis/=gap
  assert .01<gap<.05 and axis@lateral>.7 and normal@axis>.8
  R=data['rest_body_xmat'][bid].reshape(3,3);position=data['rest_body_xpos'][bid]
  result.append({'side':side,'scapula_body':int(bid),'humerus_body':int(hid),'patch_face_count':int(mask.sum()),
   'glenoid_centre_rest_native_m':centre.tolist(),'head_centre_rest_native_m':head.tolist(),
   'glenoid_centre_local_native_m':(R.T@(centre-position)).tolist(),'patch_normal_world':normal.tolist(),
   'cone_axis_rest_outward_world':axis.tolist(),'centre_spacing_native_m':float(gap),
   'source':'area-weighted lateral scapular surface patch within 40 mm of model GH centre, vertical window +/-25 mm; geometry estimate requiring review'})
 return result


def build_operators():
 b=Body();m,d=b.m,b.d;data=np.load(OUT/'fullbody_motion.npz');metrics=np.load(OUT/'bar_joint_metrics.npz');geometry=estimate_glenoid_geometry(b,data)
 parts=[subtree(m,g['humerus_body']) for g in geometry]
 joint_parts=[subtree(m,m.body(name).id) for name in ['humerus_r','ulna_r','lunate_r','humerus_l','ulna_l','lunate_l']]
 arrays={k:[] for k in ['R','C','passive_N','gain_N','shoulder_force_coeff','shoulder_force_offset','joint_force_coeff','joint_force_offset','cone_axis','bar_tau_per_kg','bar_force_per_kg']}
 max_affine_error=0.;max_internal_error=0.
 for i,q in enumerate(data['qpos']):
  b.pose(q);d.qvel[:]=data['qvel_native'][i];mujoco.mj_forward(m,d);d.qacc[:]=data['qacc_native'][i]
  B=b.reduction(q);R=B.T@b.moments();C=np.concatenate([B.T@b.jacobian(bid,local)[0].T*1000 for bid,local in b.sole],axis=1)
  passive=[];gain=[]
  for a in range(m.nu):
   lr=np.ascontiguousarray(m.actuator_lengthrange[a]);p=np.ascontiguousarray(m.actuator_biasprm[a,:9]);g=np.ascontiguousarray(m.actuator_gainprm[a,:9]);length=float(d.actuator_length[a]);velocity=float(d.actuator_velocity[a]);acc0=float(m.actuator_acc0[a])
   passive.append(max(0,-mujoco.mju_muscleBias(length,lr,acc0,p))*b.length_scale**2)
   gain.append(max(0,-mujoco.mju_muscleGain(length,velocity,lr,acc0,g))*b.length_scale**2)
  F=spatial_force_matrices(b);max_internal_error=max(max_internal_error,float(abs(F.sum(0)).max()))
  S=np.array([F[indices].sum(0) for indices in parts])
  offset=metrics['shoulder_reaction_world_N'][i]+S@data['force_N'][i]
  joint_coeff=np.array([F[indices].sum(0) for indices in joint_parts])
  old_joint=np.array([metrics['shoulder_reaction_world_N'][i,0],metrics['joint_reaction_world_N'][i,0],metrics['joint_reaction_world_N'][i,1],metrics['shoulder_reaction_world_N'][i,1],metrics['joint_reaction_world_N'][i,2],metrics['joint_reaction_world_N'][i,3]])
  joint_offset=old_joint+joint_coeff@data['force_N'][i]
  max_affine_error=max(max_affine_error,float(abs(offset-S@data['force_N'][i]-metrics['shoulder_reaction_world_N'][i]).max()))
  axes=[]
  for g in geometry:
   sid=g['scapula_body'];hid=g['humerus_body'];centre=d.xpos[sid]+d.xmat[sid].reshape(3,3)@np.array(g['glenoid_centre_local_native_m'])
   axis=d.xpos[hid]-centre;axis/=np.linalg.norm(axis);axes.append(axis)
  J=sum(b.jacobian(b.grip_body[side],b.grip_local)[0] for side in ['r','l'])/2
  bar_force=metrics['bar_acceleration_m_s2'][i]+np.array([0,0,9.80665])
  for key,value in [('R',R),('C',C),('passive_N',passive),('gain_N',gain),('shoulder_force_coeff',S),('shoulder_force_offset',offset),('joint_force_coeff',joint_coeff),('joint_force_offset',joint_offset),('cone_axis',axes),('bar_tau_per_kg',B.T@J.T@bar_force),('bar_force_per_kg',bar_force)]:arrays[key].append(value)
 assert max_affine_error<1e-7 and max_internal_error<1e-8
 np.savez_compressed(OUT/'physiology_operators.npz',times=data['times'],target=data['reduced_required_generalized_force'],**arrays)
 report={'status':'operators_built','grip_width_cm':CONFIG['grip_width_m']*100,'original_bar_mass_kg':CONFIG['bar_mass_kg'],'geometry':geometry,
  'affine_reaction_reconstruction_max_error_N':max_affine_error,'internal_tendon_unit_force_sum_error':max_internal_error,
  'stability':'outward force on humerus constrained within glenoid-centre-to-head cone; default half-angle 20 degrees, inscribed 16-face approximation',
  'stability_source':'https://doi.org/10.1371/journal.pone.0295003',
  'limits':['Bone patch orientation is a generic geometry estimate, not subject-specific CT validation.','GH directional stability does not model every ligament, translation or stabilizing neural co-contraction.','Original muscle parameters retained; no latissimus floor or arbitrary strength increase.']}
 (OUT/'physiology_operators.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('PHYSIOLOGY_OPERATORS_BUILT',CONFIG['grip_width_m'],flush=True)

if __name__=='__main__':build_operators()
