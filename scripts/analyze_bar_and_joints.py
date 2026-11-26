"""Bar work/power and model wrist/elbow loads; muscles included in transmitted force.

Joint resultant: distal-segment Newton balance using allocated spatial-tendon
forces, bar force, segment gravity and COM acceleration. No cartilage stress.
Tendon force assembly follows MuJoCo's documented spatial-tendon convention.
"""
from pathlib import Path
import json,csv
import numpy as np
import mujoco
from solve_fullbody import Body,OUT,CONFIG,profile
from bar_path import forward_motion


def tendon_body_wrenches(body,forces):
 m,d=body.m,body.d;wrenches=np.zeros((m.nbody,6));points=d.wrap_xpos.reshape(-1,3);objects=d.wrap_obj.ravel()
 def add(bid,point,force):
  wrenches[bid,:3]+=force;wrenches[bid,3:]+=np.cross(point-d.xipos[bid],force)
 for actuator,F in enumerate(forces):
  if F<1e-10:continue
  tid=int(m.actuator_trnid[actuator,0]);p=int(d.ten_wrapadr[tid]);end=p+int(d.ten_wrapnum[tid]);previous=None
  for j in range(int(m.tendon_adr[tid]),int(m.tendon_adr[tid]+m.tendon_num[tid])):
   kind=int(m.wrap_type[j]);obj=int(m.wrap_objid[j])
   if kind==int(mujoco.mjtWrap.mjWRAP_SITE):count=1;bid=int(m.site_bodyid[obj])
   elif kind in [int(mujoco.mjtWrap.mjWRAP_SPHERE),int(mujoco.mjtWrap.mjWRAP_CYLINDER)]:
    count=2 if p<end and objects[p]==obj else 0;bid=int(m.geom_bodyid[obj])
   else:raise ValueError('Unsupported tendon branch type '+str(kind))
   for _ in range(count):
    point=points[p]
    if previous is not None and previous[0]!=bid:
     other,prior=previous;delta=point-prior;length=np.linalg.norm(delta)
     if length>1e-12:
      pull=F*m.actuator_gear[actuator,0]*delta/length
      add(other,prior,pull);add(bid,point,-pull)
    previous=(bid,point);p+=1
  assert p==end,(tid,p,end)
 return wrenches


def subtree(m,root):
 return [i for i in range(1,m.nbody) if root in ancestors(m,i)]


def ancestors(m,i):
 chain=[]
 while i>0:chain.append(i);i=int(m.body_parentid[i])
 return chain


def main():
 b=Body();m,d=b.m,b.d;data=np.load(OUT/'fullbody_motion.npz');meta=json.loads((OUT/'fullbody_metadata.json').read_text('utf-8'))
 s=b.length_scale;times=data['times'];custom='bar_velocity_m_s' in data
 rise=float(data['bar_height_gain_m'].max()) if custom else meta.get('bar_vertical_rise_m',float(np.diff(data['physical_grips'][[0,-1],:,2].mean(1))[0]));mass=CONFIG['bar_mass_kg'];g=9.80665
 names=['elbow_r','wrist_r','elbow_l','wrist_l'];roots=[m.body(n).id for n in ['ulna_r','lunate_r','ulna_l','lunate_l']];parts=[subtree(m,k) for k in roots]
 moment_names=['elbow_flexion_r','flexion_r','deviation_r','elbow_flexion_l','flexion_l','deviation_l']
 moment_indices=[meta['independent_coordinates'].index(name) for name in moment_names]
 shoulder_parts=[subtree(m,m.body('humerus_'+side).id) for side in ['r','l']]
 reactions=[];velocities=[];accelerations=[];bar_forces=[];work=[];powers=[];gravwork=[];kinetic=[];moments=[];external_moments=[];shoulder_reactions=[]
 nodal_error=0.;native_reaction_error=0.;internal_force_error=0.
 for i,t in enumerate(times):
  b.pose(data['qpos'][i]);d.qvel[:]=data['qvel_native'][i];mujoco.mj_forward(m,d);d.qacc[:]=data['qacc_native'][i]
  progress,v,a=profile(float(t));_,df,ddf=forward_motion(t,progress,v,a,CONFIG)
  velocity=np.array([0,-df,rise*v]);accel=np.array([0,-ddf,rise*a])
  if custom:
   progress=float(data['bar_height_gain_m'][i]/rise);velocity=data['bar_velocity_m_s'][i];accel=data['bar_acceleration_m_s2'][i]
  force_bar=mass*(accel+np.array([0,0,g]));hand_force=-force_bar/2
  ke=.5*mass*float(velocity@velocity);potential=mass*g*rise*progress
  velocities.append(velocity);accelerations.append(accel);bar_forces.append(force_bar);gravwork.append(potential);kinetic.append(ke);work.append(potential+ke);powers.append(float(force_bar@velocity))
  wrenches=tendon_body_wrenches(b,data['force_N'][i]);internal_force_error=max(internal_force_error,float(np.abs(wrenches[:,:3].sum(0)).max()))
  required=np.zeros((m.nbody,3));required_native=np.zeros_like(required)
  for bid in range(1,m.nbody):
   if m.body_mass[bid]==0:continue
   J=np.zeros((3,m.nv));Jdot=np.zeros_like(J);rot=np.zeros_like(J)
   mujoco.mj_jacBodyCom(m,d,J,rot,bid);mujoco.mj_jacDot(m,d,Jdot,rot,d.xipos[bid],bid)
   acceleration=J@d.qacc+Jdot@d.qvel
   required[bid]=m.body_mass[bid]*(s*acceleration-m.opt.gravity)
   required_native[bid]=m.body_mass[bid]*(acceleration-m.opt.gravity)
  current=[];native=[]
  for indices in parts:
   current.append((required[indices]-wrenches[indices,:3]).sum(0)-hand_force)
   native.append((required_native[indices]-wrenches[indices,:3]).sum(0)-hand_force)
  reactions.append(current)
  shoulder_reactions.append([(required[indices]-wrenches[indices,:3]).sum(0)-hand_force for indices in shoulder_parts])
  joint_centres=[d.xanchor[b.joints[n]]*s for n in ['elbow_flexion_r','flexion_r','elbow_flexion_l','flexion_l']]
  external_moments.append([np.cross(data['physical_grips'][i,k//2]-point,hand_force) for k,point in enumerate(joint_centres)])
  moments.append(data['reduced_required_generalized_force'][i,moment_indices]*s)
  if i in [0,30,60,75]:
   # Independent virtual-work check for all tendon nodal forces and moments.
   assembled=np.zeros(m.nv)
   for bid in range(1,m.nbody):mujoco.mj_applyFT(m,d,wrenches[bid,:3],wrenches[bid,3:],d.xipos[bid],bid,assembled)
   nodal_error=max(nodal_error,float(np.abs(assembled-b.moments()@data['force_N'][i]).max()))
   # Independent native MuJoCo post-constraint RNE reaction, same applied loads.
   d.xfrc_applied[:]=0;d.actuator_force[:]=-data['force_N'][i];d.qfrc_constraint[:]=0
   for side in ['r','l']:
    bid=b.grip_body[side];d.xfrc_applied[bid,:3]=hand_force
    d.xfrc_applied[bid,3:]=np.cross(b.grip(side)-d.xipos[bid],hand_force)
   mujoco.mj_rnePostConstraint(m,d)
   native_reaction_error=max(native_reaction_error,float(np.abs(d.cfrc_int[roots,3:]-native).max()))
 assert nodal_error<1e-5,('Tendon nodal virtual work',nodal_error)
 assert native_reaction_error<1e-5,('Native reaction cross-check',native_reaction_error)
 assert internal_force_error<1e-6,internal_force_error
 np.savez_compressed(OUT/'bar_joint_metrics.npz',times=times,bar_velocity_m_s=velocities,bar_acceleration_m_s2=accelerations,
  bar_force_N=bar_forces,bar_hand_work_J=work,bar_gravity_work_J=gravwork,bar_kinetic_energy_J=kinetic,bar_power_W=powers,
  bar_height_gain_m=data['bar_height_gain_m'] if custom else rise*np.array([profile(float(t))[0] for t in times]),joint_reaction_world_N=reactions,joint_net_moments_Nm=moments,
  bar_external_joint_moment_world_Nm=external_moments,shoulder_reaction_world_N=shoulder_reactions)
 reaction=np.array(reactions);powers=np.array(powers);work=np.array(work)
 integration_error=float(abs(np.trapezoid(powers,times)-work[-1]));assert integration_error<.1
 stationary=np.linalg.norm(np.array(velocities),axis=1)<1e-12
 assert np.abs(powers[stationary]).max()<1e-9
 assert np.abs(np.linalg.norm(np.array(bar_forces)[stationary],axis=1)-mass*g).max()<1e-8
 report={'status':'passed','joint_reaction_names':names,'joint_moment_coordinates':moment_names,
  'bar_mass_kg':mass,'grip_width_m':CONFIG['grip_width_m'],'bar_rise_m':rise,'bar_final_hand_work_J':float(work[-1]),
  'bar_peak_power_W':float(max(powers)),'bar_average_ascent_power_W':float(work[np.argmin(abs(times-CONFIG['ascent_s']))]/CONFIG['ascent_s']),
  'work_power_integral_error_J':integration_error,'nodal_virtual_work_max_error':nodal_error,'native_joint_reaction_max_error_N':native_reaction_error,
  'internal_tendon_force_sum_error_N':internal_force_error,'peak_joint_resultant_N':dict(zip(names,np.linalg.norm(reaction,axis=2).max(0).tolist())),
  'reaction_definition':'force exerted by proximal on distal subtree, world axes; allocated muscles, bar load, segment masses and COM accelerations included',
  'work_definition':'hands to bar, delta gravitational potential plus delta translational kinetic energy; bar rotation omitted',
  'limits':['Net hinge-coordinate moments are not a 3D joint resultant moment.','Model transmitted force is not cartilage contact pressure, ligament load, tendon stress or injury probability.',
   'Fixed fingers and reduced coupled coordinates remain prescribed; reserve torques and muscle-force-allocation assumptions affect interpretation.',
   'Static holding can have nonzero tension and transmitted force while bar mechanical power is zero; metabolic cost is not estimated.']}
 (OUT/'bar_joint_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
 with (OUT/'bar_joint_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
  writer=csv.writer(stream);writer.writerow(['time_s','bar_height_gain_m','bar_speed_m_s','hand_to_bar_work_J','gravity_work_J','kinetic_energy_J','bar_power_W','right_elbow_bar_external_moment_Nm','right_wrist_bar_external_moment_Nm','right_elbow_model_transmitted_force_N','right_wrist_model_transmitted_force_N','right_elbow_flexion_net_moment_Nm','right_wrist_flexion_net_moment_Nm'])
  for i,t in enumerate(times):writer.writerow([float(t),float(data['bar_height_gain_m'][i]) if custom else rise*profile(float(t))[0],float(np.linalg.norm(velocities[i])),float(work[i]),gravwork[i],kinetic[i],float(powers[i]),float(np.linalg.norm(external_moments[i][0])),float(np.linalg.norm(external_moments[i][1])),float(np.linalg.norm(reactions[i][0])),float(np.linalg.norm(reactions[i][1])),float(moments[i][0]),float(moments[i][1])])

if __name__=='__main__':main()
