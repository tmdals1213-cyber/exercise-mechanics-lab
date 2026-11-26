"""Prepare continuous muscle colours and joint reactions from one force record."""
from pathlib import Path
import argparse,json,shutil,hashlib
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from retarget_measured_jump import MeasuredJumpRig
from high_flex_profile import apply
from published_patella_profile import apply as apply_patella
from build_contact_jump import contact_geometry
from mechanics_engine_adapter import sample
from muscle_mechanics import tendon_body_wrenches
from newton_euler_reference import accumulate
from tension_groups import REGISTRY,matches
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['jump','squat'],required=True);ap.add_argument('--published-leg-paths',action='store_true');ap.add_argument('--motion',required=True);ap.add_argument('--tensions',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();src=ROOT/a.motion;tp=ROOT/a.tensions;out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);dest=out/'analysis';dest.mkdir(exist_ok=True);visual=dest/'visualization';visual.mkdir(exist_ok=True);raw=dict(np.load(src/'motion.npz'));z=dict(np.load(tp/'tensions.npz'));s=json.loads((tp/'tension_summary.json').read_text());r=MeasuredJumpRig(87,1.78);apply(r);apply_patella(r)
 if a.published_leg_paths:
  from published_knee_paths_profile import apply as apply_legs
  apply_legs(r)
 b,m,d=r.b,r.m,r.d;r.contact_stand_override=raw['qpos'][0]
 if a.case=='squat':definition=json.loads((src/'analysis/contact_definition.json').read_text());mechanics_motion=dict(np.load(src/'analysis/motion.npz'))
 else:definition={'points':[{'body_id':int(x[0]),'body':m.body(int(x[0])).name} for x in contact_geometry(r)['points']]};mechanics_motion=raw
 data={k:v.copy() for k,v in raw.items()};q=mechanics_motion['qpos'];u=mechanics_motion.get('u',np.c_[mechanics_motion['qvel'][:,:6],mechanics_motion['qvel'][:,r.dofs]]);ud=np.c_[mechanics_motion['qacc'][:,:6],mechanics_motion['qacc'][:,r.dofs]];qs=[];vs=[];acs=[];xs=[];rs=[]
 for qi,ui,ai in zip(q,u,ud):
  qi=b.complete(qi);B,c=r.set_state(qi,ui);qs.append(qi.copy());vs.append(B@ui);acs.append(B@ai+c);xs.append(d.xpos.copy());rs.append(d.xmat.copy())
 prepared=dict(mechanics_motion,qpos=np.array(qs),qvel=np.array(vs),qacc=np.array(acs),u=u,body_xpos=np.array(xs),body_xmat=np.array(rs));data.update({k:v for k,v in prepared.items() if k in ['times','qpos','qvel','qacc','u','body_xpos','body_xmat','com','velocity','contact_positions_m','contact_forces_N','contact_free_moments_Nm','force_world_N']});np.savez_compressed(out/'motion.npz',**data);np.savez_compressed(dest/'motion.npz',**prepared);shutil.copy2(tp/'tensions.npz',dest/'tensions.npz');shutil.copy2(tp/'tension_summary.json',dest/'tension_summary.json');meta=json.loads((src/'summary.json').read_text());(out/'summary.json').write_text(json.dumps(meta,indent=2));b.pose(b.q0);np.savez_compressed(out/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy());names=s['names'];count=s['model_muscle_elements'];registry=REGISTRY+[('trapezius','Trapezius',('Trapezius',),'TrapeziusScapula_S'),('serratus','Serratus anterior',('Serratus',),'SerratusAnterior_I'),('rhomboids','Rhomboids',('Rhomboideus',),'Rhomboideus_I'),('levator','Levator scapulae',('Levator',),'LevatorScapulae'),('pectoralis_minor','Pectoralis minor',('PectoralisMinor',),'PectoralisMinor')];groups=[]
 for key,label,prefix,rep in registry:
  for side in ['r','l']:
   ids=[i for i,n in enumerate(names[:count]) if matches(n,prefix) and ('l' if n.endswith('_l') else 'r')==side]
   if ids:groups.append(dict(key=key,label=label,side=side,indices=ids,released=key not in ['toe_flexors','toe_extensors']))
 joints=[dict(key=key+'_'+side,joint=joint+'_'+side,label=key.title()) for key,joint in [('shoulder','shoulder_rot'),('elbow','elbow_flexion'),('wrist','flexion'),('hip','hip_flexion'),('knee','knee_angle'),('ankle','ankle_angle')] for side in ['r','l']];node_names=['pelvis']+[j['key'] for j in joints];b.pose(b.q0);rest_nodes=np.array([d.xpos[m.body('pelvis').id].copy()]+[d.xanchor[b.joints[j['joint']]].copy() for j in joints]);reactions=[]
 for k,(time,F) in enumerate(zip(z['times'],z['force_N'])):
  i=int(abs(prepared['times']-time).argmin());contacts=[(p['body_id'],prepared['contact_positions_m'][i,j],prepared['contact_forces_N'][i,j],prepared['contact_free_moments_Nm'][i,j] if 'contact_free_moments_Nm' in prepared else np.zeros(3)) for j,p in enumerate(definition['points'])];mech=sample(r,prepared['qpos'][i],prepared['qvel'][i],prepared['qacc'][i],contacts);native=tendon_body_wrenches(b,F[:m.nu]);cuts=accumulate(m.body_parentid,mech['net_wrenches'][:,:3]-native[:,:3]);reactions.append([cuts[m.jnt_bodyid[b.joints[j['joint']]]] for j in joints])
  if k%30==0:print('FULL_VISUAL_JOINTS',a.case,k,len(z['times']),flush=True)
 J=np.array(reactions);groupF=np.array([z['force_N'][:,g['indices']].sum(1) for g in groups]);v=dict(times=z['times'].tolist(),names=names[:count],groups=groups,group_tension_N=groupF.tolist(),joint_force_N=np.linalg.norm(J,axis=2).T.tolist(),joints=joints,node_names=node_names,body_names=meta['body_names'],force_display_scope='Entire motion, no blank or zero-filled phases',force_file_sha256=hashlib.sha256((dest/'tensions.npz').read_bytes()).hexdigest());(visual/'data.json').write_text(json.dumps(v,separators=(',',':')));np.savez_compressed(visual/'geometry.npz',rest_nodes=rest_nodes);np.savez_compressed(visual/'joint_reactions.npz',times=z['times'],joint_force_world_N=J);print('FULL_VISUALS_READY',a.case,len(z['times']),len(groups),flush=True)
if __name__=='__main__':main()
