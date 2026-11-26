"""Time-reverse kinematics, not forces; rebuild descent FLV and check RNE."""
import json,shutil
import numpy as np
import mujoco
from solve_fullbody import Body,OUT,CONFIG

def main():
 b=Body();m,d=b.m,b.d;source=np.load(OUT/'fullbody_motion.npz');old=np.load(OUT/'physiology_operators.npz');metrics=np.load(OUT/'bar_joint_metrics.npz')
 # 2 s up, .3 s at top, 2 s down, .3 s at bottom; endpoint included.
 indices=np.r_[np.arange(61),np.full(9,75),np.arange(59,-1,-1),np.zeros(9,dtype=int)].astype(int)
 direction=np.r_[np.ones(61),np.zeros(9),-np.ones(60),np.zeros(9)];times=np.arange(len(indices))/30
 values={k:(source[k][indices].copy() if source[k].shape[0]==len(source['times']) else source[k].copy()) for k in source.files}
 values['times']=times;values['qvel_native']*=direction[:,None]
 values['phase_code']=np.r_[np.zeros(61),np.ones(9),np.full(60,2),np.full(9,3)].astype(int)
 values['bar_velocity_m_s']=metrics['bar_velocity_m_s'][indices]*direction[:,None]
 values['bar_acceleration_m_s2']=metrics['bar_acceleration_m_s2'][indices]
 values['bar_height_gain_m']=metrics['bar_height_gain_m'][indices]
 operators={k:old[k][indices].copy() for k in old.files};operators['times']=times
 max_target_error=0.
 for i,q in enumerate(values['qpos']):
  b.pose(q);d.qvel[:]=values['qvel_native'][i];mujoco.mj_forward(m,d);d.qacc[:]=values['qacc_native'][i]
  tau=np.zeros(m.nv);mujoco.mj_rne(m,d,1,tau);gravity=m.opt.gravity.copy();m.opt.gravity[:]=0
  inertia=np.zeros(m.nv);mujoco.mj_rne(m,d,1,inertia);m.opt.gravity[:]=gravity;tau+=(b.length_scale-1)*inertia
  hand=-CONFIG['bar_mass_kg']*(values['bar_acceleration_m_s2'][i]+np.array([0,0,9.80665]))/2
  for side in ['r','l']:tau-=b.jacobian(b.grip_body[side],b.grip_local)[0].T@hand
  target=b.reduction(q).T@tau;max_target_error=max(max_target_error,float(abs(target-operators['target'][i]).max()))
  values['reduced_required_generalized_force'][i]=target;operators['target'][i]=target
  for a in range(m.nu):
   operators['gain_N'][i,a]=max(0.,-mujoco.mju_muscleGain(float(d.actuator_length[a]),float(d.actuator_velocity[a]),
    np.ascontiguousarray(m.actuator_lengthrange[a]),float(m.actuator_acc0[a]),np.ascontiguousarray(m.actuator_gainprm[a,:9])))*b.length_scale**2
 assert max_target_error<1e-7,max_target_error
 destination=OUT/'cycle';destination.mkdir(exist_ok=True)
 np.savez_compressed(destination/'fullbody_motion.npz',**values);np.savez_compressed(destination/'physiology_operators.npz',**operators)
 meta=json.loads((OUT/'fullbody_metadata.json').read_text('utf-8'));meta.update(duration_s=float(times[-1]),cycle_duration_s=float(times[-1]))
 (destination/'fullbody_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 landmarks=json.loads((OUT/'anatomy_landmarks.json').read_text('utf-8'))
 landmarks['lat_insertion_frames']=[landmarks['lat_insertion_frames'][i] for i in indices]
 (destination/'anatomy_landmarks.json').write_text(json.dumps(landmarks),encoding='utf-8')
 shutil.copyfile(OUT/'bone_meshes.json',destination/'bone_meshes.json');shutil.copyfile(OUT/'physiology_operators.json',destination/'physiology_operators.json')
 gain_change=float(abs(operators['gain_N'][70:130]-old['gain_N'][indices[70:130]]).max())
 assert gain_change>1.,'Descent force-velocity law was not recomputed'
 report={'status':'passed','duration_s':float(times[-1]),'frames':len(times),'source_indices':indices.tolist(),
  'phase_intervals_s':{'ascent':[0,2],'top_hold':[2,2.3],'descent':[2.3,4.3],'bottom_hold':[4.3,4.6]},
  'recomputed_RNE_target_max_difference':max_target_error,'descent_active_gain_max_change_N':gain_change,
  'method':'Same spatial path, reversed generalized velocity, unchanged time-reversed acceleration; active FLV gain recomputed. Forces solved separately. Replays do not simulate fatigue.'}
 (destination/'cycle_preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('CYCLE_PREPARED',len(times),gain_change,flush=True)

if __name__=='__main__':main()
