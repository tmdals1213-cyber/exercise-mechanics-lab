"""Check native force laws and independently recomputed joint reactions."""
import json
import numpy as np
import mujoco
from solve_fullbody import Body,OUT,CONFIG


def main():
 b=Body();m,d=b.m,b.d;o=np.load(OUT/'physiology_operators.npz');case=OUT/'physiology'
 s=np.load(case/'fullbody_motion.npz');v=np.load(case/'physiology_solution.npz');metrics=np.load(case/'bar_joint_metrics.npz')
 meta=json.loads((case/'fullbody_metadata.json').read_text('utf-8'))
 assert meta['strength_ceiling_enforced'] is False
 assert np.isfinite(s['force_N']).all() and s['force_N'].min()>=0
 component_error=float(abs(s['force_N']-s['active_force_N']-s['passive_force_N']).max())
 effort_error=float(abs(o['gain_N']*s['normalized_active_effort']-s['active_force_N']).max())
 assert component_error<1e-7 and effort_error<1e-7
 native_passive_error=0.;native_gain_error=0.;balance_error=0.;angles=[];compression=[]
 for i,q in enumerate(s['qpos']):
  b.pose(q);d.qvel[:]=s['qvel_native'][i];mujoco.mj_forward(m,d)
  for a in range(m.nu):
   lr=np.ascontiguousarray(m.actuator_lengthrange[a]);acc=float(m.actuator_acc0[a]);length=float(d.actuator_length[a]);velocity=float(d.actuator_velocity[a])
   passive=max(0.,-mujoco.mju_muscleBias(length,lr,acc,np.ascontiguousarray(m.actuator_biasprm[a,:9])))*b.length_scale**2
   gain=max(0.,-mujoco.mju_muscleGain(length,velocity,lr,acc,np.ascontiguousarray(m.actuator_gainprm[a,:9])))*b.length_scale**2
   native_passive_error=max(native_passive_error,abs(passive-s['passive_force_N'][i,a]))
   native_gain_error=max(native_gain_error,abs(gain-o['gain_N'][i,a]))
  residual=o['R'][i]@s['force_N'][i]+o['C'][i]@(s['contacts_N'][i].ravel()/1000)-o['target'][i]
  balance_error=max(balance_error,float(abs(residual).max()))
  for side in range(2):
   # Reaction from the separate spatial-tendon / subtree force balance analyzer.
   reaction=metrics['shoulder_reaction_world_N'][i,side];axis=o['cone_axis'][i,side]
   assert abs(np.linalg.norm(axis)-1)<1e-10
   axial=float(reaction@axis);compression.append(axial)
   angles.append(float(np.degrees(np.arccos(np.clip(axial/np.linalg.norm(reaction),-1,1)))))
 contacts=s['contacts_N'];assert contacts[:,:,2].min()>-1e-4
 assert (abs(contacts[:,:,:2])-CONFIG['ground_friction']*contacts[:,:,2,None]).max()<1e-3
 reaction_error=float(abs(metrics['shoulder_reaction_world_N']-v['shoulder_reaction_world_N']).max())
 assert native_passive_error<1e-7 and native_gain_error<1e-7 and balance_error<.01
 assert reaction_error<1e-6 and min(compression)>0 and max(angles)<20.01
 report={'status':'passed','strength_ceiling_enforced':False,'frames':len(s['times']),
  'native_passive_force_max_error_N':native_passive_error,'native_active_gain_max_error_N':native_gain_error,
  'force_component_sum_max_error_N':component_error,'active_effort_identity_max_error_N':effort_error,
  'generalized_balance_max_error':balance_error,'independent_shoulder_reaction_max_error_N':reaction_error,
  'independent_GH_max_angle_deg':max(angles),'minimum_compressive_GH_force_N':min(compression),
  'limits':'Numerical law and equilibrium checks, not experimental validation or an individual recruitment prediction.'}
 (case/'physiology_independent_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
