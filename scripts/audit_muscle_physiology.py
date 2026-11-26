"""Audit imported force-length-velocity/passive parameters without changing SO.

These are outputs of the imported muscle law, not validated individual forces.
They must not be added to a solved force history without re-solving equilibrium.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from solve_fullbody import Body,OUT,CONFIG

def main():
 b=Body();m,d=b.m,b.d;data=np.load(OUT/'fullbody_motion.npz');passive=[];gain=[];lengths=[];normalized=[]
 for i,q in enumerate(data['qpos']):
  b.pose(q);d.qvel[:]=data['qvel_native'][i];mujoco.mj_forward(m,d)
  bias=[];active=[];norm=[]
  for a in range(m.nu):
   length=float(d.actuator_length[a]);velocity=float(d.actuator_velocity[a]);lr=np.ascontiguousarray(m.actuator_lengthrange[a]);gp=np.ascontiguousarray(m.actuator_gainprm[a,:9]);bp=np.ascontiguousarray(m.actuator_biasprm[a,:9]);acc0=float(m.actuator_acc0[a])
   bias.append(max(0,-mujoco.mju_muscleBias(length,lr,acc0,bp))*b.length_scale**2)
   active.append(max(0,-mujoco.mju_muscleGain(length,velocity,lr,acc0,gp))*b.length_scale**2)
   lopt=(lr[1]-lr[0])/(gp[1]-gp[0]);slack=lr[0]-gp[0]*lopt
   norm.append((length-slack)/lopt)
  passive.append(bias);gain.append(active);lengths.append(d.actuator_length.copy());normalized.append(norm)
 passive=np.array(passive);gain=np.array(gain);normalized=np.array(normalized)
 assert np.isfinite(passive).all() and np.isfinite(gain).all()
 np.savez_compressed(OUT/'muscle_physiology_audit.npz',times=data['times'],passive_force_from_imported_law_N=passive,
  force_per_unit_activation_from_imported_law_N=gain,actuator_length_native_m=lengths,normalized_fiber_length_from_imported_mapping=normalized)
 lat=[i for i,n in enumerate(b.names) if n.startswith('LAT')]
 entries=[]
 for a in lat:
  entries.append({'actuator':b.names[a],'passive_at_start_N':float(passive[0,a]),'passive_at_lockout_N':float(passive[60,a]),
   'active_gain_min_N':float(gain[:,a].min()),'active_gain_max_N':float(gain[:,a].max()),
   'imported_lengthrange_m':m.actuator_lengthrange[a].tolist(),'observed_length_native_min_m':float(np.array(lengths)[:,a].min()),'observed_length_native_max_m':float(np.array(lengths)[:,a].max()),
   'review_required_zero_active_gain':bool(gain[:,a].max()<1e-8)})
 report={'status':'audit_only','grip_width_cm':CONFIG['grip_width_m']*100,'not_used_in_existing_force_solution':True,
  'law':'MuJoCo imported muscle gain(length,velocity) and passive bias(length); original parameters, common length scale squared for force',
  'latissimus':entries,'latissimus_passive_sum_at_lockout_N':float(passive[60,lat].sum()),
  'review':'Imported force-length mapping has to be checked before applying it to a biological recruitment comparison. Zero gain flags are parameter/operating-range findings, not proof that a human muscle is inactive.',
  'next_model_requirements':['Re-solve active plus passive equilibrium.','Audit force-length/tendon-slack mapping for this full-body assembly.','Specify joint stability/control constraints for active co-contraction.'],
  'source':'https://mujoco.readthedocs.io/en/stable/modeling.html#muscles'}
 (OUT/'muscle_physiology_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('MUSCLE_LAW_AUDIT_COMPLETE',CONFIG['grip_width_m'],'LAT passive at lockout',report['latissimus_passive_sum_at_lockout_N'],flush=True)

if __name__=='__main__':main()
