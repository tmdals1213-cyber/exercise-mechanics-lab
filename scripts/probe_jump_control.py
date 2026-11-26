from pathlib import Path
import json,numpy as np,mujoco
from retarget_measured_jump import MeasuredJumpRig
r=MeasuredJumpRig(87,1.78);m=r.m
source=Path(__file__).resolve().parents[1]/'results/athletic_box_jump/yun_160_final_cycle_v4/motion.npz'
q=np.load(source)['qpos'][0];r.b.pose(q)
names=['flex_extension']+[stem+'_'+s for s in ['r','l'] for stem in ['hip_flexion','knee_angle','ankle_angle','mtp_angle','shoulder_elv','elbow_flexion']]
for name in names:
 j=m.joint(name).id;bid=m.jnt_bodyid[j];print(name,'q',round(float(q[m.jnt_qposadr[j]]),3),'body',m.body(bid).name,'bodyjoints',[m.joint(k).name for k in range(m.njnt) if m.jnt_bodyid[k]==bid],'axis',np.round(r.d.xaxis[j],3),'anchor',np.round(r.d.xanchor[j],3))
print('ROOT',q[:7]);print('com',r.d.subtree_com[1]);print('mass',m.body_mass.sum(),'dims',m.nq,m.nv,m.neq)
print('feet',[(s,r.d.site_xpos[m.site('subject_sole_reference_'+s).id])for s in ['r','l']])
