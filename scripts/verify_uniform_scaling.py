"""Check similarity inverse dynamics against an independently resized MuJoCo model."""
import copy,json
import numpy as np
import mujoco
from solve_fullbody import Body,OUT

body=Body();base=body.m;data=np.load(OUT/'fullbody_motion.npz')
trans_q=list(range(3));trans_v=list(range(3))
for j in range(1,base.njnt):
 if base.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE:
  trans_q.append(int(base.jnt_qposadr[j]));trans_v.append(int(base.jnt_dofadr[j]))
reports=[]
for scale in [.9,1.1]:
 scaled=copy.copy(base)
 for key in ['body_pos','body_ipos','jnt_pos','geom_pos','geom_size','site_pos','site_size']:
  getattr(scaled,key)[:]*=scale
 scaled.body_inertia[:]*=scale**2
 scaled.qpos0[trans_q]*=scale
 d=mujoco.MjData(scaled);max_error=0.
 for frame in [0,30,60]:
  q=data['qpos'][frame];velocity=np.sin(np.arange(base.nv))*.15;acceleration=np.cos(np.arange(base.nv))*.2
  body.pose(q);body.d.qvel[:]=velocity;mujoco.mj_forward(base,body.d);body.d.qacc[:]=acceleration
  full=np.zeros(base.nv);mujoco.mj_rne(base,body.d,1,full)
  gravity=base.opt.gravity.copy();base.opt.gravity[:]=0
  inertia=np.zeros(base.nv);mujoco.mj_rne(base,body.d,1,inertia);base.opt.gravity[:]=gravity
  equivalent=full+(scale-1)*inertia
  predicted=equivalent*scale;predicted[trans_v]=equivalent[trans_v]
  d.qpos[:]=q;d.qpos[trans_q]*=scale;d.qvel[:]=velocity;d.qvel[trans_v]*=scale
  mujoco.mj_forward(scaled,d);d.qacc[:]=acceleration;d.qacc[trans_v]*=scale
  actual=np.zeros(base.nv);mujoco.mj_rne(scaled,d,1,actual)
  max_error=max(max_error,float(np.abs(actual-predicted).max()))
 assert max_error<1e-7,(scale,max_error)
 reports.append({'length_scale':scale,'max_inverse_dynamics_difference':max_error})
report={'status':'passed','reference':'independently scaled body offsets, COMs, inertias, joint/slide coordinates; same masses', 'cases':reports}
(OUT/'validation_uniform_scaling.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
