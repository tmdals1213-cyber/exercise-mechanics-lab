from pathlib import Path
import sys,json,numpy as np,mujoco
from scipy.spatial.transform import Rotation
p=Path(sys.argv[1]);z=np.load(p/'control_motion.npz');names=json.loads((p/'joint_names.json').read_text());m=mujoco.MjModel.from_binary_path(str(p/'control_model.mjb'));d=mujoco.MjData(m)
for tt in [1.15,1.4,1.75,2.1,2.2]:
 i=np.argmin(abs(z['times']-tt));d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);print('TIME',z['times'][i],'rootpitch',Rotation.from_matrix(d.xmat[1].reshape(3,3)).as_rotvec()[0])
 for n in ['flex_extension','shoulder_elv_r','hip_flexion_r','knee_angle_r','ankle_angle_r']:
  j=m.joint(n).id;qa=m.jnt_qposadr[j];ni=names.index(n);print(n,round(d.qpos[qa],2),'target',round(z['target_angles_rad'][i,ni],2),'axis',np.round(d.xaxis[j],2))
 print('leg vectors',np.round(d.xpos[m.body('tibia_r').id]-d.xpos[m.body('femur_r').id],2),np.round(d.xpos[m.body('talus_r').id]-d.xpos[m.body('tibia_r').id],2))
