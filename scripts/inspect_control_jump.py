from pathlib import Path
import numpy as np,json
import sys
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'results/controlled_broad_jump/pilot_v1';z=np.load(p/'control_motion.npz');names=json.loads((p/'joint_names.json').read_text());from scipy.spatial.transform import Rotation
for t in [0,.25,.5,.8,1.15,1.3,1.43,1.65,1.9,2.1]:
 i=np.argmin(abs(z['times']-t));print(round(float(z['times'][i]),3),'C',np.round(z['com'][i],2),'V',np.round(z['velocity'][i],2),'H',np.round(z['angular_momentum'][i],1),'F',np.round(z['force_world_N'][i]),'rot',np.round(Rotation.from_quat(np.roll(z['qpos'][i,3:7],-1)).as_rotvec(),2),'j',np.round(z['qpos'][i,7:],2))
print(names)
