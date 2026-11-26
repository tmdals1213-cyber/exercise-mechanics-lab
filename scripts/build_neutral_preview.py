"""A relaxed generic standing pose for visual inspection, without force claims.

This is a separate pose reference; it does not silently replace jump motion.
"""
import json,numpy as np
from jump_rig import JumpRig,ROOT
r=JumpRig();q,info=r.planted_pose(.025,.04,.025,arms=(.15,.15),trunk=.015);r.b.pose(q)
out=ROOT/'results/tension_visualization'
np.savez_compressed(out/'neutral_geometry.npz',qpos=q,body_xpos=r.d.xpos.copy(),body_xmat=r.d.xmat.copy())
report={'purpose':'visual posture reference only; no tension solution assigned','source':'same native rigid-body skeleton and joint constraints','arm_plane_rad':.15,'shoulder_elevation_rad':.15,'elbow_flexion_rad':.12,'changes_jump_analysis':False,'pose_fit':info}
(out/'neutral_pose.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('NEUTRAL_REFERENCE',info)
