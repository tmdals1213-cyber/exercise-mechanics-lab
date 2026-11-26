"""Regression: hip flexion must be a balanced generalized coordinate."""
import json,numpy as np
from solve_fullbody import Body,is_finger_joint,OUT
assert not is_finger_joint('hip_flexion_r') and not is_finger_joint('hip_flexion_l')
assert is_finger_joint('ip_flexion_r') and is_finger_joint('mp_flexion_l') and is_finger_joint('mcp2_flexion_r')
b=Body();B=b.reduction(b.base)
for side in ['r','l']:
 name='hip_flexion_'+side;j=b.joints[name];assert j in b.independent and j not in b.dependent
 column=6+b.ind_names.index(name);assert abs(B[b.m.jnt_dofadr[j],column]-1)<1e-9
report={'status':'passed','independent_coordinate_count':6+len(b.ind_names),'balanced_hips':['hip_flexion_r','hip_flexion_l'],
 'classification':'finger joint name prefixes, not substrings'}
(OUT/'joint_classification_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
