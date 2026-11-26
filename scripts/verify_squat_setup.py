"""Specification/model preflight only; this does not validate squat poses."""
from pathlib import Path
import json
from solve_fullbody import Body,ROOT
OUT=ROOT/'results/squat_comparison'
spec=json.loads((ROOT/'scenarios/squat_comparison.json').read_text('utf-8'))
assert spec['status']=='specification_only_not_simulated'
assert [v['id'] for v in spec['variants']]==['high_bar','low_bar','front','zercher']
for v in spec['variants']:
 case=json.loads((ROOT/f'scenarios/squat_{v["id"]}.json').read_text('utf-8'))
 assert case['common']==spec['common'] and case['variant']==v
assert spec['common']['strength_ceiling_enforced'] is False
assert spec['presentation']['language']=='en' and spec['presentation']['bottom_reserved_px']==384
b=Body();required=['hip_flexion_r','hip_flexion_l','knee_angle_r','knee_angle_l','ankle_angle_r','ankle_angle_l','flex_extension']
assert all(n in b.ind_names for n in required)
report={'status':'specification_checks_passed','simulated':False,'squat_contact_and_trajectory_solver_ready':False,
 'variants':[v['label'] for v in spec['variants']],'matched_inputs':spec['common']['matching'],
 'independent_coordinate_count':6+len(b.ind_names),'available_balanced_coordinates':required,
 'pending':['actual rack-surface contact fitting','variant-specific full-cycle kinematics','support/friction feasibility','inverse dynamics and muscle allocation','anatomical and rendered visual checks']}
(OUT/'setup_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
