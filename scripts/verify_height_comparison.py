"""Cross-check comparison controls, baseline reproduction and physical identities."""
from pathlib import Path
import json
import numpy as np

OUT=Path(__file__).resolve().parents[1]/'results/ohp_fullbody'
report=json.loads((OUT/'height_comparison.json').read_text('utf-8'))
base=np.load(OUT/'fullbody_motion.npz');forces=np.load(OUT/'height_comparison_forces.npz')
cases=report['cases'];assert len(cases)==9
baseline_error=0.;moment_ratio_error=0.
for c in cases:
 f=forces[c['key']];assert f.shape==(76,416) and np.isfinite(f).all() and f.min()>=0
 assert c['max_balance_error']<.01 and c['max_nonroot_reserve_equivalent']<.1
 if c['height_cm']==180:baseline_error=max(baseline_error,float(np.abs(f-base['force_N']).max()))
 centre=next(r for r in cases if r['policy']==c['policy'] and r['height_cm']==180)
 moment_ratio_error=max(moment_ratio_error,abs(c['peak_bar_only_shoulder_elv_generalized_moment_Nm']/centre['peak_bar_only_shoulder_elv_generalized_moment_Nm']-c['height_cm']/180))
 rise=base['physical_grips'][-1,:,2].mean()-base['physical_grips'][0,:,2].mean()
 assert abs(c['bar_gravitational_work_J']-100*9.80665*rise*c['height_cm']/180)<1e-8
 if c['policy']=='same_bmi':assert abs(c['bmi']-90/1.8**2)<1e-10
 if c['policy']=='same_mass':assert c['mass_kg']==90
assert baseline_error<.001 and moment_ratio_error<1e-10
validation={'status':'passed','case_count':9,'frames_per_case':76,'baseline_max_force_difference_N':baseline_error,
 'bar_only_moment_similarity_max_ratio_error':moment_ratio_error,'max_balance_error':max(c['max_balance_error'] for c in cases),
 'limits':'Numerical controls and scaling identities; not experimental validation of muscle forces'}
(OUT/'validation_height_comparison.json').write_text(json.dumps(validation,indent=2),encoding='utf-8');print(json.dumps(validation,indent=2))
