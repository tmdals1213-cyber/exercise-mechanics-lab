"""Check real operator fixtures with and without the GH direction constraint."""
import json
from pathlib import Path
import numpy as np
from build_tension_reference import ohp_inputs,jump_inputs
from tension_reference import solve_history
ROOT=Path(__file__).resolve().parents[1]

def main():
    rule=json.loads((ROOT/'scenarios/tension_reference.json').read_text(encoding='utf-8'))['solution_selection'];out={}
    for task,loader in [('ohp',ohp_inputs),('jump',jump_inputs)]:
        d=loader(25);free=solve_history(d['times'],d['R'],d['target'],d['gain'],d['passive'],d['reference'],None,rule['temporal_weight'],rule['level_weight']);stable=dict(np.load(ROOT/f'results/tension_reference/{task}/standard.npz'));reaction=d['offset']-np.einsum('tscn,tn->tsc',d['S'],free['force_N']);angle=np.rad2deg(np.arccos(np.clip((d['axes']*reaction).sum(2)/np.maximum(np.linalg.norm(reaction,axis=2),1e-12),-1,1)))
        out[task]={'max_GH_angle_without_constraint_deg':float(angle.max()),'largest_tension_change_from_GH_constraint_N':float(abs(stable['force_N']-free['force_N']).max()),'force_balance_without_stability':free['checks']['moment_balance_max_error']}
    if not (out['jump']['largest_tension_change_from_GH_constraint_N']>1e-3 and out['jump']['max_GH_angle_without_constraint_deg']>rule['GH_cone_half_angle_deg']):raise RuntimeError('Jump fixture did not exercise the stability constraint')
    report={'status':'passed','cases':out,'scope':'Constraint can be inactive when the selected solution is already inside the cone. These are mechanical fixtures, not a universal antagonist recruitment or dislocation claim.'};(ROOT/'results/tension_reference/stability_effect_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
