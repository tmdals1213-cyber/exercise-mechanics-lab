"""Explain the high quadriceps estimate without fitting it to a target force."""
import json
from pathlib import Path
import numpy as np
from tension_reference import solve_causal_history,reaction_cone
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'results/measured_jump_analysis/s10_CMJ_t1';s=json.loads((source/'tension_summary.json').read_text());z=np.load(source/'tensions.npz');names=s['names'];coord=s['coordinate_names'];ids=[j for j,n in enumerate(names) if n.startswith(('vas','recfem'))];curve=z['force_N'][:,ids].sum(1)/2;k=int(curve.argmax());sides=[]
    for side in ['r','l']:
        row=coord.index('knee_angle_'+side);quad=[j for j in ids if ('l' if names[j].endswith('_l') else 'r')==side];details=[dict(element=names[j],tension_N=float(z['force_N'][k,j]),moment_arm_cm=float(z['R'][k,row,j]*100),knee_moment_Nm=float(z['force_N'][k,j]*z['R'][k,row,j])) for j in range(len(names)) if abs(z['force_N'][k,j]*z['R'][k,row,j])>.05];qforce=float(z['force_N'][k,quad].sum());qmoment=float(z['R'][k,row,quad]@z['force_N'][k,quad]);sides.append(dict(side=side,net_knee_moment_Nm=float(z['target'][k,row]),quadriceps_tension_N=qforce,quadriceps_moment_Nm=qmoment,effective_quadriceps_moment_arm_cm=abs(qmoment/qforce)*100,other_elements=details))
    solved=[i for i,n in enumerate(coord) if not n.startswith('mtp_angle_')];stability=[reaction_cone(S,o,ax,20.,16) for S,o,ax in zip(z['S'],z['offset'],z['axes'])];alt=solve_causal_history(z['times'],z['R'][:,solved],z['target'][:,solved],np.ones_like(z['passive']),z['passive'],z['reference_N'],stability,temporal_weight=0.,level_weight=1.);y=alt['force_N'][:,ids].sum(1)/2;result=dict(status='diagnostic_not_physiological_validation',peak_time_s=float(z['times'][k]),baseline_quadriceps_peak_mean_per_side_N=float(curve.max()),no_temporal_penalty_peak_mean_per_side_N=float(y.max()),alternative_case_is_not_applied=True,no_temporal_penalty_checks=alt['checks'],sides=sides,interpretation='High quadriceps force reflects the short native moment arms and selected biarticular/antagonist forces. Force sharing and transferred anatomical paths remain assumptions; force agreement with ground measurements does not validate this muscle magnitude.')
    (source/'quadriceps_peak_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['peak_time_s','baseline_quadriceps_peak_mean_per_side_N','no_temporal_penalty_peak_mean_per_side_N']}),flush=True)

if __name__=='__main__':main()
