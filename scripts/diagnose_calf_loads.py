"""Controlled allocation ablations; these are NOT corrected human predictions.

Motion, ground forces, muscle paths and joint demand stay fixed. Only stated
passive floors or numerical selection parameters change. No force caps.
"""
import json,hashlib,numpy as np
from jump_rig import ROOT
from tension_reference import solve_causal_history,reaction_cone

def main():
    p=ROOT/'results/contact_jump_scapula/600hz';out=ROOT/'results/jump_contact_visualization';z=np.load(p/'tensions.npz');s=json.loads((p/'tension_summary.json').read_text());names=s['names'];coords=s['coordinate_names'];keep=[i for i,n in enumerate(coords) if not n.startswith('mtp_angle_')]
    calf={side:[i for i,n in enumerate(names) if n.startswith(('soleus','gasmed','gaslat')) and n.endswith('_'+side)] for side in ['r','l']}
    lower=[i for i,n in enumerate(names) if n.startswith(('soleus','gas','tibant','tibpost','perlong','perbrev','fdl','fhl','edl','ehl'))]
    cones=[reaction_cone(S,o,ax,20.,16) for S,o,ax in zip(z['S'],z['offset'],z['axes'])];records=[];outputs={}
    cases=[('baseline',None,.002,1.),('zero_lower_leg_passive','lower',.002,1.),('zero_all_passive','all',.002,1.),('no_temporal_penalty',None,0.,1.),('calf_cost_reference_minus_10pct',None,.002,.9),('calf_cost_reference_plus_10pct',None,.002,1.1)]
    for key,remove,temporal,factor in cases:
        passive=z['passive'].copy();ref=z['reference_N'].copy()
        if remove=='lower':passive[:,lower]=0
        elif remove=='all':passive[:]=0
        ref[calf['r']+calf['l']]*=factor
        try:
            sol=solve_causal_history(z['times'],z['R'][:,keep],z['target'][:,keep],np.ones_like(passive),passive,ref,cones,temporal_weight=temporal,level_weight=1.);F=sol['force_N'];sides=[]
            if key=='baseline':assert np.max(abs(F-z['force_N']))<1e-6,'Baseline replay differs'
            for side,ids in calf.items():
                series=F[:,ids].sum(1);k=int(series.argmax());row=coords.index('ankle_angle_'+side);moment=float(z['R'][k,row,ids]@F[k,ids]);net=float(z['target'][k,row])
                sides.append(dict(side=side,peak_N=float(series[k]),peak_time_s=float(z['times'][k]),calf_moment_Nm=moment,required_net_ankle_moment_Nm=net,other_muscle_moment_Nm=net-moment,passive_calf_at_peak_N=float(passive[k,ids].sum())))
            records.append(dict(case=key,status='solved',sides=sides,moment_error_Nm=sol['checks']['moment_balance_max_error'],reaction_cone_violation_N=sol['checks']['stability_max_violation_N']));outputs[key]=F
        except RuntimeError as e:records.append(dict(case=key,status='infeasible',error=str(e)))
        print('CALF_ABLATION',json.dumps(records[-1]),flush=True)
    result=dict(status='diagnostic_only',motion_sha256=hashlib.sha256((p/'motion.npz').read_bytes()).hexdigest(),fixed=['motion','GRF/COP','muscle paths and moment arms','joint moment requirements','GH cone'],reference_scaling_note='Changes only numerical cost normalization; passive floors fixed. Not an anatomical muscle-size intervention.',zero_passive_note='Removes a model assumption to diagnose sensitivity; not a proposed physiological correction.',strength_ceiling_enforced=False,cases=records,canonical_forces_modified=False)
    (out/'calf_allocation_sensitivity.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'calf_allocation_sensitivity.npz',times=z['times'],**outputs)

if __name__=='__main__':main()
