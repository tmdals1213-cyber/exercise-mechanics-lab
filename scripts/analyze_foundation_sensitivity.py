"""Deterministic assumption sweep. These ranges are NOT confidence intervals."""
import argparse,json,hashlib
import numpy as np
from solve_fullbody import Body,ROOT,CONFIG
from contact_mechanics import Contact
from build_model_foundation import solve


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--grip',type=int,choices=[65,80],required=True); args=ap.parse_args()
    if abs(CONFIG['grip_width_m']-args.grip/100)>1e-10: raise ValueError('Scenario / grip mismatch')
    dest=ROOT/f'results/model_foundation/grip_{args.grip}'; op=np.load(dest/'analysis_operators.npz'); base=np.load(dest/'fullbody_motion.npz')
    b=Body(); contacts=[Contact('foot_'+str(i),bid,p,friction=CONFIG['ground_friction']) for i,(bid,p) in enumerate(b.sole)]
    names=b.names; sample=np.unique(np.r_[np.arange(0,len(base['times']),10),60,69,70,129,130,138]).astype(int)
    variants=[('reference',20.,1.,1.),('GH15',15.,1.,1.),('GH25',25.,1.,1.),('deltoid_cheaper_triceps_costlier',20.,.75,1.25),('deltoid_costlier_triceps_cheaper',20.,1.25,.75)]
    forces=[]; failures=[]; labels=[]
    for label,angle,delt,tri in variants:
        weight=np.ones(len(names))
        for i,n in enumerate(names):
            if n.startswith('DELT'): weight[i]=delt
            if n.startswith('TRI'): weight[i]=tri
        row=[]
        for index in sample:
            if label=='reference': row.append(base['force_N'][index]); continue
            solution,check=solve(op,int(index),contacts,angle,weight)
            if solution is None:
                failures.append({'variant':label,**check}); row.append(np.full(len(names),np.nan))
            else: row.append(solution['force_N'])
        forces.append(row); labels.append(label); print('SENSITIVITY',args.grip,label,flush=True)
    forces=np.asarray(forces); np.savez_compressed(dest/'sensitivity.npz',indices=sample,times=base['times'][sample],forces_N=forces,variant_names=labels)
    spread=np.nanmax(forces,axis=0)-np.nanmin(forces,axis=0)
    report={'status':'completed','implementation_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),'operators_sha256':hashlib.sha256((dest/'analysis_operators.npz').read_bytes()).hexdigest(),'scope':'5 assumed parameter/cost variants at selected phase samples; NOT a probability interval or continuous-trajectory peak envelope.','sampled_times_s':base['times'][sample].tolist(),
            'variants':[{'name':v[0],'GH_half_angle_deg':v[1],'deltoid_effort_cost_weight':v[2],'triceps_effort_cost_weight':v[3]} for v in variants],
            'failures':failures,'all_variants_feasible_at_all_samples':not failures,'largest_force_spread_N':float(np.nanmax(spread)),
            'largest_spread_actuator':names[np.unravel_index(np.nanargmax(spread),spread.shape)[1]],
            'force_sharing_is_model_dependent':True,'no_strength_ceiling':True,'scope_note':'Other uncertainty (anatomy, PCSA, tendon properties, motion, unmodeled stabilizers) is not covered.'}
    (dest/'sensitivity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SENSITIVITY_COMPLETE',args.grip,report['largest_force_spread_N'],flush=True)


if __name__=='__main__': main()
