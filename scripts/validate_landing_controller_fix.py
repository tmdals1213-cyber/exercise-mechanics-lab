"""Three body/control parameter cases, both fixes, at three native timesteps."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import copy,json
import numpy as np
from jump_monitor_data import ROOT,read,atomic
from audit_landing_controller_components import one,OUT
MODELS={}

def refined_case(task):
    from controlled_broad_jump import run,build
    ref,dt=task;case=copy.deepcopy(ref)
    if dt!=ref['dt_s']:
        if dt not in MODELS:MODELS[dt]=build(dt,fall_contacts=True)
        source=read(Path(ref['folder'])/'control_summary.json');folder=OUT/'refined_originals'/ref['id']/f'dt_{round(dt*1e6)}us';run(source['parameters'],folder,dt=dt,prepared=MODELS[dt]);case['folder']=str(folder)
    case['dt_s']=dt;case['id']=ref['id']+f'_dt_{round(dt*1e6)}us';return case

def main():
    ranked=read(OUT/'relapse_cases.json');references=[ranked[0],next(q for q in ranked if q['dt_s']==.00025)]
    stable=ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1/stable_reference'
    refs=references+[dict(id='verified_stable_reference',folder=str(stable),quiet_time_s=3.,dt_s=.0005)]
    jobs=[]
    # Keep the saved source timestep for exact precontact replay checks. New
    # refined original-controller sources are generated first for a fair A/B.
    results=[];atomic(OUT/'fix_validation_status.json',dict(status='preparing_refined_sources',completed=0,total=9))
    with ProcessPoolExecutor(max_workers=3)as pool:
        sources=[pool.submit(refined_case,(ref,dt))for ref in refs for dt in [.0005,.00025,.000125]]
        for future in as_completed(sources):
            case=future.result()
            for label in ['centroidal_no_root_pitch','state_recovery_centroidal']:jobs.append((case,label))
        atomic(OUT/'fix_validation_status.json',dict(status='running',completed=0,total=len(jobs)))
        futures=[pool.submit(one,job)for job in jobs]
        for future in as_completed(futures):
            row=future.result();results.append(row);atomic(OUT/'fix_validation_results.json',results);atomic(OUT/'fix_validation_status.json',dict(status='running',completed=len(results),total=len(jobs)));print('LANDING_FIX_CHECK',row['case'],row['variant'],row['strict_stable'],row['dynamics_eligible'],flush=True)
    summary={label:dict(stable_cases=sum(r['strict_stable']and r['dynamics_eligible']for r in results if r['variant']==label),total_cases=sum(r['variant']==label for r in results),all_precontact_replays_exact=all(r['precontact_qpos_max_error']==0 for r in results if r['variant']==label),maximum_first_contact_error_m=max(r['first_contact_error_m']for r in results if r['variant']==label))for label in ['centroidal_no_root_pitch','state_recovery_centroidal']}
    atomic(OUT/'fix_validation_summary.json',summary);atomic(OUT/'fix_validation_status.json',dict(status='completed',completed=len(results),total=len(jobs)));print('LANDING_FIX_VALIDATED',json.dumps(summary),flush=True)

if __name__=='__main__':main()
