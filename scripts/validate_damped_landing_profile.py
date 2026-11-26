"""Complete the selected controller's nine-case regression set."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from jump_monitor_data import ROOT,read,atomic
from audit_landing_controller_components import one,OUT

def main():
    profile='state_recovery_damped';ranked=read(OUT/'relapse_cases.json');refs=[ranked[0],next(q for q in ranked if q['dt_s']==.00025)];stable=ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1/stable_reference';refs.append(dict(id='verified_stable_reference',folder=str(stable),quiet_time_s=3.,dt_s=.0005))
    results=[r for r in read(OUT/'damping_probe_results.json')if r['variant']==profile];assert len(results)==2 and all(r['strict_stable']and r['dynamics_eligible']for r in results)
    jobs=[]
    for ref in refs:
        for dt in [.0005,.00025,.000125]:
            if ref['id']!='verified_stable_reference'and dt==.000125:continue
            case=ref.copy();folder=Path(ref['folder'])if dt==ref['dt_s']else OUT/'refined_originals'/ref['id']/f'dt_{round(dt*1e6)}us';case.update(folder=str(folder),id=ref['id']+f'_damped_dt_{round(dt*1e6)}us',dt_s=dt);jobs.append((case,profile))
    atomic(OUT/'damped_profile_status.json',dict(status='running',completed=len(results),total=9))
    with ProcessPoolExecutor(max_workers=3)as pool:
        futures=[pool.submit(one,job)for job in jobs]
        for future in as_completed(futures):
            row=future.result();results.append(row);atomic(OUT/'damped_profile_results.json',results);atomic(OUT/'damped_profile_status.json',dict(status='running',completed=len(results),total=9));print('DAMPED_PROFILE_VALIDATION',row['case'],row['strict_stable'],row['fell'],flush=True)
    caps=results[0]['fixed_torque_capacities'];summary=dict(profile=profile,stable_cases=sum(r['strict_stable']and r['dynamics_eligible']for r in results),total_cases=len(results),all_precontact_replays_exact=all(r['precontact_qpos_max_error']==0 for r in results),maximum_first_contact_error_m=max(r['first_contact_error_m']for r in results),fixed_torque_capacities_identical=all(r['fixed_torque_capacities']==caps for r in results),no_root_actuation=all(r['maximum_root_applied_force_or_torque']==0 for r in results),results=results)
    atomic(OUT/'damped_profile_summary.json',summary);atomic(OUT/'damped_profile_status.json',dict(status='completed',completed=len(results),total=9));print('DAMPED_PROFILE_SUMMARY',summary['stable_cases'],summary['total_cases'],flush=True)

if __name__=='__main__':main()
