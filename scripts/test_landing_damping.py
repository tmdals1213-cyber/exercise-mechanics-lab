from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from jump_monitor_data import ROOT,read,atomic
from audit_landing_controller_components import one,OUT

def main():
    ranked=read(OUT/'relapse_cases.json');refs=[ranked[0],next(q for q in ranked if q['dt_s']==.00025)]
    profiles=['state_recovery_damped','centroidal_damped','state_centroidal_only_damped'];jobs=[]
    for ref in refs:
        folder=OUT/'refined_originals'/ref['id']/'dt_125us';case=ref.copy();case.update(folder=str(folder),id=ref['id']+'_damping_probe_125us',dt_s=.000125)
        for label in profiles:jobs.append((case,label))
    results=[]
    with ProcessPoolExecutor(max_workers=3)as pool:
        futures=[pool.submit(one,job)for job in jobs]
        for future in as_completed(futures):
            row=future.result();results.append(row);atomic(OUT/'damping_probe_results.json',results);print('DAMPING_PROBE',row['case'],row['variant'],row['strict_stable'],row['fell'],flush=True)
    print('DAMPING_PROBE_DONE',flush=True)

if __name__=='__main__':main()
