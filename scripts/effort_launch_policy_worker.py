"""Persistent native simulator for the new 52-output launch parameter policy."""
from pathlib import Path
import argparse,contextlib,json,sys,time
import numpy as np
from controlled_broad_jump import ROOT,build,run
from effort_launch_parameters import decode
from jump_phase_objective import phase_metrics
from effort_launch_objective import evaluate_launch
from jump_monitor_data import read,atomic

def emit(kind,value):print(kind+' '+json.dumps(value,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--definition',required=True);ap.add_argument('--out',required=True);ap.add_argument('--dt',type=float,default=.0005);a=ap.parse_args();config=read(a.definition);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);base=config['base_parameters'];items=config['actions'];count=0
    with contextlib.redirect_stdout(sys.stderr):prepared=build(a.dt)
    emit('EFFORT_WORKER_READY',dict(action_count=len(items),observation=config['observation'],actions=items,dt_s=a.dt,primary_objective='Precontact-valid heel projection at first foot contact',recovery_separate=True,individual_muscle_tension=False))
    for line in sys.stdin:
        try:
            command=json.loads(line)
            if command['cmd']=='close':break
            p=decode(base,items,command['action']);p.update(stop_after_touchdown_s=0.,record_energy_ledger=False)
            with contextlib.redirect_stdout(sys.stderr):report,z=run(p,dt=a.dt,prepared=prepared)
            metrics=evaluate_launch(report,z,prepared[1]);distance=metrics['heel_projection_at_first_foot_contact_m'];reference=config.get('reference_distances_by_dt_s',{}).get(format(a.dt,'g'),config['seed_distance_m']);reward=(distance-reference)/.25 if metrics['optimization_domain_eligible']else -3.
            result=dict(episode=count,reward=float(reward),distance_m=distance,eligible=metrics['optimization_domain_eligible'],metrics=metrics,parameters=p,compute_seconds=report['compute_seconds'],finished_at=time.time(),kind=command.get('kind','eval'))
            with(out/'episodes.jsonl').open('a',encoding='utf-8')as file:file.write(json.dumps(result)+'\n')
            emit('EFFORT_RESULT',result);count+=1
        except Exception as exc:emit('EFFORT_ERROR',dict(error=repr(exc)))

if __name__=='__main__':main()
