"""Persistent isolated native solver; every attempt including falls is saved."""
from pathlib import Path
import argparse,contextlib,copy,json,sys,time
import numpy as np
from controlled_broad_jump import ROOT,build,run
from effort_launch_parameters import decode
from landing_learning_objective import score,motion_preview
from jump_monitor_data import read,atomic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--worker',type=int,required=True);a=ap.parse_args();root=Path(a.run);out=root/f'worker_{a.worker:02d}';out.mkdir(parents=True,exist_ok=True);definition=read(root/'definition.json');count=len(list(out.glob('episode_*')));models={};caches={}
    import landing_learning_objective as objective
    objective.TARGET_FIRST=float(definition.get('goal_first_contact_m',objective.TARGET_FIRST));objective.TARGET_RETAINED=float(definition.get('goal_retained_contact_m',objective.TARGET_RETAINED));objective.VERSION=definition.get('score_version',objective.VERSION)
    def make_model(dt):
        if definition.get('registered_example'):
            from analyze_example import load_example
            return load_example(ROOT/definition['registered_example'],dt)
        return build(dt,fall_contacts=True)
    with contextlib.redirect_stdout(sys.stderr):models[.0005]=make_model(.0005)
    print('LANDING_READY '+json.dumps(dict(worker=a.worker)),flush=True)
    for line in sys.stdin:
        try:
            command=json.loads(line)
            if command['cmd']=='close':break
            dt=float(command.get('dt',.0005))
            if dt not in models:
                with contextlib.redirect_stdout(sys.stderr):models[dt]=make_model(dt)
            prepared=models[dt];base=command.get('base',definition['long_anchor']);items=command.get('items');p=decode(base,items,command['action'])if items is not None else copy.deepcopy(base)
            folder=out/f'episode_{count:06d}';count+=1;cache=None
            if command.get('cache_launch'):cache=caches.setdefault((command.get('cache_id','long'),dt),{})
            with contextlib.redirect_stdout(sys.stderr):report,z=run(p,folder,dt=dt,prepared=prepared,flight_cache=cache)
            result=score(report,z,prepared[1]);preview=motion_preview(report,z,prepared[1]);atomic(folder/'motion.json',preview)
            identifier=(root.name+'_'if root.name!='landing_stability_v2'else'')+f'{out.name}_{folder.name}'
            result.update(id=identifier,worker=a.worker,episode=count-1,kind=command.get('kind','evaluation'),context=command.get('context'),action=command.get('action'),parameters=p,folder=str(folder),dt_s=dt,compute_seconds=report['compute_seconds'],finished_at=time.time())
            atomic(folder/'score.json',result)
            with(out/'episodes.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(result)+'\n')
            from maintain_project_storage import prune_worker_failures
            prune_worker_failures(out,folder,result)
            print('LANDING_RESULT '+json.dumps(result),flush=True)
        except Exception as exc:print('LANDING_ERROR '+json.dumps(dict(error=repr(exc))),flush=True)

if __name__=='__main__':main()
