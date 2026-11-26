"""Re-solve a compact fixed-motion inverse problem with explicit force weights."""
from pathlib import Path
import argparse,json,csv
import numpy as np
from tension_reference import solve_history
from jump_monitor_data import ROOT,read,atomic

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--example',default=str(ROOT/'examples/long_jump'));ap.add_argument('--temporal-weight',type=float,default=.0002);ap.add_argument('--reference-scale',action='append',default=[],metavar='MUSCLE=FACTOR');ap.add_argument('--output',default=str(ROOT/'results/muscle_example'));args=ap.parse_args();folder=Path(args.example);meta=read(folder/'inverse_metadata.json');z=np.load(folder/'inverse_problem.npz');reference=z['reference_N'].copy();names=meta['force_names']
    for item in args.reference_scale:
        name,value=item.split('=',1);factor=float(value)
        if name not in names or factor<=0:raise ValueError('Expected known muscle name and positive reference factor')
        reference[names.index(name)]*=factor
    result=solve_history(z['times'],z['R'],z['target'],np.ones_like(z['passive_force_N']),z['passive_force_N'],reference,temporal_weight=args.temporal_weight,level_weight=1.,time_limit_s=240.);force=result['force_N'];error=float(abs(np.einsum('tqn,tn->tq',z['R'],force)-z['target']).max());assert error<.01;output=Path(args.output);output.mkdir(parents=True,exist_ok=True);np.savez_compressed(output/'forces.npz',times=z['times'],force_N=force)
    with(output/'forces.csv').open('w',encoding='utf-8',newline='')as f:
        writer=csv.writer(f);writer.writerow(['time_s']+names);writer.writerows(np.c_[z['times'],force])
    summary=dict(conditional_model_forces=True,fixed_motion_input=True,muscle_paths=meta['muscle_paths'],source_label=meta['source_label'],moment_balance_max_error_Nm=error,minimum_force_N=float(force.min()),reference_scale_changes=args.reference_scale,temporal_weight=args.temporal_weight,force_ceiling=False,not_neural_activation=True,not_a_new_motion_analysis=True,checks=result['checks']);atomic(output/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
