"""Resumable, joint 43-coordinate native-physics initialization, separate from PPO."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse, json, time, hashlib
import numpy as np
import search_two_metre_initial_solution as search
from jump_parameter_rl_worker import FULL_SPECS
from jump_monitor_data import ROOT, read, atomic, lines

def initialize(base, specs):
    search.initialize_search(base, .0005, specs, True, True)

def evaluate(job):
    row=search.evaluate(job)
    if 'error' in row:return row
    # Continuous failure ranking: landing survival/posture improves before a
    # stable record exists. First heel projection remains only a diagnostic.
    if not row['stable']:
        row['cost'] += 10*max(.85-row.get('end_z',0),0)**2 + 2*max(.10-row.get('clearance_m',0),0)**2
    row['feasible']=bool(row.get('feasible')and row['max_ROM_excess_rad']<.04 and (row.get('refinement')or{}).get('max_ROM_excess_rad',1)<.04)
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='rigid_full_initial_v31');ap.add_argument('--base',required=True);ap.add_argument('--generations',type=int,default=4000);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--population',type=int,default=32);ap.add_argument('--resume',action='store_true');ap.add_argument('--deadline',type=float,required=True);args=ap.parse_args()
    out=ROOT/'results/controlled_broad_jump'/args.label;out.mkdir(parents=True,exist_ok=True)
    base=read(args.base)['parameters'].copy();base.pop('stop_after_takeoff_s',None);base.update(seconds=6.,record_energy_ledger=False,integration_method='RK4',joint_limit_time_constant_s=.004)
    spans={'hip_bottom':.25,'knee_bottom':.35,'trunk_bottom':.15,'hip_top':.2,'shoulder_top':.35,'pitch_bottom':.22,'pitch_top':.25,'push_s':.055,'arm_delay_s':.08,'prepush_lean_s':.08,'hip_extension_duration_fraction':.20,'knee_extension_duration_fraction':.15,'flight_hip':.65,'flight_knee':.65,'flight_shoulder':.75,'flight_trunk':.25,'flight_transition_s':.07,'prepare_hip':.60,'prepare_knee':.40,'prepare_shoulder':.7,'prepare_trunk':.3,'prepare_landing_s':.16,'prepare_transition_s':.07,'landing_hip':.65,'landing_knee':.65,'landing_ankle':.4,'landing_trunk':.25,'landing_shoulder':.6,'landing_transition_s':.10,'lean_m':.15}
    specs=[(k,spans.get(k,h),u)for k,h,u in FULL_SPECS];n=len(specs);assert n==43
    rng=np.random.default_rng(31031);mean=np.zeros(n);cov=np.eye(n)*.22**2;rows=[];startgen=0
    definition=dict(method='Full-covariance CEM initialization, not PPO',all_parameters_jointly_varied=n,full_policy_parameters=43,population=args.population,dt_s=.0005,secondary_dt_s=.00025,integration_method='RK4',joint_limit_time_constant_s=.004,spans=specs,base=base,body_kg=87,height_m=1.78,all_posture_and_timing_parameters_active=True,controller_gains_contact_and_anthropometry_fixed=True,deadline=args.deadline)
    if args.resume and(out/'optimizer_state.npz').exists():
        state=np.load(out/'optimizer_state.npz');mean=state['mean'];cov=state['cov'];startgen=int(state['generation'])+1;rows=lines(out/'trials.jsonl');saved=read(out/'search_definition.json');base=saved['base'];specs=saved['spans'];definition=saved;rng.bit_generator.state=read(out/'rng_state.json')
    atomic(out/'search_definition.json',definition)
    best=min(rows,key=lambda r:r['cost'])if rows else None;start=time.time();stale=0;lastcost=best['cost']if best else 1e9
    rejected=set(read(out/'rejected_parameter_hashes.json',[]))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(base,specs))as pool:
        for gen in range(startgen,args.generations):
            if time.time()>args.deadline or(out/'stop.request').exists():break
            actions=np.clip(rng.multivariate_normal(mean,cov,args.population),-1,1)
            actions[-4:]=np.clip(rng.normal(mean,.32,(4,n)),-1,1);actions[0]=mean
            if best:actions[1]=best['action']
            if gen==0:actions[1]=np.zeros(n)
            batch=list(pool.map(evaluate,[(len(rows)+i,x)for i,x in enumerate(actions)]));rows.extend(batch)
            with(out/'trials.jsonl').open('a',encoding='utf-8')as log:
                for row in batch:log.write(json.dumps(row)+'\n')
            archive=sorted(rows,key=lambda r:r['cost'])[:3];unique={tuple(r['action']):r for r in batch+archive};elite=sorted(unique.values(),key=lambda r:r['cost'])[:8]
            x=np.array([r['action']for r in elite]);weights=np.exp(-np.arange(len(x))*.25);weights/=weights.sum();newmean=weights@x;delta=x-newmean;newcov=(delta.T*weights)@delta;mean=.4*mean+.6*newmean;cov=.4*cov+.6*(.85*newcov+.15*np.diag(np.diag(newcov)));v,Q=np.linalg.eigh(cov);cov=(Q*np.clip(v,.04**2,.5**2))@Q.T
            best=min(rows,key=lambda r:r['cost']);atomic(out/'best_candidate.json',best)
            stale=stale+1 if best['cost']>=lastcost-.003 else 0;lastcost=best['cost']
            if stale>=28:
                # Reheat every coordinate; preserve best and RNG checkpoint.
                mean=np.array(best['action']);cov=.35*cov+.65*np.eye(n)*(.18 if gen%2 else .30)**2;stale=0
                atomic(out/f'reheat_{gen:04d}.json',dict(generation=gen,all_dimensions_active=n))
            np.savez(out/'optimizer_state.npz',mean=mean,cov=cov,generation=gen);atomic(out/'rng_state.json',rng.bit_generator.state)
            valid=[r for r in rows if r.get('robust_stable')];distance=max([min(r['valid_distance_m'],r['refinement']['valid_distance_m'])for r in valid],default=None)
            atomic(out/'progress.json',dict(generation=gen,physical_trials=len(rows),best_cost=best['cost'],best_stable_distance_m=distance,wall_s=time.time()-start,last_update=time.time()))
            print('RIGID_FULL_INITIAL',gen,len(rows),'cost',round(best['cost'],4),'robust_stable',len(valid),'distance',distance,flush=True)
            feasible=sorted([r for r in batch if r.get('feasible')],key=lambda r:r['cost'])
            for row in feasible:
                digest=hashlib.sha256(json.dumps(row['parameters'],sort_keys=True).encode()).hexdigest()
                if digest in rejected:continue
                atomic(out/'feasible_candidate.json',row);print('RIGID_FULL_SEED_READY',row['valid_distance_m'],flush=True);return
    atomic(out/'search_exit.json',dict(time=time.time(),reason='deadline'if time.time()>args.deadline else'checkpoint_stop_or_generation_budget',physical_trials=len(rows)))

if __name__=='__main__':main()
