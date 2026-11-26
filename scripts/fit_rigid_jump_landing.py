"""Causal initialization block: reuse the computed ground prefix, fit landing.

The later full policy retains all43 outputs; this block avoids repeatedly
solving an unchanged ground segment while constructing its initial motion.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,time,argparse
import numpy as np
import search_two_metre_initial_solution as search
from controlled_broad_jump import ROOT,run as native_run

CACHES={}
def cached_run(parameters,dt,prepared):return native_run(parameters,dt=dt,prepared=prepared,flight_cache=CACHES.setdefault(dt,{}))
def initialize(base,specs):
 search.FULL_SPECS=specs;search.initialize_search(base,.0005,specs,True,True);search.run=cached_run
def evaluate(job):
 row=search.evaluate(job)
 if row.get('max_ROM_excess_rad',1)>.04 or(row.get('refinement')and row['refinement'].get('max_ROM_excess_rad',1)>.04):row['feasible']=False
 return row

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='rigid_landing_initial_v29');ap.add_argument('--generations',type=int,default=80);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();launch=ROOT/'results/controlled_broad_jump/rigid_launch_fit_v28';ready=launch/'ready_launch.json';best=launch/'best_launch.json'
 while not ready.exists()and not(launch/'summary.json').exists():time.sleep(3)
 record=json.loads((ready if ready.exists()else best).read_text());base=record['parameters'].copy();base.pop('stop_after_takeoff_s',None);base.update(seconds=6.,record_energy_ledger=False,flight_hip=1.1,flight_knee=1.,flight_shoulder=1.3,flight_trunk=.0,flight_transition_s=.16,prepare_hip=1.7,prepare_knee=.3,prepare_shoulder=1.2,prepare_trunk=.25,prepare_landing_s=.30,prepare_transition_s=.16,landing_hip=1.2,landing_knee=1.2,landing_ankle=.15,landing_shoulder=.9,landing_trunk=.15,landing_transition_s=.20)
 specs=[('flight_hip',.45,'rad'),('flight_knee',.45,'rad'),('flight_shoulder',.45,'rad'),('flight_trunk',.20,'rad'),('flight_transition_s',.04,'s'),('prepare_hip',.35,'rad'),('prepare_knee',.20,'rad'),('prepare_shoulder',.45,'rad'),('prepare_trunk',.20,'rad'),('prepare_landing_s',.10,'s'),('prepare_transition_s',.04,'s'),('landing_hip',.30,'rad'),('landing_knee',.30,'rad'),('landing_ankle',.40,'rad'),('landing_shoulder',.5,'rad'),('landing_trunk',.20,'rad'),('landing_transition_s',.08,'s')];out=ROOT/'results/controlled_broad_jump'/a.label;out.mkdir(exist_ok=True);(out/'launch_snapshot.json').write_text(json.dumps(record,indent=2));definition=dict(method='Phase-wise causal initializer; full-covariance CEM with exact cached native ground prefix. Later PPO jointly trains all43 parameters.',initialization_stage='Airborne and landing; ground launch is the separately fitted native solution.',all_parameters_jointly_varied=len(specs),full_policy_parameters=43,population=24,dt_s=.0005,secondary_dt_s=.00025,integration_method='RK4',joint_limit_time_constant_s=.004,base=base,spans=specs,body_kg=87,height_m=1.78,no_prescribed_launch_state=True);(out/'search_definition.json').write_text(json.dumps(definition,indent=2));rng=np.random.default_rng(28029);n=len(specs);mean=np.zeros(n);cov=np.eye(n)*.30**2;rows=[];best_row=None;start=time.time()
 with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(base,specs))as pool:
  for gen in range(a.generations):
   actions=np.clip(rng.multivariate_normal(mean,cov,24),-1,1);actions[0]=mean
   if best_row is not None:actions[1]=best_row['action']
   batch=list(pool.map(evaluate,[(len(rows)+j,actions[j])for j in range(24)]));rows.extend(batch)
   with(out/'trials.jsonl').open('a')as log:
    for row in batch:log.write(json.dumps(row)+'\n')
   archive=sorted(rows,key=lambda r:r['cost'])[:3];unique={tuple(r['action']):r for r in batch+archive};elite=sorted(unique.values(),key=lambda r:r['cost'])[:6];x=np.array([r['action']for r in elite]);w=np.exp(-np.arange(len(x))*.25);w/=w.sum();newmean=w@x;delta=x-newmean;newcov=(delta.T*w)@delta;mean=.4*mean+.6*newmean;cov=.4*cov+.6*(.9*newcov+.1*np.diag(np.diag(newcov)));v,Q=np.linalg.eigh(cov);cov=(Q*np.clip(v,.06**2,.4**2))@Q.T;best_row=min(rows,key=lambda r:r['cost']);(out/'best_candidate.json').write_text(json.dumps(best_row,indent=2));valid=[r for r in rows if r.get('robust_stable')];distance=max([min(r['valid_distance_m'],r['refinement']['valid_distance_m'])for r in valid],default=None);np.savez(out/'optimizer_state.npz',mean=mean,cov=cov,generation=gen);(out/'progress.json').write_text(json.dumps(dict(generation=gen,physical_trials=len(rows),best_cost=best_row['cost'],best_stable_distance_m=distance,wall_s=time.time()-start),indent=2));print('RIGID_LANDING_FIT',gen,len(rows),round(best_row['cost'],3),'stable',len(valid),'distance',distance,flush=True)
   feasible=sorted([r for r in batch if r.get('feasible')],key=lambda r:r['cost'])
   if feasible:(out/'feasible_candidate.json').write_text(json.dumps(feasible[0],indent=2));print('RIGID_TWO_METRE_SEED',feasible[0]['valid_distance_m'],flush=True);break
   if(out/'stop.request').exists():break
 print('RIGID_LANDING_FIT_DONE',out,flush=True)
if __name__=='__main__':main()
