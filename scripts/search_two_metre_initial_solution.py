"""Joint posture/timing CEM initialization search; native unactuated-root physics.

All 43 posture/timing coordinates vary jointly. Body, gains, contact and initial
state are fixed. This initialization optimizer is separate from later PPO.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,os,time,math
import numpy as np
from controlled_broad_jump import ROOT,DEFAULT,build,run
from jump_parameter_rl_worker import FULL_SPECS,parameter_bounds

PREPARED=None;BASE=None;SPECS=None;BOUNDS=None;DT=None;PRIORITIZE=False;CHECK_FINE=False;FINE_PREPARED=None
def initialize(base,dt):
 global PREPARED,BASE,SPECS,BOUNDS,DT
 os.environ['OPENBLAS_NUM_THREADS']='1';BASE=base;DT=dt;PREPARED=build(dt);SPECS=FULL_SPECS;BOUNDS=[parameter_bounds(PREPARED[1],k,h,u,BASE[k])for k,h,u in SPECS]

def decode(x):
 p=BASE.copy()
 for v,(key,_,_),(lo,hi)in zip(x,SPECS,BOUNDS):p[key]=float(BASE[key]+v*((hi-BASE[key])if v>=0 else(BASE[key]-lo)))
 return p

def evaluate(job):
 global FINE_PREPARED,PREPARED,DT,CHECK_FINE
 trial,x=job;p=decode(x)
 try:
  r,z=run(p,dt=DT,prepared=PREPARED);t=z['times'];late=t>t[-1]-.5;late_speed=float(np.linalg.norm(z['velocity'][late],axis=1).max());stable=bool(r['stable_landing']and late_speed<.15);td=r['touchdown_s'];to=r['takeoff_s'];d=r.get('first_contact_heel_distance_m');valid=r.get('valid_contact_distance_m')if stable else None;v=np.array(r.get('takeoff_velocity_m_s',[0,0,0]));capture=r.get('landing_capture_error_m',1.5);tilt=abs(r.get('landing_torso_pitch_rad',2.));clear=r['minimum_head_torso_pelvis_clearance_proxy_m'];end_speed=float(np.linalg.norm(r['final_COM_velocity_m_s']));end_z=float(z['com'][-1,2]);flight=(t>to)&(t<td if td is not None else True)if to is not None else np.zeros(len(t),bool);peakqd=float(abs(z['qvel'][flight,3:]).max())if flight.any()else 0.;ft=z['physics_times_s'];ff=z['physics_force_world_N'][:,2];pushpeak=float(ff[ft<to].max()/(87*9.80665))if to is not None else 0.;work=r['net_actuator_work_J'];fw=work['flight']['positive'];gw=work['ground']['positive'];rom=float(np.maximum(z['qpos'][:,3:]-PREPARED[1].jnt_range[3:,1],PREPARED[1].jnt_range[3:,0]-z['qpos'][:,3:]).max());dist=float(d)if d is not None else -1.
  # Smooth progress terms rank failed landings without awarding them records.
  cost=5*max(2.05-dist,0)**2+2*min(capture,2)**2+.7*tilt**2+1.5*max(.90-end_z,0)**2+.10*min(end_speed,6)**2+2*max(.12-clear,0)**2
  if PRIORITIZE and stable:cost+=10*max(2.05-float(valid),0)-5*max(2.05-dist,0)**2
  if not stable:cost+=1.5
  else:cost-=2.+.2*float(valid)
  if to is None or td is None:cost+=6.
  cost+=.000006*max(fw-450,0)**2+.000001*max(gw-1400,0)**2+.015*max(peakqd-18,0)**2+.25*max(pushpeak-4.5,0)**2+15*max(rom-.08,0)**2
  feasible=bool(stable and valid is not None and valid>=2.0 and clear>.10 and peakqd<23 and fw<650 and pushpeak<5.5 and rom<.12)
  refinement=None;primary_cost=cost;robust_stable=stable
  if stable and CHECK_FINE:
   if FINE_PREPARED is None:FINE_PREPARED=build(DT/2)
   original=PREPARED,DT,CHECK_FINE
   try:
    PREPARED=FINE_PREPARED;DT=DT/2;CHECK_FINE=False;fine=evaluate(job)
   finally:PREPARED,DT,CHECK_FINE=original
   refinement={key:fine.get(key)for key in ['cost','stable','valid_distance_m','first_heel_m','late_max_speed','capture_error_m','landing_tilt_rad','max_ROM_excess_rad','propulsion_peak_BW']};refinement['dt_s']=DT/2;cost=max(cost,fine['cost']);robust_stable=bool(stable and fine.get('stable'));feasible=bool(feasible and fine.get('feasible'))
  return dict(trial=trial,cost=float(cost),objective_version=3 if CHECK_FINE else 2 if PRIORITIZE else 1,cost_primary=primary_cost,robust_stable=robust_stable,refinement=refinement,stable=stable,feasible=feasible,valid_distance_m=valid,first_heel_m=d,takeoff_velocity_m_s=v.tolist(),capture_error_m=capture,landing_tilt_rad=tilt,late_max_speed=late_speed,end_z=end_z,clearance_m=clear,flight_peak_joint_velocity_rad_s=peakqd,propulsion_peak_BW=pushpeak,flight_positive_net_work_J=fw,ground_positive_net_work_J=gw,max_ROM_excess_rad=rom,seconds=r['compute_seconds'],action=list(map(float,x)),parameters=p)
 except Exception as e:return dict(trial=trial,cost=100.,feasible=False,stable=False,error=repr(e),action=list(map(float,x)),parameters=p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--generations',type=int,default=80);ap.add_argument('--population',type=int,default=32);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--resume',action='store_true');ap.add_argument('--recenter-best',action='store_true');ap.add_argument('--prioritize-two-metres',action='store_true');ap.add_argument('--robust',action='store_true');ap.add_argument('--base-summary');ap.add_argument('--dt',type=float,default=.0005);ap.add_argument('--initial-std',type=float,default=.35);ap.add_argument('--warm-state');ap.add_argument('--label',default='two_metre_initial_solution_v18');a=ap.parse_args();out=ROOT/'results/controlled_broad_jump'/a.label;out.mkdir(parents=True,exist_ok=True);base=DEFAULT.copy();base.update(json.loads((ROOT/'results/controlled_broad_jump/normal_base_v13/candidate_01/control_summary.json').read_text())['parameters']);base.update(seconds=6.,flight_transition_s=.16,prepare_transition_s=.16,flight_hip=1.3,flight_knee=1.15,prepare_landing_s=.18,landing_transition_s=.20,landing_ankle=.28,landing_trunk=.15,landing_shoulder=.8,hip_extension_duration_fraction=1.,knee_extension_duration_fraction=1.);
 if a.base_summary:
  base=DEFAULT.copy();base.update(json.loads(Path(a.base_summary).read_text())['parameters']);base.update(seconds=6.,record_energy_ledger=False)
 n=len(FULL_SPECS);rng=np.random.default_rng(20261124);mean=np.zeros(n);cov=np.eye(n)*a.initial_std**2;history=[];best=None;start_generation=0
 if a.warm_state:
  warm=np.load(a.warm_state);mean=warm['mean'];cov=.6*warm['cov']+.4*np.eye(n)*.20**2;assert mean.shape==(n,)
 # Broader initialization bounds than the later small-step policy neighborhood.
 spans={'flight_hip':.5,'flight_knee':.5,'flight_shoulder':.6,'flight_trunk':.18,'prepare_hip':.35,'prepare_knee':.3,'prepare_shoulder':.6,'prepare_trunk':.2,'prepare_landing_s':.09,'flight_transition_s':.05,'prepare_transition_s':.05,'landing_hip':.35,'landing_knee':.35,'landing_transition_s':.08,'landing_ankle':.3,'landing_trunk':.2,'landing_shoulder':.35,'arm_delay_s':.08,'prepush_lean_s':.10,'push_s':.045,'lean_m':.1,'hip_top':.2,'trunk_bottom':.12}
 # FULL_SPECS is rebound in parent and each spawned initializer via passed data.
 specs=[(k,spans.get(k,h),u)for k,h,u in FULL_SPECS]
 if a.resume and(out/'optimizer_state.npz').exists():
  state=np.load(out/'optimizer_state.npz');mean=state['mean'];cov=state['cov'];start_generation=int(state['generation'])+1;history=[json.loads(x)for x in(out/'trials.jsonl').read_text().splitlines()];best=min(history,key=lambda x:x['cost']);rng.bit_generator.state=json.loads((out/'rng_state.json').read_text())
  if a.prioritize_two_metres:
   for row in history:
    if row.get('objective_version',1)==1:
     row['cost_objective_v1']=row['cost']
     if row.get('stable'):row['cost']+=10*max(2.05-row['valid_distance_m'],0)-5*max(2.05-row['first_heel_m'],0)**2
     row['objective_version']=2
   (out/'trials.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in history));best=min(history,key=lambda row:row['cost'])
  if a.recenter_best:
   mean=np.array(best['action']);cov=.6*cov+.4*np.eye(n)*.20**2;(out/f'restart_generation_{start_generation:03d}.json').write_text(json.dumps(dict(reason='Retain the best working seed and locally refine all coordinates; retain broad exploration samples.',best_trial=best['trial'],starting_generation=start_generation,all_dimensions_active=n),indent=2))
 start=time.time();(out/'search_definition.json').write_text(json.dumps(dict(method='Full-covariance cross-entropy initialization search, not PPO',all_parameters_jointly_varied=n,body_kg=87,height_m=1.78,dt_s=a.dt,secondary_dt_s=a.dt/2 if a.robust else None,base=base,spans=specs,body_gains_contact_fixed=True,target='>=2m nearest supported foot contact with stable recovery and plausible motion/work; no applied root force or launch impulse.'),indent=2))
 with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize_search,initargs=(base,a.dt,specs,a.prioritize_two_metres,a.robust))as pool:
  for gen in range(start_generation,a.generations):
   x=rng.multivariate_normal(mean,cov,a.population);x[-4:]=rng.normal(mean,.32,(4,n));x=np.clip(x,-1,1);x[0]=mean
   if best is not None:x[1]=best['action']
   if gen==0:x[1]=np.zeros(n)
   jobs=[(len(history)+i,x[i])for i in range(a.population)];rows=list(pool.map(evaluate,jobs));history.extend(rows)
   with(out/'trials.jsonl').open('a')as f:
    for row in rows:f.write(json.dumps(row)+'\n')
   archive=sorted(history,key=lambda r:r['cost'])[:3];unique={tuple(r['action']):r for r in rows+archive};ranked=sorted(unique.values(),key=lambda r:r['cost']);elite=np.array([r['action']for r in ranked[:max(6,a.population//4)]]);weights=np.exp(-np.arange(len(elite))*.25);weights/=weights.sum();newmean=weights@elite;delta=elite-newmean;newcov=(delta.T*weights)@delta;mean=.45*mean+.55*newmean;cov=.4*cov+.6*(.9*newcov+.1*np.diag(np.diag(newcov)));values,vectors=np.linalg.eigh(cov);cov=(vectors*np.clip(values,.06**2,.5**2))@vectors.T
   current=min(rows,key=lambda r:r['cost'])
   if best is None or current['cost']<best['cost']:best=current;(out/'best_candidate.json').write_text(json.dumps(best,indent=2))
   np.savez(out/'optimizer_state.npz',mean=mean,cov=cov,generation=gen);(out/'rng_state.json').write_text(json.dumps(rng.bit_generator.state));valid=[r for r in history if r.get('robust_stable',r['stable'])];progress=dict(generation=gen,physical_trials=len(history),best_cost=best['cost'],best_stable_distance_m=max([min(r['valid_distance_m'],r['refinement']['valid_distance_m'])if r.get('refinement')and r['refinement'].get('valid_distance_m')is not None else r['valid_distance_m']for r in valid],default=None),best_current=current,wall_s=time.time()-start);(out/'progress.json').write_text(json.dumps(progress,indent=2));print('INITIAL_SEARCH',gen,'trials',len(history),'cost',round(best['cost'],3),'stable',sum(r['stable']for r in rows),'bestD',progress['best_stable_distance_m'],'generation_best_heel',current.get('first_heel_m'),flush=True)
   feasible=sorted([r for r in rows if r['feasible']],key=lambda r:r['cost'])
   if feasible:
    winner=feasible[0];(out/'feasible_candidate.json').write_text(json.dumps(winner,indent=2));print('FEASIBLE_TWO_METRE_SEED',winner['trial'],winner['valid_distance_m'],flush=True);break
   if(out/'stop.request').exists():print('CHECKPOINT_STOP',gen,flush=True);break
 print('INITIAL_SEARCH_FINISHED',out,flush=True)

def initialize_search(base,dt,specs,priority=False,robust=False):
 global FULL_SPECS,PRIORITIZE,CHECK_FINE
 FULL_SPECS=specs;PRIORITIZE=priority;CHECK_FINE=robust;initialize(base,dt)

if __name__=='__main__':main()
