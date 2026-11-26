"""Find one force+posture initial solution, then rank all pre-contact variables.

Initialization uses CEM, not reinforcement learning. The later policy design
keeps every causally relevant coordinate active; recovery is a separate task.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,copy,json,time
import numpy as np
from controlled_broad_jump import ROOT,build,run
from jump_monitor_data import atomic,read,lines
from jump_phase_objective import phase_metrics,launch_distance_reward
from effort_launch_parameters import definition,decode,POSTURE_SPECS
from jump_effort_control import EFFORT_STEMS
from effort_launch_objective import evaluate_launch,INITIAL_CONTACT_ENVELOPE

PREPARED=None;BASE=None;ITEMS=None
def initialize(base,items):
    global PREPARED,BASE,ITEMS
    BASE=base;ITEMS=items;PREPARED=build(.0005)

def evaluate(job):
    trial,action=job;started=time.time();p=decode(BASE,ITEMS,action)
    try:
        r,z=run(p,dt=.0005,prepared=PREPARED);m=evaluate_launch(r,z,PREPARED[1]);distance=m['heel_projection_at_first_foot_contact_m'];valid=m['optimization_domain_eligible']
        cost=-float(distance)if valid else 3.+10*max(m['precontact_ROM_excess_rad']-.04,0)+10*max(.1-m['precontact_nonfoot_clearance_proxy_m'],0)
        return dict(trial=trial,action=list(map(float,action)),cost=cost,distance_m=distance,eligible=valid,metrics=m,parameters=p,wall_s=time.time()-started)
    except Exception as exc:return dict(trial=trial,action=list(map(float,action)),cost=100.,eligible=False,distance_m=None,error=repr(exc),parameters=p,wall_s=time.time()-started)

def replay(parameters,folder,dt):
    p=copy.deepcopy(parameters);p['record_energy_ledger']=True;r,z=run(p,folder,dt=dt);m=evaluate_launch(r,z,__import__('mujoco').MjModel.from_binary_path(str(folder/'control_model.mjb')));m.update(dt_s=dt,folder=str(folder))
    atomic(folder/'phase_metrics.json',m);print('EFFORT_SEED_REPLAY',dt,m['heel_projection_at_first_foot_contact_m'],m['precontact_dynamics_eligible'],flush=True);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='effort_launch_seed_v2');ap.add_argument('--generations',type=int,default=8);ap.add_argument('--population',type=int,default=24);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();out=ROOT/'results/controlled_broad_jump'/a.label;out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/controlled_broad_jump/overnight_review_2026_11_25/best_500us';base=read(source/'control_summary.json')['parameters'];base.update(seconds=6.,record_energy_ledger=False,stop_after_touchdown_s=0.,propulsion_effort_knots_fraction={stem:[0.,0.,0.]for stem in EFFORT_STEMS});prepared=build(.0005);items=definition(base,prepared[1]);n=len(items);del prepared
    atomic(out/'search_definition.json',dict(method='Full-covariance CEM initializer, not RL',source=str(source),base=base,actions=items,posture_timing_variables=len(POSTURE_SPECS),effort_variables=15,all_precontact_parameters_active=True,shared_fixed_torque_cap=base['joint_torque_limits_Nm'],force_scope='Internal equivalent joint torque residual, not individual muscle force',GRF_is_native_contact_output=True,recovery_is_separate=True,initial_contact_envelope=INITIAL_CONTACT_ENVELOPE,population=a.population,generations=a.generations))
    rng=np.random.default_rng(11252026);mean=np.zeros(n);std=np.array([.13 if x['kind']=='posture_timing'else .30 for x in items]);cov=np.diag(std**2);history=[];started=time.time()
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(base,items))as pool:
        for generation in range(a.generations):
            actions=np.clip(rng.multivariate_normal(mean,cov,a.population),-1,1);actions[0]=mean
            if history:actions[1]=min(history,key=lambda r:r['cost'])['action']
            elif len(actions)>1:actions[1]=np.zeros(n)
            batch=list(pool.map(evaluate,[(len(history)+i,x)for i,x in enumerate(actions)]));history.extend(batch)
            with(out/'trials.jsonl').open('a',encoding='utf-8')as file:
                for row in batch:file.write(json.dumps(row)+'\n')
            elite=sorted(batch+sorted(history,key=lambda r:r['cost'])[:2],key=lambda r:r['cost'])[:6];x=np.array([r['action']for r in elite]);w=np.exp(-np.arange(len(x))*.25);w/=w.sum();mu=w@x;dx=x-mu;newcov=(dx.T*w)@dx;mean=.4*mean+.6*mu;cov=.5*cov+.5*(.8*newcov+.2*np.diag(np.diag(newcov)));v,Q=np.linalg.eigh(cov);cov=(Q*np.clip(v,.04**2,.45**2))@Q.T
            best=min(history,key=lambda r:r['cost']);atomic(out/'best_candidate.json',best);atomic(out/'progress.json',dict(stage='joint_force_posture_initialization',generation=generation,physical_trials=len(history),best_eligible_distance_m=best['distance_m']if best['eligible']else None,wall_s=time.time()-started));np.savez(out/'optimizer_state.npz',mean=mean,cov=cov,generation=generation);print('EFFORT_INITIAL',generation,len(history),best['distance_m'],flush=True)
            if(out/'stop.request').exists():break
    # Refine unique leaders, retaining every original trial and rejected replay.
    candidates=[];seen=set()
    for row in sorted(history,key=lambda r:r['cost']):
        if not row['eligible']:continue
        key=tuple(np.round(row['action'],7))
        if key in seen:continue
        seen.add(key);candidates.append(row)
        if len(candidates)>=3:break
    qualified=[]
    for candidate in candidates:
        folder=out/f"refinement_trial_{candidate['trial']:05d}";cases=[replay(candidate['parameters'],folder/f'dt_{us}us',us/1e6)for us in [500,250,125]]
        distances=[r['heel_projection_at_first_foot_contact_m']for r in cases];valid=all(r['optimization_domain_eligible']for r in cases);spread=(max(distances)-min(distances))/min(distances)if all(d is not None for d in distances)else 1.
        check=dict(trial=candidate['trial'],cases=cases,passed=valid and spread<.03 and min(distances)>=2.,distance_relative_spread=spread,parameters=candidate['parameters'],robust_distance_m=min(distances)if valid else None)
        atomic(folder/'refinement.json',check)
        if check['passed']:qualified.append(check)
    assert qualified,'No new effort candidate passed. Preserve evidence and refine initialization.'
    chosen=max(qualified,key=lambda r:r['robust_distance_m']);atomic(out/'initial_solution.json',chosen);center=chosen['parameters'];local_prepared=build(.0005);local_items=definition(center,local_prepared[1]);del local_prepared
    # Paired local finite perturbations of every primary coordinate. No hidden
    # freezing of less influential variables in the proposed later policy.
    eps=.20;jobs=[]
    for i in range(n):
        for sign in [-1,1]:
            x=np.zeros(n);x[i]=sign*eps;jobs.append((2*i+(sign>0),x))
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(center,local_items))as pool:perturbations=list(pool.map(evaluate,jobs))
    sensitivity=[]
    for i,item in enumerate(local_items):
        minus,plus=perturbations[2*i:2*i+2];both=minus['eligible']and plus['eligible'];delta=(plus['distance_m']-minus['distance_m'])if both else None;score=abs(delta)/(2*eps)if both else max([abs(r['distance_m']-chosen['cases'][0]['heel_projection_at_first_foot_contact_m'])for r in [minus,plus]if r['eligible']],default=0.)/eps
        sensitivity.append(dict(**item,minus_distance_m=minus['distance_m'],plus_distance_m=plus['distance_m'],minus_eligible=minus['eligible'],plus_eligible=plus['eligible'],paired_central_difference=both,local_sensitivity_m_per_normalized_action=score,signed_central_derivative=delta/(2*eps)if both else None,perturbation_normalized=eps))
    atomic(out/'sensitivity_trials.json',perturbations);atomic(out/'sensitivity.json',sensitivity)
    posture_rank=sorted([r for r in sensitivity if r['kind']=='posture_timing'],key=lambda r:r['local_sensitivity_m_per_normalized_action'],reverse=True);major=posture_rank[:10];major_keys={r['key']for r in major}
    stds=[.04 if item['kind']=='equivalent_effort'else .04 if item['key']in major_keys else .02 for item in local_items]
    capacity=center['joint_torque_limits_Nm'];observation=[.87,.89,capacity['hip_flexion_r']/1000,capacity['knee_angle_r']/1000,capacity['ankle_angle_r']/1000,.8,.980665]
    policy=dict(status='designed_and_initialized_not_trained',algorithm='PPO-Clip, one-action-per-jump parameter policy',framewise_motor_policy=False,actor=[7,64,64,n],critic=[7,64,64,1],observation=observation,actions=local_items,base_parameters=center,major_posture_variables=major,all_52_outputs_and_log_standard_deviations_trainable=True,initial_normalized_std=stds,initial_mean='Zero offset exactly reproduces the new effort-controlled seed',episode_primary_reward='(eligible first-contact heel projection - seed_distance_m) / 0.25; invalid precontact dynamics -> -3; recovery reported separately',seed_distance_m=chosen['robust_distance_m'],physical_validity_gates='Same precontact dynamics gates as initialization; fixed shared net torque capacity, native friction/contact and joint ROM.',fixed_conditions=['mass/inertia','joint torque ceilings','PD gains','contact/friction/gravity'],training_proposal=dict(native_workers=3,n_steps_per_worker=32,batch_size=32,n_epochs=4,learning_rate=.00001,target_kl=.01,clip_range=.1,entropy_coefficient=.0001,max_grad_norm=.5,checkpoint_every_training_episodes=96),evaluation='Deterministic mean and sampled best separately, periodically replayed at500/250/125us; no recovery hard-zero on primary launch reward.',scope='Local starting policy, no claim of global optimum, human strength calibration, muscle tension or landable official record.')
    atomic(out/'policy_design.json',policy)
    scenario=dict(name='Force-and-posture standing long jump initial solution',parameters=center,refinement=chosen,policy_definition=str(out/'policy_design.json'),recovery_separate=True,individual_muscle_force_model=False)
    atomic(ROOT/'scenarios/standing_long_jump_effort_seed_v1.json',scenario)
    atomic(out/'summary.json',dict(status='initial_solution_found_and_policy_designed',initial_solution=chosen,search_trials=len(history),sensitivity_trials=len(perturbations),total_initialization_and_sensitivity_evaluations=len(history)+len(perturbations)+9,wall_s=time.time()-started,major_posture_keys=[r['key']for r in major],RL_training_episodes=0))
    print('EFFORT_SEED_AND_POLICY_DESIGN_READY',chosen['robust_distance_m'],[r['key']for r in major],flush=True)

if __name__=='__main__':main()
