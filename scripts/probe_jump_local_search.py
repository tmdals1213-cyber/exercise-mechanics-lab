"""Bounded CEM diagnostic in the SAME58-coordinate domain and fixed strength."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import copy,json
import numpy as np
from jump_monitor_data import ROOT,read,atomic
from train_landing_stability import map_items

OUT=ROOT/'results/jump_search_space_audit_v1';MODELS={}

def evaluate(task):
    from controlled_broad_jump import build,run
    from effort_launch_parameters import decode
    from landing_learning_objective import score
    index,base,items,action,dt,energy=task
    if dt not in MODELS:MODELS[dt]=build(dt,fall_contacts=True)
    p=decode(base,items,action);p['record_energy_ledger']=energy;folder=OUT/f'local_trial_{index:04d}_{round(dt*1e6)}us';r,z=run(p,folder,dt=dt,prepared=MODELS[dt]);result=score(r,z,MODELS[dt][1]);result.update(index=index,dt_s=dt,parameters=p,action=action,folder=str(folder),compute_seconds=r['compute_seconds'])
    if energy:
        names=read(folder/'joint_names.json');caps=np.array([p['joint_torque_limits_Nm'][name]for name in names]);ft=z['physics_times_s'];mask=(ft>=p['ready_s']+p['crouch_s'])&(ft<r['takeoff_s']);usage={}
        for stem in ['hip_flexion','knee_angle','ankle_angle','shoulder_elv','flex_extension']:
            ids=[j for j,name in enumerate(names)if name.startswith(stem)];ratio=abs(z['physics_joint_net_torque_Nm'][mask][:,ids])/caps[ids];usage[stem]=dict(max_fraction=float(ratio.max()),fraction_near_cap=float(np.mean(np.any(ratio>.98,axis=1))))
        result['physics_rate_torque_usage']=usage
    atomic(folder/'score.json',result);return result

def main():
    root=ROOT/'results/controlled_broad_jump/landing_stability_v3';config=read(root/'definition.json');verified=read(root/'verified_best.json');source=next(row for row in verified['cases']if row['dt_s']==.00025);base=copy.deepcopy(source['parameters']);base['record_controller_components']=False;_,items=map_items(config,float(source.get('context',1.)or 0.))
    for item in items:
        c=base[item['key']]if item['kind']=='posture_timing'else base['propulsion_effort_knots_fraction'][item['stem']][item['knot']];assert item['minimum']-1e-9<=c<=item['maximum']+1e-9;item['center']=c
    atomic(OUT/'local_search_definition.json',dict(base=base,actions=items,same_absolute_parameter_bounds_as_existing_context=True,parameter_outputs=58,optimizer='CEM diagnostic, not PPO',strength_capacities_unchanged=True,controller_unchanged=True))
    rng=np.random.default_rng(112505);mean=np.zeros(58);std=np.full(58,.15);std[52:]=.06;leader=None;best_action=np.zeros(58);history=[];counter=0
    with ProcessPoolExecutor(max_workers=3)as pool:
        baseline=list(pool.map(evaluate,[(counter,base,items,[0.]*58,.00025,True)]))[0];counter+=1;leader=baseline;history.append(baseline);assert abs(baseline['distance_m']-source['distance_m'])<1e-8
        atomic(OUT/'recentered_baseline.json',baseline);print('LOCAL_SEARCH_BASELINE',baseline['retained_distance_m'],baseline['physics_rate_torque_usage'],flush=True)
        for generation in range(3):
            population=np.clip(rng.normal(mean,std,(12,58)),-1,1);population[0]=best_action;population[1]=0.
            jobs=[]
            for action in population:jobs.append((counter,base,items,action.tolist(),.00025,False));counter+=1
            rows=list(pool.map(evaluate,jobs));history.extend(rows)
            for row in rows:
                if row['eligible']and row['stable']and row['retained_distance_m']>leader['retained_distance_m']:leader=row;best_action=np.array(row['action'])
            order=sorted(range(len(rows)),key=lambda i:rows[i]['reward'],reverse=True);elite=population[order[:4]];mean=.3*mean+.7*elite.mean(0);std=np.maximum(.045,.5*std+.5*elite.std(0));atomic(OUT/'local_search_history.json',history);atomic(OUT/'local_search_leader.json',leader);atomic(OUT/'local_search_status.json',dict(status='running',generation=generation+1,total_generations=3,evaluations=len(history),best_stable_m=leader['retained_distance_m'],baseline_m=baseline['retained_distance_m'],PPO_updates=0))
            print('LOCAL_SEARCH_GENERATION',generation+1,leader['retained_distance_m'],flush=True)
        finalists=list(pool.map(evaluate,[(counter+i,base,items,leader['action'],dt,False)for i,dt in enumerate([.0005,.000125])]))
    cases=[leader]+finalists;qualified=all(row['eligible']and row['stable']for row in cases);values=[row['retained_distance_m']for row in cases];spread=(max(values)-min(values))/max(values);qualified=bool(qualified and spread<.05)
    summary=dict(status='completed',same_58_parameters=True,same_original_absolute_bounds=True,unchanged_controller_and_torque_capacities=True,initial_250us_stable_m=baseline['retained_distance_m'],best_250us_stable_m=leader['retained_distance_m'],qualified_at_3_timesteps=qualified,min_3dt_retained_m=min(values)if qualified else None,relative_spread=spread,evaluations=len(history)+len(finalists),generations=3,PPO_updates=0,algorithm='Recentered CEM search, not reinforcement learning',cases=cases,physics_rate_reference_torque_usage=baseline['physics_rate_torque_usage'],scope='A bounded test of local search efficacy. Improvement does not prove global optimum or adequate expressivity; no improvement does not prove a hard parameter/strength limit.')
    atomic(OUT/'local_search_summary.json',summary);atomic(OUT/'local_search_status.json',dict(status='completed',evaluations=summary['evaluations'],best_stable_m=leader['retained_distance_m'],qualified=qualified));print('LOCAL_SEARCH_COMPLETE',json.dumps({k:summary[k]for k in ['initial_250us_stable_m','best_250us_stable_m','qualified_at_3_timesteps','min_3dt_retained_m','evaluations']}),flush=True)

if __name__=='__main__':main()
