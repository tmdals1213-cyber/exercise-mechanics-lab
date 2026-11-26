"""Real 58-output PPO pilot: distance, falling and recovery all feed updates."""
from pathlib import Path
import argparse,contextlib,copy,json,subprocess,sys,time
import numpy as np
from jump_monitor_data import ROOT,read,atomic

OUT=ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1'

def prepare():
    from controlled_broad_jump import build,run
    from effort_launch_parameters import definition,POSTCONTACT_ONLY
    from jump_parameter_rl_worker import FULL_SPECS,parameter_bounds
    from jump_distance_stability import score
    OUT.mkdir(parents=True,exist_ok=True)
    prepared=build(.0005,fall_contacts=True)
    candidate=copy.deepcopy(read(ROOT/'results/controlled_broad_jump/effort_launch_seed_v2/initial_solution.json')['parameters'])
    candidate.pop('stop_after_touchdown_s',None)
    candidate.update(fall_contact_model=True,fall_followthrough_s=.55,recovery_evaluation_s=2.5,seconds=7.,record_energy_ledger=False)
    items=definition(candidate,prepared[1])
    for key,span,unit in FULL_SPECS:
        if key in POSTCONTACT_ONLY:
            candidate.setdefault(key,dict(landing_ankle=.28,landing_trunk=.15,landing_shoulder=.8,landing_transition_s=.25).get(key))
            lo,hi=parameter_bounds(prepared[1],key,span,unit,candidate[key])
            items.append(dict(key=key,kind='posture_timing',center=candidate[key],minimum=lo,maximum=hi,unit=unit))
    assert len(items)==58
    old=read(ROOT/'results/controlled_broad_jump/two_metre_initial_solution_verified/verification.json')['source_parameters']
    cap=read(ROOT/'results/controlled_broad_jump/two_metre_initial_solution_verified/fixed_virtual_capacity.json')
    old.update(joint_torque_limits_Nm=cap['joint_torque_limits_Nm'],joint_torque_limit_basis=cap.get('basis','net_actuator'),fall_contact_model=True,fall_followthrough_s=.55,recovery_evaluation_s=2.5,seconds=7.,record_energy_ledger=False)
    comparisons={}
    for label,p in [('stable_reference',old),('long_distance_fall',candidate)]:
        r,z=run(p,OUT/label,dt=.0005,prepared=prepared);comparisons[label]=score(r,z,prepared[1]);print('FULL_REPLAY',label,comparisons[label]['distance_m'],comparisons[label]['reward'],r['status'],flush=True)
    original=read(ROOT/'results/controlled_broad_jump/effort_launch_seed_v2/initial_solution.json')['cases'][0]['heel_projection_at_first_foot_contact_m']
    assert abs(comparisons['long_distance_fall']['distance_m']-original)<1e-8
    assert comparisons['long_distance_fall']['body_floor_contact_s']is not None
    capacity=candidate['joint_torque_limits_Nm']
    obs=[.87,.89,capacity['hip_flexion_r']/1000,capacity['knee_angle_r']/1000,capacity['ankle_angle_r']/1000,.8,.980665]
    atomic(OUT/'definition.json',dict(base_parameters=candidate,actions=items,observation=obs,outputs=58,posture_timing_outputs=43,equivalent_effort_outputs=15,fixed_torque_limits_Nm=capacity,learning_starts_from='long_distance_fall',objective='Distance retained, stability graded, body-floor fall penalized. Six recovery variables also learn.',dt_s=.0005))
    atomic(OUT/'comparison.json',comparisons)
    print('RECOVERY_LEARNING_PREPARED',flush=True)

def worker(directory):
    from controlled_broad_jump import build,run
    from effort_launch_parameters import decode
    from jump_distance_stability import score
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);config=read(OUT/'definition.json')
    with contextlib.redirect_stdout(sys.stderr):prepared=build(.0005,fall_contacts=True)
    print('RL_WORKER_READY '+json.dumps(config),flush=True)
    count=0
    for line in sys.stdin:
        try:
            cmd=json.loads(line)
            if cmd['cmd']=='close':break
            p=decode(config['base_parameters'],config['actions'],cmd['action'])
            folder=directory/f'episode_{count:04d}'
            with contextlib.redirect_stdout(sys.stderr):r,z=run(p,folder,dt=.0005,prepared=prepared)
            result=score(r,z,prepared[1]);result.update(episode=count,kind=cmd.get('kind','train'),action=cmd['action'],parameters=p,folder=str(folder),finished_at=time.time(),compute_seconds=r['compute_seconds'],contact_distance_m=r.get('valid_contact_distance_m'))
            atomic(folder/'score.json',result)
            with(directory/'episodes.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(result)+'\n')
            print('RL_RESULT '+json.dumps(result),flush=True);count+=1
        except Exception as exc:print('RL_ERROR '+json.dumps(dict(error=repr(exc))),flush=True)

def train():
    import torch,gymnasium as gym
    from stable_baselines3.common.vec_env import VecEnv
    from train_jump_overnight import PhysicsVec,InstrumentedPPO,capture
    class RecoveryVec(PhysicsVec):
        def __init__(self):
            self.out=OUT;self.workers=[];self.logs=[];self.results=[];self.total_rollouts=0;self.best=None
            for i in range(2):
                directory=OUT/f'worker_{i:02d}';directory.mkdir(exist_ok=True);log=(directory/'worker.log').open('w',encoding='utf-8');self.logs.append(log)
                self.workers.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--directory',str(directory)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
            definitions=[self.receive(i,'RL_WORKER_READY')for i in range(2)];assert definitions[0]==definitions[1]
            self.definition=definitions[0];self.obs=np.array(self.definition['observation'],np.float32)
            VecEnv.__init__(self,2,gym.spaces.Box(-5,5,shape=(7,),dtype=np.float32),gym.spaces.Box(-1,1,shape=(58,),dtype=np.float32))
        def accept(self,results):
            self.results=results;self.total_rollouts+=len(results)
            atomic(OUT/'live.json',dict(status='training',physical_rollouts=self.total_rollouts,latest=[{k:r[k]for k in ['distance_m','stability','fell','reward']}for r in results],updated_at=time.time()))
    torch.set_num_threads(1);env=RecoveryVec()
    model=InstrumentedPPO('MlpPolicy',env,learning_rate=1e-5,n_steps=8,batch_size=16,n_epochs=4,gamma=1.,gae_lambda=1.,clip_range=.1,ent_coef=.0001,target_kl=.01,max_grad_norm=.5,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,log_std_init=float(np.log(.045))),device='cpu',seed=1125,verbose=0)
    with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    model.save(OUT/'policy_initial');capture(model,env.obs,OUT,0,'before')
    before=model.predict(env.obs,deterministic=True)[0];initial=env.evaluate(before,'initial_policy_mean')
    evidence=[]
    try:
        for update in range(1,3):
            model.learn(total_timesteps=16,reset_num_timesteps=False);model.save(OUT/f'policy_update_{update:02d}');capture(model,env.obs,OUT,update,'after')
            buffer=model.rollout_buffer
            np.savez_compressed(OUT/f'rollout_batch_update_{update:02d}.npz',**{name:getattr(buffer,name)for name in ['observations','actions','rewards','episode_starts','values','log_probs','advantages','returns']})
            action=model.predict(env.obs,deterministic=True)[0];row=env.evaluate(action,f'policy_mean_after_update_{update}')
            evidence.append(dict(update=update,training_episodes=int(model.num_timesteps),optimizer_epochs=int(model._n_updates),weight_change_L2=model.last_weight_delta,action_mean=action.tolist(),evaluation=row))
            atomic(OUT/'updates.json',evidence);print('REAL_PPO_UPDATE',update,model.last_weight_delta,row['distance_m'],row['stability'],row['reward'],flush=True)
    finally:env.close()
    rows=[]
    for directory in sorted(OUT.glob('worker_*')):
        rows.extend(json.loads(line)for line in(directory/'episodes.jsonl').read_text().splitlines())
    eligible=[r for r in rows if r['eligible']]
    pareto=[r for r in eligible if not any(q['distance_m']>=r['distance_m']and q['stability']>=r['stability']and(q['distance_m']>r['distance_m']or q['stability']>r['stability'])for q in eligible)]
    atomic(OUT/'pareto.json',pareto)
    best=max(eligible,key=lambda r:r['reward'])
    result=dict(status='completed_bounded_integration_pilot',training_episodes=32,policy_updates=2,optimizer_epochs=model._n_updates,all_58_outputs_trainable=True,fallen_training_episodes=sum(r['fell']for r in rows if r['kind']=='train'),initial_evaluation=initial,final_evaluation=evidence[-1]['evaluation'],best_observed=best,pareto_count=len(pareto),real_weight_change_each_update=[x['weight_change_L2']for x in evidence],global_optimum_claimed=False,individual_muscle_forces=False,scope='Actual native simulations and on-policy PPO; short pipeline check, not a converged optimization. All episode motions and reward components saved.')
    atomic(OUT/'pilot_summary.json',result);atomic(OUT/'live.json',dict(status='completed',training_episodes=32,policy_updates=2));print('RECOVERY_PPO_PILOT_COMPLETED',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','worker','train']);ap.add_argument('--directory');a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='worker':worker(a.directory)
    else:train()
