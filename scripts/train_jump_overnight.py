"""Checkpointed PPO with parallel, persistent native physics workers.

One Gaussian action selects all 43 posture/timing parameters of a complete jump.
No generated rewards, no imposed airborne COM path, no framewise motor policy.
"""
from pathlib import Path
import argparse, contextlib, json, os, subprocess, sys, time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from jump_monitor_data import ROOT, read, atomic, lines, event

def save_policy(model, path):
    """Only expose a finished ZIP to an unattended resume process."""
    path=Path(path).with_suffix('.zip')
    temporary=path.with_name(path.stem+'_pending.zip')
    model.save(temporary)
    os.replace(temporary,path)

class PhysicsVec(VecEnv):
    def __init__(self,source,out,capacity,dt,workers=4):
        self.out=Path(out);self.workers=[];self.logs=[];self.results=[];self.total_rollouts=0;self.best=None
        for i in range(workers):
            directory=self.out/f'worker_{i:02d}';directory.mkdir(parents=True,exist_ok=True)
            log=(directory/'worker.log').open('a',encoding='utf-8');self.logs.append(log)
            args=[sys.executable,str(ROOT/'scripts/jump_parameter_rl_worker.py'),'--source',str(source),'--out',str(directory),'--stage','full','--capacity-file',str(capacity),'--dt',str(dt)]
            process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.workers.append(process)
        definitions=[self.receive(i,'RL_WORKER_READY')for i in range(workers)]
        self.definition=definitions[0];assert all(d==self.definition for d in definitions)
        self.obs=np.array(self.definition['observation'],np.float32);n=len(self.definition['actions']);assert n==43
        super().__init__(workers,gym.spaces.Box(-5,5,shape=self.obs.shape,dtype=np.float32),gym.spaces.Box(-1,1,shape=(n,),dtype=np.float32))
        atomic(self.out/'environment.json',self.definition)
        previous=[r for i in range(workers)for r in lines(self.out/f'worker_{i:02d}/episodes.jsonl')]
        self.total_rollouts=len(previous)
        valid=[r for r in previous if r.get('stable')]
        if valid:self.best=max(r['contact_distance_m']for r in valid)

    def send(self,i,action,kind):
        self.workers[i].stdin.write(json.dumps(dict(cmd='rollout',action=np.asarray(action,float).tolist(),kind=kind))+'\n');self.workers[i].stdin.flush()
    def receive(self,i,prefix):
        process=self.workers[i]
        while True:
            line=process.stdout.readline()
            if not line:raise RuntimeError(f'Physics worker {i} exited ({process.poll()}); see worker log')
            if line.startswith('RL_ERROR '):raise RuntimeError(line)
            if line.startswith(prefix+' '):return json.loads(line[len(prefix)+1:])
    def reset(self):return np.tile(self.obs,(self.num_envs,1))
    def step_async(self,actions):
        for i,action in enumerate(actions):self.send(i,action,'train')
    def step_wait(self):
        results=[self.receive(i,'RL_RESULT')for i in range(self.num_envs)];self.accept(results)
        info=[]
        for row in results:
            item=row.copy();item['rollout_id']=item.pop('episode');item['terminal_observation']=self.obs.copy();info.append(item)
        return self.reset(),np.array([r['reward']for r in results],np.float32),np.ones(self.num_envs,bool),info
    def accept(self,results):
        self.results=results;self.total_rollouts+=len(results)
        for row in results:
            if row['stable']and(self.best is None or row['contact_distance_m']>self.best):self.best=row['contact_distance_m']
    def evaluate(self,action,kind='policy_mean_eval'):
        self.send(0,action,kind);row=self.receive(0,'RL_RESULT');self.accept([row]);return row
    def close(self):
        for p in self.workers:
            if p.poll()is None:
                with contextlib.suppress(BrokenPipeError,OSError):p.stdin.write('{"cmd":"close"}\n');p.stdin.flush()
        for p in self.workers:
            with contextlib.suppress(subprocess.TimeoutExpired):p.wait(timeout=30)
        for log in self.logs:log.close()
    def get_attr(self,attr_name,indices=None):return [getattr(self,attr_name,None)for _ in self._get_indices(indices)]
    def set_attr(self,attr_name,value,indices=None):setattr(self,attr_name,value)
    def env_method(self,method_name,*args,indices=None,**kwargs):return [None for _ in self._get_indices(indices)]
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]

class LiveCallback(BaseCallback):
    def __init__(self,out,deadline):super().__init__();self.out=out;self.deadline=deadline
    def _on_step(self):
        env=self.training_env
        atomic(self.out/'live_training.json',dict(status='training',heartbeat=time.time(),training_episodes=self.num_timesteps,physical_rollouts=env.total_rollouts,best_valid_distance_m=env.best,optimizer_epochs=self.model._n_updates,latest_checkpoint=str(self.out/'policy_latest.zip'),deadline=self.deadline,parameter_outputs=43))
        return time.time()<self.deadline and not(self.out/'stop.request').exists()

class InstrumentedPPO(PPO):
    def train(self):
        before=torch.cat([p.detach().flatten().cpu()for p in self.policy.parameters()]).numpy().copy()
        super().train()
        after=torch.cat([p.detach().flatten().cpu()for p in self.policy.parameters()]).numpy()
        self.last_weight_delta=float(np.linalg.norm(after-before))

def capture(model,obs,out,index,stage):
    """Actual weights and forward activations for the later learning animation."""
    weights={k:v.detach().cpu().numpy()for k,v in model.policy.state_dict().items()}
    np.savez_compressed(out/f'network_update_{index:04d}_{stage}.npz',**weights)
    x=torch.as_tensor(obs[None],dtype=torch.float32);activations=[obs.tolist()]
    with torch.no_grad():
        for layer in model.policy.mlp_extractor.policy_net:
            x=layer(x);activations.append(x[0].numpy().tolist())
        mean=model.policy.action_net(x)[0].numpy().tolist()
    atomic(out/f'forward_update_{index:04d}_{stage}.json',dict(observation=obs.tolist(),policy_hidden_activations=activations,gaussian_mean=mean,std=model.policy.log_std.exp().detach().numpy().tolist()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--capacity-file',required=True);ap.add_argument('--dt',type=float,default=.0005);ap.add_argument('--label',default='ppo_full_overnight_v30');ap.add_argument('--deadline',required=True,type=float);ap.add_argument('--episodes',type=int,default=65536);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--resume');ap.add_argument('--smoke-test',action='store_true');a=ap.parse_args()
    if a.smoke_test:
        assert a.label.endswith('_smoke')and a.episodes<=16,'Diagnostic runs must be separate and small.'
    rollout_steps=2 if a.smoke_test else 16;batch_size=rollout_steps*a.workers if a.smoke_test else 32
    torch.set_num_threads(1);torch.manual_seed(31);np.random.seed(31);out=ROOT/'results/controlled_broad_jump'/a.label;out.mkdir(parents=True,exist_ok=True)
    atomic(out/'training_config.json',dict(algorithm='PPO-Clip',actor=[7,32,32,43],critic=[7,32,32,1],parallel_workers=a.workers,rollout_steps_per_worker=rollout_steps,batch_size=batch_size,epochs_per_update=8,max_episodes=a.episodes,deadline=a.deadline,all_43_outputs_jointly_trainable=True,one_action_selects_complete_jump=True,dt_s=a.dt,source=a.source,capacity=a.capacity_file,initialization_search_is_not_PPO=True,diagnostic_only=a.smoke_test))
    env=None;model=None;started=time.time();status='failed';updates=lines(out/'updates.jsonl');count=len(updates)
    try:
        env=PhysicsVec(a.source,out,a.capacity_file,a.dt,a.workers)
        baseline=env.evaluate(np.zeros(43),'baseline')
        if not a.smoke_test:assert baseline['stable']and baseline['contact_distance_m']>=2.,baseline
        if a.resume:model=InstrumentedPPO.load(a.resume,env=env,device='cpu')
        else:
            model=InstrumentedPPO('MlpPolicy',env,learning_rate=5e-5,n_steps=rollout_steps,batch_size=batch_size,n_epochs=8,gamma=1.,gae_lambda=1.,clip_range=.15,ent_coef=.0005,target_kl=.02,max_grad_norm=.5,policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32]),activation_fn=torch.nn.Tanh,log_std_init=float(np.log(.06))),device='cpu',seed=31,verbose=0)
            with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
            save_policy(model,out/'policy_initial')
        initial=model.num_timesteps;callback=LiveCallback(out,a.deadline);capture(model,env.obs,out,count,'before')
        while model.num_timesteps<initial+a.episodes and time.time()<a.deadline and not(out/'stop.request').exists():
            old=model.num_timesteps;before_count=model._n_updates
            model.learn(total_timesteps=rollout_steps*a.workers,reset_num_timesteps=False,callback=callback)
            # On a deadline callback SB3 may have a partial, untrained rollout.
            trained=model._n_updates>before_count
            save_policy(model,out/'policy_latest')
            if not trained or time.time()>=a.deadline:break
            mean,_=model.predict(env.obs,deterministic=True);result=env.evaluate(mean)
            info=model.logger.name_to_value
            row=dict(update=count,finished_at=time.time(),training_episodes=model.num_timesteps,physical_rollouts=env.total_rollouts,mean_policy_stable=result['stable'],mean_policy_distance_m=result['contact_distance_m'],mean_policy_reward=result['reward'],best_sample_distance_m=env.best,optimizer_epochs=model._n_updates,weight_delta_L2=model.last_weight_delta,approx_kl=float(info.get('train/approx_kl',0)),std=float(model.policy.log_std.exp().mean().detach()),policy_loss=float(info.get('train/policy_gradient_loss',0)),value_loss=float(info.get('train/value_loss',0)),clip_fraction=float(info.get('train/clip_fraction',0)))
            with(out/'updates.jsonl').open('a',encoding='utf-8')as file:file.write(json.dumps(row)+'\n')
            atomic(out/'live_training.json',dict(status='training',heartbeat=time.time(),training_episodes=model.num_timesteps,physical_rollouts=env.total_rollouts,best_valid_distance_m=env.best,optimizer_epochs=model._n_updates,latest_checkpoint=str(out/'policy_latest.zip'),deadline=a.deadline,parameter_outputs=43))
            save_policy(model,out/f'policy_{model.num_timesteps:06d}')
            if count<2:
                capture(model,env.obs,out,count,'after')
                capture(model,env.obs,out,count+1,'before')
            print('PPO_FULL_UPDATE',json.dumps(row),flush=True);count+=1
            if model.num_timesteps==old:raise RuntimeError('No physical episodes executed')
        save_policy(model,out/'policy_final');status='checkpointed_deadline'if time.time()>=a.deadline else'checkpointed_stop'if(out/'stop.request').exists()else'episode_budget_completed'
    except Exception as exc:
        event('training_error',error=repr(exc));atomic(out/'failure.json',dict(time=time.time(),error=repr(exc)))
        if model is not None:save_policy(model,out/'policy_on_error')
        raise
    finally:
        if env is not None:
            atomic(out/'live_training.json',dict(status=status,heartbeat=time.time(),training_episodes=model.num_timesteps if model else 0,physical_rollouts=env.total_rollouts,best_valid_distance_m=env.best,optimizer_epochs=model._n_updates if model else 0,latest_checkpoint=str(out/'policy_final.zip'if status!='failed'else out/'policy_latest.zip'),deadline=a.deadline,parameter_outputs=43,wall_s=time.time()-started))
            env.close()

if __name__=='__main__':main()
