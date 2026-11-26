"""PPO parameter policy: one bounded posture/timing vector per physical jump.

Landing stage is a one-step episodic problem (a parameter bandit), not a
frame-by-frame motor policy. The full stage additionally requires fixed torque
capacity limits. Physics runs in a separate persistent native process.
"""
from pathlib import Path
import argparse,json,subprocess,sys,os,hashlib,time
import numpy as np,gymnasium as gym,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT=Path(__file__).resolve().parents[1]

class JumpParameterEnv(gym.Env):
 metadata={'render_modes':[]}
 def __init__(self,source,out,stage='landing',capacity_file=None):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.log=(self.out/'physics_worker.log').open('w',encoding='utf-8');args=[sys.executable,str(ROOT/'scripts/jump_parameter_rl_worker.py'),'--source',str(source),'--out',str(out),'--stage',stage]
  if capacity_file:args+=['--capacity-file',str(capacity_file)]
  self.worker=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.definition=self._read('RL_WORKER_READY');self.obs=np.array(self.definition['observation'],np.float32);self.action_space=gym.spaces.Box(-1.,1.,shape=(len(self.definition['actions']),),dtype=np.float32);self.observation_space=gym.spaces.Box(-5.,5.,shape=self.obs.shape,dtype=np.float32);self.history=[]
 def _read(self,prefix):
  while True:
   line=self.worker.stdout.readline()
   if not line:raise RuntimeError('Physics worker exited; see physics_worker.log')
   if line.startswith('RL_ERROR '):raise RuntimeError(line)
   if line.startswith(prefix+' '):return json.loads(line[len(prefix)+1:])
 def evaluate(self,action,kind='eval'):
  self.worker.stdin.write(json.dumps(dict(cmd='rollout',action=np.asarray(action,float).tolist(),kind=kind))+'\n');self.worker.stdin.flush();result=self._read('RL_RESULT');self.history.append(result);print('PPO_ROLLOUT',result['episode'],kind,'stable',result['stable'],'distance',result['contact_distance_m'],'best',result['best_valid_distance_m'],flush=True);return result
 def reset(self,seed=None,options=None):super().reset(seed=seed);return self.obs.copy(),{}
 def step(self,action):
  result=self.evaluate(action,'train');info=result.copy();info['rollout_id']=info.pop('episode');return self.obs.copy(),float(result['reward']),True,False,info
 def close(self):
  if getattr(self,'worker',None)and self.worker.poll()is None:
   self.worker.stdin.write('{"cmd":"close"}\n');self.worker.stdin.flush();self.worker.wait(timeout=10)
  if getattr(self,'log',None):self.log.close()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--episodes',type=int,default=32);ap.add_argument('--seed',type=int,default=23);ap.add_argument('--label',default='ppo_landing_pilot_v1');ap.add_argument('--stage',choices=['landing','full'],default='landing');ap.add_argument('--capacity-file');ap.add_argument('--resume');ap.add_argument('--source');ap.add_argument('--initial-std',type=float);a=ap.parse_args();assert a.episodes%8==0;torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed);out=ROOT/'results/controlled_broad_jump'/a.label;source=Path(a.source)if a.source else ROOT/'results/controlled_broad_jump/broad_jump_seed_refined_v4/candidate_01';env=JumpParameterEnv(source,out,a.stage,a.capacity_file);vec=DummyVecEnv([lambda:env]);start=time.perf_counter();progress=[]
 try:
  baseline=env.evaluate(np.zeros(env.action_space.shape),'baseline');initial_std=a.initial_std if a.initial_std is not None else(.08 if a.stage=='full'else float(np.exp(-1.6)));assert 0<initial_std<1;kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32]),activation_fn=torch.nn.Tanh,log_std_init=float(np.log(initial_std)))
  model=PPO.load(a.resume,env=vec,device='cpu')if a.resume else PPO('MlpPolicy',vec,learning_rate=.001,n_steps=8,batch_size=8,n_epochs=4,gamma=1.,gae_lambda=1.,clip_range=.15,ent_coef=.001,target_kl=.02,policy_kwargs=kwargs,device='cpu',seed=a.seed,verbose=0)
  if not a.resume:
   with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
  initial=np.concatenate([p.detach().cpu().numpy().ravel()for p in model.policy.parameters()]);model.save(out/'policy_initial')
  for count in range(8,a.episodes+1,8):
   model.learn(total_timesteps=8,reset_num_timesteps=False);model.save(out/f'policy_{model.num_timesteps:04d}');action,_=model.predict(env.obs,deterministic=True);evaluation=env.evaluate(action,'policy_mean_eval');progress.append(dict(training_episodes=model.num_timesteps,new_training_episodes=count,policy_mean_valid=evaluation['stable'],policy_mean_distance_m=evaluation['contact_distance_m'],best_sample_distance_m=evaluation['best_valid_distance_m'],updates=model._n_updates));(out/'learning_progress.json').write_text(json.dumps(progress,indent=2));print('PPO_UPDATE',count,progress[-1],flush=True)
  final=np.concatenate([p.detach().cpu().numpy().ravel()for p in model.policy.parameters()]);change=float(np.linalg.norm(final-initial));assert change>0 and model._n_updates>0;model.save(out/'policy_final');summary=dict(status='pilot_training_completed',stage=a.stage,new_training_episodes=a.episodes,cumulative_training_episodes=model.num_timesteps,resumed_from=a.resume,network='Separate 7-32-32-action Gaussian actor and 7-32-32-1 critic, Tanh',algorithm='Stable-Baselines3 PPO-Clip; one action selects a full jump parameter vector',parameter_policy_not_framewise_motor_policy=True,weight_change_L2=change,optimizer_updates=model._n_updates,baseline_contact_distance_m=baseline['contact_distance_m'],final_mean_policy=progress[-1],best_sample_is_not_the_mean_policy=True,wall_seconds=time.perf_counter()-start,torch_version=torch.__version__,SB3_version=__import__('stable_baselines3').__version__,strength_scope='Landing stage fixes the entire takeoff state; no human strength ceiling is inferred. Full stage requires one fixed virtual torque-capacity profile.',converged=False,source_motion_is_measured_human=False);(out/'training_summary.json').write_text(json.dumps(summary,indent=2));print('PPO_PILOT_READY',out,flush=True)
 finally:vec.close()
if __name__=='__main__':main()
