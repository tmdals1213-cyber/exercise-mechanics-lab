"""Recover the actual first on-policy batch after an info-field logging clash."""
from pathlib import Path
import json,numpy as np,torch,gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/controlled_broad_jump/ppo_landing_pilot_v1'
class SetupOnlyEnv(gym.Env):
 def __init__(self,obs,n):self.obs=obs;self.observation_space=gym.spaces.Box(-5,5,shape=obs.shape,dtype=np.float32);self.action_space=gym.spaces.Box(-1,1,shape=(n,),dtype=np.float32)
 def reset(self,seed=None,options=None):super().reset(seed=seed);return self.obs.copy(),{}
 def step(self,action):raise RuntimeError('This environment initializes a buffer only; physics data must come from recorded real rollouts.')
def main():
 torch.set_num_threads(1);definition=json.loads((OUT/'environment.json').read_text());rows=[json.loads(line)for line in(OUT/'episodes.jsonl').read_text().splitlines()];batch=[r for r in rows if r['kind']=='train'];assert len(batch)==8;obs=np.array(definition['observation'],np.float32);n=len(definition['actions']);assert max(abs(x)for r in batch for x in r['action'])<.9999
 env=DummyVecEnv([lambda:SetupOnlyEnv(obs,n)]);model=PPO.load(OUT/'policy_initial.zip',env=env,device='cpu');model.learn(total_timesteps=0);initial=np.concatenate([p.detach().numpy().ravel()for p in model.policy.parameters()]);buffer=model.rollout_buffer;buffer.reset()
 for row in batch:
  action=np.array(row['action'],np.float32)[None]
  with torch.no_grad():values,logp,_=model.policy.evaluate_actions(torch.as_tensor(obs[None]),torch.as_tensor(action))
  buffer.add(obs[None],action,np.array([row['reward']],np.float32),np.array([True]),values,logp)
 buffer.compute_returns_and_advantage(torch.zeros(1),np.array([True]));model.num_timesteps=8;model.train();model.save(OUT/'policy_0008_recovered');final=np.concatenate([p.detach().numpy().ravel()for p in model.policy.parameters()]);change=float(np.linalg.norm(final-initial));assert change>0;report=dict(status='recovered_actual_on_policy_batch',episodes=8,optimizer_updates=model._n_updates,weight_change_L2=change,no_synthetic_rewards_or_physics=True,reason='An integer episode ID collided with SB3 reserved Monitor episode info. Gym info now uses rollout_id.',source='episodes.jsonl',checkpoint='policy_0008_recovered.zip');(OUT/'batch_recovery.json').write_text(json.dumps(report,indent=2));print('ACTUAL_PPO_BATCH_RECOVERED',change,flush=True)
if __name__=='__main__':main()
