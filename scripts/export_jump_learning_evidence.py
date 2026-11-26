from pathlib import Path
import json,numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from recover_jump_ppo_batch import SetupOnlyEnv
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'results/controlled_broad_jump/ppo_landing_pilot_v1';B=ROOT/'results/controlled_broad_jump/ppo_landing_pilot_v2';OUT=ROOT/'deliverables/standing_long_jump_learning'

def rows(folder):return[json.loads(x)for x in(folder/'episodes.jsonl').read_text().splitlines()]
def snapshot(path,obs,episode):
 model=PPO.load(path,device='cpu');policy=model.policy;weights=[];biases=[]
 for layer in [policy.mlp_extractor.policy_net[0],policy.mlp_extractor.policy_net[2],policy.action_net]:weights.append(layer.weight.detach().numpy().tolist());biases.append(layer.bias.detach().numpy().tolist())
 with torch.no_grad():
  x=torch.as_tensor(obs[None]);features=policy.extract_features(x);h1=torch.tanh(policy.mlp_extractor.policy_net[0](features));h2=torch.tanh(policy.mlp_extractor.policy_net[2](h1));mean=policy.action_net(h2);value=policy.predict_values(x)
 return dict(episode=episode,path=str(path),weights=weights,biases=biases,activations=[obs.tolist(),h1[0].numpy().tolist(),h2[0].numpy().tolist(),mean[0].numpy().tolist()],critic_value=float(value.item()),standard_deviation=policy.log_std.detach().exp().numpy().tolist(),parameter_vector=np.concatenate([p.detach().numpy().ravel()for p in policy.parameters()]))

def replay_gradient(before,after,records,obs,index):
 env=DummyVecEnv([lambda:SetupOnlyEnv(obs,len(records[0]['action']))]);model=PPO.load(before,env=env,device='cpu');model.learn(total_timesteps=0);buffer=model.rollout_buffer;buffer.reset()
 for r in records:
  action=np.array(r['action'],np.float32)[None]
  with torch.no_grad():value,logp,_=model.policy.evaluate_actions(torch.as_tensor(obs[None]),torch.as_tensor(action))
  buffer.add(obs[None],action,np.array([r['reward']],np.float32),np.array([True]),value,logp)
 buffer.compute_returns_and_advantage(torch.zeros(1),np.array([True]));model.train();target=PPO.load(after,device='cpu');errors=[float(abs(x-y).max())for x,y in zip(model.policy.parameters(),target.policy.parameters())];gradients={n.replace('.','__'):p.grad.detach().numpy()for n,p in model.policy.named_parameters()if p.grad is not None};np.savez_compressed(OUT/f'backprop_replay_{index:02d}.npz',**gradients);losses={k:float(v)for k,v in model.logger.name_to_value.items()if isinstance(v,(float,int,np.floating))};return dict(saved_checkpoint_max_weight_error=max(errors),replayed_gradients_match_original_update=max(errors)<1e-5,gradient_L2=float(np.sqrt(sum(float((v*v).sum())for v in gradients.values()))),losses=losses,gradient_file=f'backprop_replay_{index:02d}.npz',source='Recomputed PPO backward pass on saved actual on-policy rollouts; numerical comparison to the original saved after-checkpoint is recorded.')

def main():
 torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);definition=json.loads((B/'environment.json').read_text());obs=np.array(definition['observation'],np.float32);r1=[x for x in rows(A)if x['kind']=='train'];r2=[x for x in rows(B)if x['kind']=='train'];assert len(r1)==8 and len(r2)==24;allrows=r1+r2
 for i,r in enumerate(allrows,1):r['training_trial']=i
 checkpoints=[(0,A/'policy_initial.zip'),(8,A/'policy_0008_recovered.zip'),(16,B/'policy_0016.zip'),(24,B/'policy_0024.zip'),(32,B/'policy_0032.zip')];snaps=[snapshot(p,obs,n)for n,p in checkpoints];updates=[]
 for i in range(4):
  delta=snaps[i+1]['parameter_vector']-snaps[i]['parameter_vector'];item=dict(before_episode=checkpoints[i][0],after_episode=checkpoints[i+1][0],weight_delta_L2=float(np.linalg.norm(delta)),batch=allrows[i*8:(i+1)*8],mean_reward=float(np.mean([r['reward']for r in allrows[i*8:(i+1)*8]])))
  if i<2:item['backprop_evidence']=replay_gradient(checkpoints[i][1],checkpoints[i+1][1],allrows[i*8:(i+1)*8],obs,i+1)
  updates.append(item)
 for s in snaps:s.pop('parameter_vector')
 baseline=rows(A)[0]['contact_distance_m'];best=max([r for r in allrows if r['stable']],key=lambda r:r['contact_distance_m']);progress=json.loads((B/'learning_progress.json').read_text());summary=json.loads((B/'training_summary.json').read_text());bundle=dict(architecture=dict(actor=[7,32,32,8],critic=[7,32,32,1],activation='Tanh',action_distribution='Diagonal Gaussian',algorithm='PPO-Clip / Adam'),environment=definition,checkpoints=snaps,updates=updates,episodes=allrows,policy_evaluations=progress,baseline_distance_m=baseline,best=best,best_gain_percent=(best['contact_distance_m']/baseline-1)*100,summary=summary,scope='Landing-only parameter policy. The full takeoff state is fixed. Best sampled candidate and deterministic policy evaluation are separate.',visualization='Neuron activations and before/after weights are real saved data. Flow pulses are a schematic illustration of computation direction; update interpolation is display-only.')
 (OUT/'learning_evidence.json').write_text(json.dumps(bundle,indent=2));print('LEARNING_EVIDENCE_READY',baseline,best['contact_distance_m'],[u.get('backprop_evidence',{}).get('saved_checkpoint_max_weight_error')for u in updates],flush=True)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[2,1]});n=np.arange(1,33);valid=np.array([r['stable']for r in allrows]);dist=np.array([r['contact_distance_m']if r['stable']else np.nan for r in allrows]);bestline=np.maximum.accumulate(np.r_[baseline,np.where(valid,dist,-np.inf)])[1:];axes[0].scatter(n[valid],dist[valid],label='Valid sampled jump',color='#527f9d',s=26);axes[0].plot(n,bestline,color='#e98037',label='Best found');axes[0].axhline(baseline,color='#999999',ls='--',label='Starter');axes[0].plot([p['training_episodes']for p in progress],[p['policy_mean_distance_m']if p['policy_mean_valid']else np.nan for p in progress],'-s',color='#375c40',label='Deterministic policy');axes[0].set_ylabel('Nearest foot contact distance (m)');axes[0].legend(loc='upper left');axes[0].grid(alpha=.2);axes[1].scatter(n,[r['reward']for r in allrows],c=['#527f9d'if r['stable']else'#c85b52'for r in allrows]);axes[1].axhline(0,color='#aaaaaa',lw=.8);axes[1].set(xlabel='Physical training episode',ylabel='Reward');axes[1].grid(alpha=.2);fig.suptitle('PPO pilot: fixed takeoff, flight and landing parameters only');fig.tight_layout();fig.savefig(OUT/'learning_progress.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
