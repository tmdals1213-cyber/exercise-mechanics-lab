"""Reconcile the actual PPO buffer, native rollout rewards and policy changes."""
import json
import numpy as np
from jump_monitor_data import ROOT,read,atomic

def main():
    root=ROOT/'results/controlled_broad_jump/landing_stability_v2';batch=np.load(root/'rollout_batch_update_0002.npz');history=[json.loads(line)for line in(root/'history.jsonl').read_text().splitlines()];rows=[row for row in history if row['kind']=='ppo_train'][48:96];assert len(rows)==48
    rewards=np.asarray([row['reward']for row in rows]);native=[]
    for row in rows:
        worker,episode=row['id'].split('_episode_');native.append(read(root/worker/('episode_'+episode)/'score.json'))
    reward_error=float(abs(rewards-batch['rewards'].reshape(-1)).max());assert reward_error<1e-5
    actions=np.asarray([row['action']for row in native]);saved=batch['actions']
    if saved.ndim==2:saved=saved.reshape(3,16,58).transpose(1,0,2).reshape(48,58)
    else:saved=saved.reshape(48,58)
    action_error=float(abs(actions-saved).max());assert action_error<1e-6
    returns=batch['returns']
    if returns.shape[0]==48:returns=returns.reshape(3,16).T.reshape(-1)
    else:returns=returns.reshape(-1)
    returns_error=float(abs(returns-batch['rewards'].reshape(-1)).max());assert returns_error<1e-5
    before=np.load(root/'network_update_0001_before.npz');after=np.load(root/'network_update_0002_after.npz');delta=np.linalg.norm(after['action_net.weight']-before['action_net.weight'],axis=1)+abs(after['action_net.bias']-before['action_net.bias']);assert np.all(delta>0)
    std_delta=abs(after['log_std']-before['log_std']);assert np.all(std_delta>0)
    obs=batch['observations'].reshape(-1,9);contexts=np.unique(obs[:,7]);assert len(contexts)>2
    caps=read(root/'definition.json')['stable_anchor']['joint_torque_limits_Nm'];assert all(row['parameters']['joint_torque_limits_Nm']==caps for row in native)
    from train_landing_stability import Experiment,map_items
    test=Experiment.__new__(Experiment);test.state=dict(stable_candidates=[],tested_candidates={});sample=next(row for row in native if row['eligible']);sample['retained_distance_m']=sample.get('retained_distance_m')or sample['distance_m'];test.queue_candidate(sample);test.queue_candidate(sample);assert len(test.state['stable_candidates'])==1
    test.state['tested_candidates'][test.candidate_key(sample['parameters'])]=dict(passed=False);test.state['stable_candidates']=[];test.queue_candidate(sample);assert not test.state['stable_candidates']
    import copy
    other=copy.deepcopy(sample);other['parameters']['landing_hip']+=1e-6;other['id']='independent_candidate';test.queue_candidate(other);assert len(test.state['stable_candidates'])==1 and test.state['stable_candidates'][0]['id']=='independent_candidate'
    config=read(root/'definition.json')
    for context in [0.,.25,.5,.75,1.]:
        base,items=map_items(config,context);assert len(items)==58 and base['joint_torque_limits_Nm']==caps
    result=dict(status='passed',checked_update=2,actual_training_rollouts=48,native_vs_rollout_buffer_reward_max_error=reward_error,native_vs_sampled_action_max_error=action_error,one_complete_jump_per_episode_returns_match_rewards=returns_error,all_58_mean_outputs_changed=True,all_58_standard_deviations_changed=True,anchor_contexts=contexts.tolist(),falling_training_attempts=sum(row['fell']for row in native),stable_training_attempts=sum(row['stable']and row['eligible']for row in native),fixed_torque_capacities_identical=True,duplicate_validation_candidates_suppressed=True,failed_validation_does_not_block_other_candidates=True,all_anchor_contexts_have_58_valid_actions=True,source='Actual native episode archives, actual PPO rollout batch and before/after network checkpoints. No reconstructed rewards or synthetic motions.')
    atomic(root/'learning_audit.json',result);print('ACTIVE_LANDING_LEARNING_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
