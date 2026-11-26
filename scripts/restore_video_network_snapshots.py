"""Re-export missing animation weights from the exact retained PPO checkpoints."""
import hashlib,time
import numpy as np,torch
from stable_baselines3 import PPO
from jump_monitor_data import ROOT,read,atomic

def main():
    torch.set_num_threads(1);manifest=read(ROOT/'deliverables/jump_rl_study_draft_v4/source_manifest.json');run=ROOT/'results/controlled_broad_jump/landing_stability_v3';records=[]
    for i in range(1,manifest['actual_completed_PPO_updates']+1):
        for stage,prefix in [('before','policy_sampling_before_update'),('after','policy_update')]:
            target=run/f'network_update_{i:04d}_{stage}.npz'
            if target.exists():continue
            source=run/f'{prefix}_{i:04d}.zip'
            if not source.exists():continue
            model=PPO.load(source,device='cpu');weights={k:v.detach().cpu().numpy()for k,v in model.policy.state_dict().items()};np.savez_compressed(target,**weights)
            with np.load(target)as data:
                for k,v in weights.items():assert np.array_equal(data[k],v)
            records.append(dict(target=str(target),checkpoint=str(source),checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),exact_saved_weights=True))
    atomic(ROOT/'deliverables/storage_maintenance/network_snapshot_restoration.json',dict(at=time.time(),records=records,policy_or_training_history_modified=False,forward_activation_json_untouched=True));print('NETWORK_SNAPSHOTS_RESTORED',len(records),flush=True)

if __name__=='__main__':main()
