"""Check the four-step seed in fresh processes against the saved trajectories."""
import time,copy
from pathlib import Path
import numpy as np
import train_landing_stability as trainer
from jump_monitor_data import ROOT,read,atomic

def main():
    parent=ROOT/'results/controlled_broad_jump/landing_stability_v3';out=ROOT/'results/controlled_broad_jump/robust_seed_replay_v1';web=ROOT/'deliverables/jump_robust_seed_replay';assert not out.exists();out.mkdir(parents=True);config=read(parent/'definition.json');source=read(parent/'verified_best.json');atomic(out/'definition.json',config);atomic(web/'source_validation.json',source);trainer.OUT=out;trainer.WEB=web;trainer.TARGET_FIRST=config['goal_first_contact_m'];trainer.TARGET_RETAINED=config['goal_retained_contact_m'];trainer.VERSION=config['score_version'];exp=trainer.Experiment(config,4);checks=[]
    try:
        exp.state.update(status='running',stage='fresh_seed_replay');rows=exp.batch([dict(cmd='rollout',base=r['parameters'],dt=r['dt_s'],kind='fresh_seed_replay',context=1.)for r in source['cases']])
        for original,replay in zip(source['cases'],rows):
            a=np.load(Path(original['folder'])/'control_motion.npz');b=np.load(Path(replay['folder'])/'control_motion.npz');error=float(abs(a['qpos']-b['qpos']).max())if a['qpos'].shape==b['qpos'].shape else None
            checks.append(dict(dt_s=replay['dt_s'],source_id=original['id'],replay_id=replay['id'],source_retained_m=original['retained_distance_m'],replay_retained_m=replay['retained_distance_m'],stable=replay['stable'],eligible=replay['eligible'],qpos_max_error=error))
        passed=all(c['qpos_max_error']is not None and c['qpos_max_error']<1e-10 and c['stable']and c['eligible']for c in checks);atomic(web/'check.json',dict(passed=passed,at=time.time(),checks=checks));print('FRESH_SEED_REPLAY',passed,checks,flush=True)
    finally:exp.state.update(status='checkpointed',stage='completed');exp.close()

if __name__=='__main__':main()
