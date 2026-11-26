"""Four-step continuation between stable2.989m and the longer PPO sample."""
import copy
import numpy as np
import search_robust_jump_finish as task
from jump_monitor_data import ROOT,read
from effort_launch_parameters import decode
import train_landing_stability as trainer

def continuation(seed,config,rng):
    base,items=trainer.map_items(config,1.);start=trainer.encode_parameters(config,seed,1.)
    targets=[task.PARENT/'validation_landing_stability_v3_worker_00_episode_002596_1795668497.json',task.PARENT/'validation_landing_stability_v3_worker_02_episode_002449_1795669270.json']
    for file in targets:
        target=read(file)['cases'][0]['parameters'];finish=trainer.encode_parameters(config,target,1.)
        for alpha in [.25,.5,.125,.375,.625,.75,.875]:
            action=(1-alpha)*start+alpha*finish;yield decode(base,items,action),dict(method='all58_control_continuation',toward_report=str(file),fraction=alpha,changes_only_policy_coordinates=True)
    # Refine around any new four-step-stable best without changing the limits.
    for i in range(90):
        best=read(task.WEB/'best_parameters.json',seed);centre=trainer.encode_parameters(config,best,1.);direction=finish-centre;step=rng.normal(0,.002,58)
        action=np.clip(centre+[.05,.10,.20][i%3]*direction+step,-1,1)
        yield decode(base,items,action),dict(method='bounded58_continuation_refinement',fraction=[.05,.10,.20][i%3],normalized_noise=step.tolist())

if __name__=='__main__':
    task.RUN=ROOT/'results/controlled_broad_jump/robust_continuation_v1';task.WEB=ROOT/'deliverables/jump_robust_continuation';task.PROGRESS_NAME='robust_continuation_status.json';task.USE_LAUNCH_CACHE=False;task.proposals=continuation;task.main()
