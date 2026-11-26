"""Keep an already >3m launch; refine only the six postcontact controls."""
import copy
import search_robust_jump_finish as task
from jump_monitor_data import ROOT,read
KEYS=['landing_hip','landing_knee','landing_ankle','landing_trunk','landing_shoulder','landing_transition_s']

def proposals(stable,config,rng):
    source=task.PARENT/'validation_landing_stability_v3_worker_01_episode_002507_1795672039.json';report=read(source);long=report['cases'][0]['parameters'];assert all(r['eligible']and r['distance_m']>3. for r in report['cases'])
    for alpha in [1.,.5,.25,.75,1.25,-.25]:
        p=copy.deepcopy(long)
        for key in KEYS:p[key]=long[key]+alpha*(stable[key]-long[key])
        yield p,dict(method='postcontact_only_stable_pose_transfer',source=str(source),blend=alpha,launch_and_airborne_controls_unchanged=True)
    for key in KEYS:
        for delta in [-.005,.005,-.01,.01]:
            p=copy.deepcopy(long);p[key]+=delta
            yield p,dict(method='postcontact_only_coordinate_refinement',source=str(source),key=key,delta=delta,launch_and_airborne_controls_unchanged=True)

if __name__=='__main__':
    task.RUN=ROOT/'results/controlled_broad_jump/robust_postlanding_v1';task.WEB=ROOT/'deliverables/jump_robust_postlanding';task.PROGRESS_NAME='robust_postlanding_status.json';task.USE_LAUNCH_CACHE=True;task.proposals=proposals;task.main()
