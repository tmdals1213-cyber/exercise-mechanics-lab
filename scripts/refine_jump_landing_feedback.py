"""Tune postcontact active feedback under unchanged net torque ceilings.

These are servo feedback settings, not bone/tendon material damping. The
58-output PPO remains distinct from this supplemental controller refinement.
"""
import copy
import search_robust_jump_finish as task
from jump_monitor_data import ROOT,read

def proposals(unused,config,rng):
    source=task.PARENT/'validation_landing_stability_v3_worker_01_episode_002507_1795672039.json';report=read(source);long=report['cases'][0]['parameters'];assert all(r['eligible']and r['distance_m']>3. for r in report['cases'])
    settings=[(70.,35.,300.),(85.,40.,300.),(55.,25.,150.),(70.,35.,150.),(100.,45.,200.),(65.,30.,200.),(55.,35.,300.),(75.,25.,300.)]
    for damping,ankle,trust in settings:
        p=copy.deepcopy(long);p.update(landing_support_damping=damping,landing_ankle_support_damping=ankle,landing_response_trust_Nm=trust)
        yield p,dict(method='postcontact_active_feedback_refinement',source=str(source),hip_knee_velocity_gain=damping,ankle_velocity_gain=ankle,response_trust_Nm=trust,fixed_net_torque_capacities=True,launch_and_airborne_controls_unchanged=True,additional_PPO_updates=0)

if __name__=='__main__':
    task.RUN=ROOT/'results/controlled_broad_jump/robust_landing_feedback_v1';task.WEB=ROOT/'deliverables/jump_robust_landing_feedback';task.PROGRESS_NAME='robust_landing_feedback_status.json';task.USE_LAUNCH_CACHE=True;task.proposals=proposals;task.main()
