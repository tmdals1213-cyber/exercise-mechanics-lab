"""Check continuous-flight gates, additive shaping, and body-rest immunity."""
import copy,json
import numpy as np,mujoco
from jump_monitor_data import ROOT,read,atomic
from landing_learning_objective import score,VERSION
from jump_phase_objective import phase_metrics

def main():
    root=ROOT/'results/controlled_broad_jump/landing_stability_v2';old=read(root/'validation_worker_00_episode_000000.json');results=[]
    for row in old['cases']:
        folder=ROOT/row['folder'];report=read(folder/'control_summary.json');z=dict(np.load(folder/'control_motion.npz'));model=mujoco.MjModel.from_binary_path(str(folder/'control_model.mjb'));value=score(report,z,model);assert value['stable']and value['eligible'],value
        original=phase_metrics(report,z,model)
        results.append(dict(dt_s=report['physics_dt_s'],distance_m=value['distance_m'],retained_distance_m=value['retained_distance_m'],stable=value['stable'],eligible=value['eligible'],new_H_drift=value['phase']['flight_angular_momentum_error_kg_m2_s'],old_event_referenced_H_difference=original['flight_angular_momentum_error_kg_m2_s'],flight_reference_s=value['phase']['flight_audit_reference_time_s'],old_event_takeoff_s=report['takeoff_s']))
    folder=ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1/long_distance_fall';report=read(folder/'control_summary.json');z=dict(np.load(folder/'control_motion.npz'));model=mujoco.MjModel.from_binary_path(str(folder/'control_model.mjb'));value=score(report,z,model);fake=copy.deepcopy(z);post=fake['times']>=report['first_body_floor_contact_s'];fake['velocity'][post]=0.;fake['angular_momentum'][post]=0.;fake['com'][post,2]=1.2;changed=score(report,fake,model);assert value['stability']==changed['stability']and value['reward']==changed['reward']and not changed['stable']
    # Under fixed dynamics and distance, later body-floor contact must produce
    # a positive recovery gradient even when every rollout eventually falls.
    delayed=copy.deepcopy(report);delayed['first_body_floor_contact_s']+=.03
    other=score(delayed,z,model);assert other['survival_fraction']>value['survival_fraction']
    baseline=read(ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1/definition.json')['base_parameters']['joint_torque_limits_Nm'];config=read(root/'definition.json');assert config['stable_anchor']['joint_torque_limits_Nm']==config['long_anchor']['joint_torque_limits_Nm']==baseline
    out=dict(status='passed',objective_version=VERSION,three_timestep_reference=results,fall_stability=value['stability'],fallen_distance_retained_m=value['distance_m'],lying_still_after_impact_cannot_improve_reward=True,fall_time_has_continuous_positive_component=True,fixed_torque_capacities_identical=True,old_default_audit_retained_for_historical_runs=True,scope='Audit reference correction uses a continuous contact-free physics interval, not relaxed error tolerances or modified motion.')
    atomic(root/'objective_audit.json',out);print('LANDING_V2_OBJECTIVE_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
