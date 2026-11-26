"""Review existing native replays using the user's launch-performance objective."""
import json
import numpy as np
import mujoco
from jump_monitor_data import ROOT,atomic,read
from jump_phase_objective import phase_metrics,launch_distance_reward

def main():
    root=ROOT/'results/controlled_broad_jump/overnight_review_2026_11_25';rows=[]
    for label in ['best_500us','best_250us','best_125us']:
        folder=root/label;report=read(folder/'control_summary.json');z=dict(np.load(folder/'control_motion.npz'));model=mujoco.MjModel.from_binary_path(str(folder/'control_model.mjb'))
        metrics=phase_metrics(report,z,model);metrics.update(label=label,dt_s=report['physics_dt_s'],proposed_launch_reward=launch_distance_reward(metrics))
        # A post-touchdown-only perturbation must not change the launch metric.
        alternate={k:(v.copy()if isinstance(v,np.ndarray)else v)for k,v in z.items()};late=alternate['times']>report['touchdown_s']+.03
        alternate['qpos'][late,3:]=100.;alternate['body_xpos'][late,:,2]=-100.
        changed_report=report.copy();changed_report.update(stable_landing=not report['stable_landing'],valid_contact_distance_m=None)
        changed=phase_metrics(changed_report,alternate,model)
        assert launch_distance_reward(changed)==launch_distance_reward(metrics),'Recovery leaked into launch reward'
        metrics['recovery_independence_regression_passed']=True
        rows.append(metrics)
    distance=[r['heel_projection_at_first_foot_contact_m']for r in rows]
    result=dict(objective='Optimize launch effort/timing for jump distance; assess recovery separately.',
        cases=rows,heel_projection_spread_m=max(distance)-min(distance),
        precontact_dynamics_passed_all_replays=all(r['precontact_dynamics_eligible']for r in rows),
        current_recovery_controller_passed_all_replays=all(r['current_controller_recovery_stable']for r in rows),
        conclusion='Repeatable takeoff/first-contact candidate; recovery is not reproduced by the present rigid-foot controller. Human unrecoverability and toe omission as sole cause are not established.',
        existing_policy_retrained=False,individual_muscle_force_control_implemented=False,
        current_foot=dict(contact_points_per_foot=6,heel_and_forefoot_contact=True,
            ankle_torque_balance_feedback=True,pre_takeoff_COP_feedback=True,
            post_touchdown_contact_response_feedback=True,
            active_MTP_joint=False,active_toe_flexor_actuator=False,
            learned_recovery_steps=False,full_3D_recovery=False),
        next_interface=dict(primary_controls=['effort magnitude over time','hip/knee/ankle extension timing','arm/trunk timing'],
            internal_action='Muscle force through anatomical moment arms, or explicitly labelled equivalent joint torque in the reduced control model.',
            GRF='Native contact output from internal actuation and contact; not an independently imposed force.',
            recovery='Finite foot support/COP and foot rollover first; toe hinge/torque or recovery stepping when needed. No arbitrary free support moment.'))
    atomic(root/'phase_objective_review.json',result)
    atomic(ROOT/'scenarios/standing_long_jump_launch_objective_v1.json',dict(primary_objective='Native heel projection at first foot contact with valid precontact dynamics',primary_metric_module='scripts/jump_phase_objective.py',recovery_qualification='Separate diagnostic, not a hard zero on launch performance',preserve_historical_supported_recovery_scores=True,official_record_claim=False,fixed_conditions=['mass/inertia','gravity/friction/contact','actuation-capacity profile'],prohibit=['prescribed launch impulse','prescribed airborne COM path','arbitrary ground-force injection','unbounded stabilizing ground moment']))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
