"""Regression of zero effort and physical effect of a nonzero internal curve."""
import copy
import numpy as np
from controlled_broad_jump import ROOT,build,run
from jump_effort_control import EFFORT_STEMS,effort_curve
from jump_phase_objective import phase_metrics
from jump_monitor_data import read,atomic

def main():
    base=read(ROOT/'results/controlled_broad_jump/overnight_review_2026_11_25/best_500us/control_summary.json')['parameters'];base.update(stop_after_touchdown_s=0.,record_energy_ledger=True)
    prepared=build(.0005);a,z=run(base,dt=.0005,prepared=prepared)
    zero=copy.deepcopy(base);zero['propulsion_effort_knots_fraction']={s:[0.,0.,0.]for s in EFFORT_STEMS};b,y=run(zero,dt=.0005,prepared=prepared)
    assert np.array_equal(z['qpos'],y['qpos'])and np.array_equal(z['physics_force_world_N'],y['physics_force_world_N'])
    assert abs(a['first_contact_heel_distance_m']-2.1242995481437443)<1e-9
    effort=copy.deepcopy(zero);effort['propulsion_effort_knots_fraction']['hip_flexion']=[0.,-.05,0.]
    c,x=run(effort,dt=.0005,prepared=prepared);limits=np.array([effort['joint_torque_limits_Nm'][n]for n in prepared[3]])
    assert np.max(abs(x['physics_joint_net_torque_Nm'])-limits)<1e-6
    assert c['maximum_root_applied_force_or_torque']==0.
    assert np.max(abs(x['physics_joint_effort_feedforward_Nm']))>1.
    assert not np.allclose(a['takeoff_velocity_m_s'],c['takeoff_velocity_m_s'],atol=1e-5)
    assert all(effort_curve(u,[.1,.2,.3])==0 for u in [-1,0,1,2])
    atomic(ROOT/'results/effort_interface_regression.json',dict(status='passed',zero_effort_exact_state_and_force_replay=True,original_first_contact_replay_error_m=abs(a['first_contact_heel_distance_m']-2.1242995481437443),no_root_actuation=True,all_physics_rate_net_torques_within_fixed_limits=True,baseline_takeoff_velocity_m_s=a['takeoff_velocity_m_s'],nonzero_effort_takeoff_velocity_m_s=c['takeoff_velocity_m_s'],nonzero_effort_changed_native_launch=True,effort_scope='Equivalent joint torque, not individual muscle tension',first_contact_termination_is_not_a_stable_landing_claim=True))
    print('EFFORT_INTERFACE_REGRESSION_PASSED',a['first_contact_heel_distance_m'],c.get('first_contact_heel_distance_m'),flush=True)

if __name__=='__main__':main()
