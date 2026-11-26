"""Migrate the actual checkpoint, preserving the earlier protocol archive."""
from pathlib import Path
import copy,json
import numpy as np
from jump_monitor_data import ROOT,read,atomic

def main():
    audit=ROOT/'results/landing_controller_audit_v1';summary=read(audit/'damped_profile_summary.json');assert summary and summary['stable_cases']==summary['total_cases']==9 and summary['fixed_torque_capacities_identical']and summary['no_root_actuation']
    old=ROOT/'results/controlled_broad_jump/landing_stability_v2';config=copy.deepcopy(read(old/'definition.json'));state=read(old/'live.json');assert state['status']=='checkpointed'
    groups={}
    for row in summary['results']:
        stem=row['case'].split('_damping_probe_')[0].split('_damped_dt_')[0];groups.setdefault(stem,[]).append(row)
    robust=[]
    for stem,rows in groups.items():
        assert len(rows)==3;values=[r['retained_distance_m']for r in rows];spread=(max(values)-min(values))/max(values)
        if spread<.05:robust.append((min(values),stem,rows))
    _,stem,rows=max(robust,key=lambda q:q[0]);chosen=next(r for r in rows if r['dt_s']==.00025);stable=read(Path(chosen['folder'])/'control_summary.json')['parameters'];long=read(ROOT/'results/controlled_broad_jump/effort_launch_seed_v2/initial_solution.json')['parameters']
    profile=dict(landing_pitch_feedback=False,landing_state_recovery=True,landing_support_damping=55.,landing_ankle_support_damping=25.,landing_controller_profile='state_recovery_damped_v1',landing_feedback=True,landing_balance_feedback=True)
    for p in [stable,long]:
        p.pop('stop_after_touchdown_s',None);p.update(profile);p.update(fall_contact_model=True,fall_followthrough_s=.55,recovery_evaluation_s=2.5,seconds=7.,record_energy_ledger=False,record_controller_components=False)
    assert stable['joint_torque_limits_Nm']==long['joint_torque_limits_Nm']==config['stable_anchor']['joint_torque_limits_Nm']
    config.update(stable_anchor=stable,long_anchor=long,controller_profile='state_recovery_damped_v1',initialization_complete=True,inherited_training_episodes=state['training_episodes'],inherited_policy_updates=state['policy_updates'],warm_start_checkpoint=str(old/'policy_latest.zip'),original_budget_deadline=state['deadline'],controller_validation=str(audit/'damped_profile_summary.json'),previous_protocol_archive=str(old),initial_stable_reference=dict(source=stem,min_three_dt_retained_m=min(r['retained_distance_m']for r in rows)),long_anchor_basis='Original reproducible2.88m contact candidate, with revised postcontact controller.3.18m failed CEM cases remain in the old archive.',old_recovery_initialization_not_relabelled_as_PPO=True)
    file=ROOT/'scenarios/landing_stability_v3_definition.json';atomic(file,config);atomic(audit/'training_migration.json',dict(status='prepared',new_run='landing_stability_v3',checkpoint=str(old/'policy_latest.zip'),inherited_episodes=state['training_episodes'],remaining_training_episode_cap=4096-state['training_episodes'],original_budget_deadline=state['deadline'],profile=profile,stable_reference_min_retained_m=config['initial_stable_reference']['min_three_dt_retained_m'],strength_profile_unchanged=True,all_previous_failures_retained=True));print('CONTROLLER_FIXED_TRAINING_PREPARED',config['initial_stable_reference'],flush=True)
    from controlled_broad_jump import run
    prior=read(audit/'selected_case.json');source=Path(prior['folder']);reference=read(source/'control_summary.json');report,z=run(reference['parameters'],dt=prior['dt_s']);original=np.load(source/'control_motion.npz');qerror=float(abs(original['qpos']-z['qpos']).max());ferror=float(abs(original['physics_force_world_N']-z['physics_force_world_N']).max());assert qerror==0 and ferror==0
    atomic(audit/'legacy_default_regression.json',dict(status='passed',qpos_max_error=qerror,GRF_max_error_N=ferror,new_controller_features_default_off=True));print('LEGACY_CONTROLLER_DEFAULTS_PRESERVED',flush=True)

if __name__=='__main__':main()
