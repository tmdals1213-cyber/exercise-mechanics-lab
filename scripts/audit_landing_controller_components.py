"""Causal ablations on actual near-upright relapses; native forces unchanged."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import copy,json,os
import numpy as np
from jump_monitor_data import ROOT,read,atomic

OUT=ROOT/'results/landing_controller_audit_v1'
VARIANTS={'baseline':{},'no_contact_response':dict(landing_feedback=False),'no_ankle_balance':dict(landing_balance_feedback=False),'no_hip_pitch':dict(landing_pitch_feedback=False),'hold_landing_pose':dict(landing_hold_pose=True),'no_balance_or_pitch':dict(landing_balance_feedback=False,landing_pitch_feedback=False)}
VARIANTS.update(centroidal_no_root_pitch=dict(landing_pitch_feedback=False),state_recovery_centroidal=dict(landing_pitch_feedback=False,landing_state_recovery=True))
VARIANTS.update(state_recovery_damped=dict(landing_pitch_feedback=False,landing_state_recovery=True,landing_support_damping=55.,landing_ankle_support_damping=25.),centroidal_damped=dict(landing_pitch_feedback=False,landing_support_damping=55.,landing_ankle_support_damping=25.),state_centroidal_only_damped=dict(landing_pitch_feedback=False,landing_balance_feedback=False,landing_state_recovery=True,landing_support_damping=55.,landing_ankle_support_damping=25.))
PREPARED={}

def one(task):
    from controlled_broad_jump import build,run
    case,label=task;folder=Path(case['folder']);source=read(folder/'control_summary.json');p=copy.deepcopy(source['parameters']);p.update(VARIANTS[label]);p.update(record_controller_components=True,record_energy_ledger=True);dt=case['dt_s']
    if dt not in PREPARED:PREPARED[dt]=build(dt,fall_contacts=True)
    out=OUT/case['id']/label;report,z=run(p,out,dt=dt,prepared=PREPARED[dt]);original=np.load(folder/'control_motion.npz')
    td=source['touchdown_s'];common=np.intersect1d(original['times'],z['times']);common=common[common<=td];a=np.searchsorted(original['times'],common);b=np.searchsorted(z['times'],common);pre_error=float(abs(original['qpos'][a]-z['qpos'][b]).max());assert pre_error<1e-12
    if label=='baseline':assert np.array_equal(original['qpos'],z['qpos'])
    body=report.get('first_body_floor_contact_s');end=body-.01 if body is not None else float(z['times'][-1]);t0=case['quiet_time_s'];ft=z['physics_times_s'];mask=(ft>=t0)&(ft<=end);power=z['physics_net_joint_power_W'];work=float(np.trapezoid(power[mask],ft[mask]));energy=z['physics_energy_J'].sum(1);i=int(np.argmin(abs(ft-t0)));j=int(np.argmin(abs(ft-end)));dE=float(energy[j]-energy[i]);constraint=float(np.trapezoid(z['physics_constraint_power_W'][mask],ft[mask]));quiet=int(np.argmin(abs(z['times']-t0)));speed=np.linalg.norm(z['velocity'],axis=1);late=(z['times']>=t0)&(z['times']<=end)
    named={}
    names=read(out/'joint_names.json');da=PREPARED[dt][5];qd=z['qvel'][:,da]
    for key in ['controller_posture_Nm','controller_gravity_bias_Nm','controller_balance_Nm','controller_pitch_Nm','controller_landing_response_Nm','controller_limiter_delta_Nm','joint_velocity_feedback_Nm']:
        signed=(z[key]*qd).sum(1);named[key]=float(np.trapezoid(signed[late],z['times'][late]))
    from landing_learning_objective import score
    quality=score(report,z,PREPARED[dt][1])
    result=dict(case=case['id'],variant=label,folder=str(out),dt_s=dt,first_contact_m=report['first_contact_heel_distance_m'],first_contact_error_m=abs(report['first_contact_heel_distance_m']-source['first_contact_heel_distance_m']),precontact_qpos_max_error=pre_error,stable_landing=report['stable_landing'],strict_stable=quality['stable'],dynamics_eligible=quality['eligible'],retained_distance_m=quality['retained_distance_m'],fell=report['fell'],body_contact_s=body,quiet_time_s=t0,quiet_COM_speed_m_s=float(speed[quiet]),quiet_Hx=float(z['angular_momentum'][quiet,0]),quiet_root_angular_velocity_rad_s=float(z['qvel'][quiet,2]),max_speed_after_quiet_before_body_m_s=float(speed[late].max())if late.any()else None,signed_net_joint_work_after_quiet_J=work,total_energy_change_after_quiet_J=dE,constraint_work_after_quiet_J=constraint,energy_closure_error_J=dE-work-constraint,component_prelimit_work_J=named,maximum_root_applied_force_or_torque=report['maximum_root_applied_force_or_torque'],fixed_torque_capacities=report['parameters']['joint_torque_limits_Nm'])
    atomic(out/'component_audit.json',result);return result

def main():
    ranked=read(OUT/'relapse_cases.json');cases=[ranked[0],next(q for q in ranked if q['dt_s']==.00025)];results=[];atomic(OUT/'ablation_status.json',dict(status='running',completed=0,total=len(cases)*len(VARIANTS)))
    with ProcessPoolExecutor(max_workers=3)as pool:
        futures=[pool.submit(one,(case,label))for case in cases for label in VARIANTS]
        for future in as_completed(futures):
            row=future.result();results.append(row);atomic(OUT/'ablation_results.json',results);atomic(OUT/'ablation_status.json',dict(status='running',completed=len(results),total=len(futures)));print('CONTROLLER_ABLATION',row['case'],row['variant'],row['stable_landing'],row['body_contact_s'],row['signed_net_joint_work_after_quiet_J'],flush=True)
    atomic(OUT/'ablation_status.json',dict(status='completed',completed=len(results),total=len(results)))

if __name__=='__main__':main()
