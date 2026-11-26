"""Measure what the current parameter policy actually explores."""
from pathlib import Path
import json
import numpy as np
from jump_monitor_data import ROOT,read,atomic
from train_landing_stability import map_items

def main():
    root=ROOT/'results/controlled_broad_jump/landing_stability_v3';config=read(root/'definition.json');state=read(root/'live.json');rows=[]
    for file in root.glob('worker_*/episode_*/score.json'):
        row=read(file)
        if row and row.get('kind')=='ppo_train':rows.append(row)
    rows.sort(key=lambda row:row['finished_at']);A=np.array([row['action']for row in rows]);n=len(rows)
    stable=[(i,row)for i,row in enumerate(rows)if row['stable']and row['eligible']];best_i,best=max(stable,key=lambda pair:pair[1]['retained_distance_m'])
    update=int(state['policy_updates']);forward=read(root/f'forward_update_{update:04d}_after.json');means=np.array(forward['gaussian_mean']);std=np.array(forward['std']);_,items=map_items(config,1.);ranges=[]
    for i,item in enumerate(items):
        key=item['key'];v=A[:,i];span_left=item['center']-item['minimum'];span_right=item['maximum']-item['center'];sigma=.5*(span_left+span_right)*std[i]
        ranges.append(dict(key=key,kind=item['kind'],mean_action=float(means[i]),std_action=float(std[i]),sample_min=float(v.min()),sample_max=float(v.max()),normalized_q05=float(np.quantile(v,.05)),normalized_q95=float(np.quantile(v,.95)),nominal_lower=item['minimum'],nominal_upper=item['maximum'],physical_one_sigma=sigma,unit=item['unit']))
    batches=[]
    for start in range(0,n,48):
        chunk=rows[start:start+48];ok=[row for row in chunk if row['stable']and row['eligible']]
        batches.append(dict(from_episode=start+1,to_episode=start+len(chunk),stable_fraction=len(ok)/len(chunk),mean_reward=float(np.mean([r['reward']for r in chunk])),best_stable_m=max((r['retained_distance_m']for r in ok),default=None),fall_fraction=float(np.mean([r['fell']for r in chunk]))))
    policy_mean=[{key:row.get(key)for key in ['context','distance_m','retained_distance_m','stable','fell','reward','finished_at']}for file in root.glob('worker_*/episode_*/score.json')if(row:=read(file))and row.get('kind')=='policy_mean_eval'];policy_mean.sort(key=lambda row:row['finished_at'])
    verified=read(root/'verified_best.json');best_verified=verified['cases'][0];z=np.load(Path(best_verified['folder'])/'control_motion.npz');names=read(Path(best_verified['folder'])/'joint_names.json');caps=np.array([best_verified['parameters']['joint_torque_limits_Nm'][name]for name in names]);phase=(z['times']>=best_verified['parameters']['ready_s']+best_verified['parameters']['crouch_s'])&(z['times']<best_verified['phase']['takeoff_s']);saturation={}
    for stem in ['hip_flexion','knee_angle','ankle_angle','shoulder_elv','flex_extension']:
        idx=[i for i,name in enumerate(names)if name.startswith(stem)];ratio=abs(z['joint_net_torque_Nm'][phase][:,idx])/caps[idx];saturation[stem]=dict(max_fraction=float(ratio.max()),fraction_of_push_samples_at_98_percent_cap=float(np.mean(np.any(ratio>.98,axis=1))))
    result=dict(status='completed',training_episodes=n,policy_updates=update,parameter_outputs=58,posture_timing_parameters=43,effort_parameters=15,one_action_for_entire_jump=True,learned_framewise_state_feedback=False,rule_based_recovery_controller=True,best_training_sample_episode=best_i+1,best_training_sample_m=best['retained_distance_m'],episodes_since_best_training_sample=n-best_i-1,verified_min_retained_m=state['verified']['min_fine_retained_m'],maximum_abs_sampled_action=float(abs(A).max()),fraction_coordinates_with_abs_action_ge_08=float(np.mean(abs(A)>=.8)),median_std_action=float(np.median(std)),max_abs_policy_mean_action=float(abs(means).max()),ranges=ranges,batches=batches,recent_policy_means=policy_mean[-9:],best_verified_push_torque_usage=saturation,fixed_structure=['Sagittal motion, symmetric bilateral target/effort coordinates','Single staged crouch/push/flight/landing template with low/high and transition poses','Only3 free amplitude knots per equivalent-effort curve during ground push','Fixed low-level PD/balance/contact-response gains and state-based recovery','Policy conditions on anchor blend and physical constants, not measured state at each timestep'],interpretation='A plateau here cannot establish the strength-limited/global maximum. Action coverage distinguishes conservative local sampling from a binding action-domain boundary. Further local search can test optimizer limitations without changing expressivity or strength.')
    out=ROOT/'results/jump_search_space_audit_v1';out.mkdir(exist_ok=True);atomic(out/'search_audit.json',result)
    print(json.dumps({key:result[key]for key in ['training_episodes','policy_updates','best_training_sample_episode','episodes_since_best_training_sample','verified_min_retained_m','maximum_abs_sampled_action','fraction_coordinates_with_abs_action_ge_08','median_std_action','max_abs_policy_mean_action','best_verified_push_torque_usage']},indent=2),flush=True)
    print('KEY_PARAMETER_SAMPLING',json.dumps([r for r in ranges if r['key']in ['push_s','hip_bottom','knee_bottom','ankle_delay','hip_extension_duration_fraction','prepare_landing_s','landing_knee']],indent=2),flush=True)

if __name__=='__main__':main()
