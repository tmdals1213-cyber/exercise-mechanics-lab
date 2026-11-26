"""Freeze an honest portrait-reel snapshot from actual completed PPO batches."""
from pathlib import Path
import copy,json,shutil,time
import numpy as np
from jump_monitor_data import ROOT,read,atomic

OUT=ROOT/'deliverables/jump_rl_study_draft_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True);run=ROOT/'results/controlled_broad_jump/landing_stability_v3';live=read(run/'live.json');updates=int(live['policy_updates']);used=updates*48;raw=[json.loads(line)for line in(run/'history.jsonl').read_text().splitlines()];training=[row for row in raw if row['kind']=='ppo_train'][:used];assert len(training)==used
    records=[]
    for number,small in enumerate(training,1):
        worker,episode=small['id'].split('_episode_');worker='worker_'+worker.rsplit('worker_',1)[-1];folder=run/worker/('episode_'+episode);row=read(folder/'score.json');row['training_trial']=number;records.append(row)
    valid=[r for r in records if r['eligible']];failed=[r for r in valid if r['fell']];stable=[r for r in valid if r['stable']]
    verification=read(run/'verified_best.json');hero=min(verification['cases'],key=lambda row:row['retained_distance_m']);initial=read(run/'worker_00/episode_000000/score.json')
    selected=[];taken=set()
    def add(key,row,fps=15):
        assert row['id']not in taken;sha=row['id'];taken.add(sha);selected.append(dict(key=key,source=str(row['folder']),record=row,render_fps=fps))
    add('initial',initial,30)
    add('first_fall',next(r for r in failed if r['training_trial']<=48 and r['distance_m']>2.7),30)
    add('long_fall',max((r for r in failed if r['id']not in taken),key=lambda r:r['distance_m']))
    add('late_fall',max((r for r in failed if r['id']not in taken),key=lambda r:(r.get('body_floor_contact_s')or 0)-(r['phase'].get('first_foot_contact_s')or 0)))
    add('short_fall',next(r for r in failed if r['id']not in taken and 1.7<r['distance_m']<2.2))
    add('middle_fall',next(r for r in failed if r['id']not in taken and 2.4<r['distance_m']<2.7))
    add('other_fall',next(r for r in failed if r['id']not in taken and r['training_trial']>300 and 2.1<r['distance_m']<2.5))
    add('improve_230',next(r for r in stable if r['id']not in taken and r['retained_distance_m']>2.30))
    add('improve_246',next(r for r in stable if r['id']not in taken and r['retained_distance_m']>2.45))
    add('best_checked',hero,30)
    # Raw scene sources are immutable copies. Rendering does not alter dynamics.
    for case in selected:
        source=Path(case['source']);folder=OUT/'scenes'/case['key'];folder.mkdir(parents=True,exist_ok=True)
        report=read(source/'control_summary.json');z=dict(np.load(source/'control_motion.npz'));z['mode']=z['phase'];np.savez_compressed(folder/'motion.npz',**z)
        prior=ROOT/'deliverables/iron_man_contours_v5/render_cases/jump';meta=read(prior/'summary.json');names=read(source/'body_names.json');assert names==meta['body_names']
        for name in ['reference_pose.npz','standing_pose.npz']:shutil.copyfile(prior/name,folder/name)
        for name in ['control_summary.json','body_names.json','joint_names.json','control_model.mjb']:shutil.copyfile(source/name,folder/name)
        p=report['parameters'];meta.update(movement='cmj_replay',display_platform_only=False,landing_simulated=True,body_names=names,phase_times_s=dict(bottom=p['ready_s']+p['crouch_s'],start_ascent=p['ready_s']+p['crouch_s'],end_ascent=report['takeoff_s'],landing=report['touchdown_s']),takeoff_velocity_m_s=report['takeoff_velocity_m_s'],control_pilot=True,neutral_display_not_tension=True);atomic(folder/'summary.json',meta)
        case.update(folder=str(folder),duration_s=float(z['times'][-1]),render_start_s=0. if case['key']in ['initial','best_checked']else max(0.,p['ready_s']-.1))
        assert np.array_equal(np.load(folder/'motion.npz')['qpos'],np.load(source/'control_motion.npz')['qpos'])
    # The anatomy attribution is deliberately retained as a full sidecar.
    shutil.copyfile(ROOT/'assets/fullbody/CREDITS.md',OUT/'ANATOMY_CREDITS.md')
    history=[{k:r.get(k)for k in ['training_trial','id','distance_m','retained_distance_m','stable','fell','eligible','reward','context']}for r in records]
    evidence=dict(snapshot_at=time.time(),run=str(run),actual_completed_PPO_updates=updates,actual_training_rollouts_used=used,pending_rollouts_excluded=live['training_episodes']-used,actor=[9,64,64,58],critic=[9,64,64,1],cases=selected,training_history=history,verified_best=verification,baseline_retained_m=initial['retained_distance_m'],best_checked_min_retained_m=verification['cases'][np.argmin([r['retained_distance_m']for r in verification['cases']])]['retained_distance_m'],reseed_stage=live.get('optimizer_recenter'),comparison_metrics=['vertical ground reaction / bodyweight','right hip net joint torque / Nm','right knee net joint torque / Nm','right ankle net joint torque / Nm'],individual_muscle_tensions_available=False,kinematic_video_is_not_a_force_solver=True,language='English',resolution=[1080,1920],footer_reserved_px=384,main_camera=dict(world_fixed_side=True,width_m=4.6,centre_forward_m=1.7),outcome='Draft and frozen snapshot, not final optimum. Conclusion remains open.',learning_visualization_scope='Actual parameter-policy checkpoints. Compressed neuron graph and schematic pulses; no framewise state-policy claim. CEM/guided reseed is a separate chapter and not counted as PPO updates.')
    evidence['comparison_metrics'][0]='Resultant ground-reaction magnitude / bodyweight'
    atomic(OUT/'source_manifest.json',evidence);atomic(OUT/'draft_status.json',dict(status='sources_frozen',scenes=len(selected),snapshot=used,updates=updates));print('JUMP_RL_DRAFT_SOURCES',[(c['key'],c['record']['distance_m'],c['record']['fell'])for c in selected],flush=True)

if __name__=='__main__':main()
