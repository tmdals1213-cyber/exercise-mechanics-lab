"""Freeze four qualified stages, real policy mean updates and distinct falls."""
from pathlib import Path
import copy,json,shutil,time
import numpy as np,mujoco
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v2'
RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3'

def folder_from_id(identifier):
    worker,episode=identifier.split('_episode_');return RUN/('worker_'+worker.rsplit('worker_',1)[-1])/('episode_'+episode)

def main():
    OUT.mkdir(parents=True,exist_ok=True);raw=[json.loads(s)for s in(RUN/'history.jsonl').read_text().splitlines()];training=[r for r in raw if r['kind']=='ppo_train'][:2400];assert len(training)==2400
    history=[];byid={}
    for i,h in enumerate(training,1):
        r=read(folder_from_id(h['id'])/'score.json');r['training_trial']=i;history.append({k:r.get(k)for k in ['id','training_trial','distance_m','retained_distance_m','stable','fell','eligible','reward','context']});byid[r['id']]=r
    validations=[]
    for p in RUN.glob('validation_*.json'):
        v=read(p)
        if v['passed']:validations.append((min(c['retained_distance_m']for c in v['cases']),v))
    # Preserve the current final slot separately from the still-running search.
    final_v=read(RUN/'before_extension_2400_verified_best.json')
    stages=[]
    for key,target in [('initial',2.19),('middle_one',2.46),('middle_two',2.60)]:
        value,v=min(validations,key=lambda pair:abs(pair[0]-target));row=copy.deepcopy(min(v['cases'],key=lambda r:r['retained_distance_m']));row['training_trial']=byid.get(row['id'],{}).get('training_trial',sum(t['finished_at']<=row['finished_at']for t in training));stages.append((key,row,value))
    row=copy.deepcopy(min(final_v['cases'],key=lambda r:r['retained_distance_m']));row['training_trial']=sum(t['finished_at']<=row['finished_at']for t in training);stages.append(('best_checked',row,min(c['retained_distance_m']for c in final_v['cases'])))
    selected=[];taken=set()
    def add(key,r,fps,checked=None,role='montage'):
        taken.add(r['id']);selected.append(dict(key=key,source=r['folder'],record=r,render_fps=fps,checked_distance_m=checked,role=role))
    for key,r,value in stages:add(key,r,48,value,'final_stage')
    means=[]
    for h in raw:
        if h['kind']=='policy_mean_eval'and h.get('context')==1. and h['finished_at']<=training[-1]['finished_at']:
            r=read(folder_from_id(h['id'])/'score.json');r['training_trial']=sum(t['finished_at']<=r['finished_at']for t in training);means.append(r)
    pair=next((means[i-1],r)for i,r in enumerate(means)if i and r['stable']and not means[i-1]['stable'])
    policy_updates=[]
    for key,r in zip(['policy_before','policy_after'],pair):
        update=min(range(1,51),key=lambda u:float(abs(np.array(read(RUN/f'forward_update_{u:04d}_after.json')['gaussian_mean'])-np.array(r['action'])).max()))
        error=float(abs(np.array(read(RUN/f'forward_update_{update:04d}_after.json')['gaussian_mean'])-np.array(r['action'])).max());assert error<1e-5,(update,error);r['checkpoint_update']=update;policy_updates.append(update);add(key,r,16,role='policy_example')
    assert policy_updates[1]>policy_updates[0]and not(policy_updates[0]<16<policy_updates[1]),'Do not cross a guided restart in the demonstration'
    failures=[r for r in byid.values()if r['fell']and r['eligible']and r['id']not in taken]
    # Even coverage in learning history, distinct motions and complete fall traces.
    for j,target in enumerate(np.linspace(20,2350,12),1):
        for r in sorted(failures,key=lambda r:abs(r['training_trial']-target)):
            if r['id']in taken:continue
            z=np.load(Path(r['folder'])/'control_motion.npz');rep=read(Path(r['folder'])/'control_summary.json');x=-z['body_xpos'][:,1:,1]-.0306183966
            if float(z['times'][-1])-rep['first_body_floor_contact_s']<.45 or np.ptp([float(x.min()),float(x.max())])>4.05:continue
            add(f'fall_{j:02d}',r,10);break
        else:raise RuntimeError('No distinct fitting fall')
    reference=ROOT/'deliverables/iron_man_contours_v5/render_cases/jump';base_meta=read(reference/'summary.json')
    for c in selected:
        source=Path(c['source']);folder=OUT/'scenes'/c['key'];folder.mkdir(parents=True,exist_ok=True);r=read(source/'control_summary.json');z=dict(np.load(source/'control_motion.npz'));z['mode']=z['phase'];np.savez_compressed(folder/'motion.npz',**z)
        for name in ['reference_pose.npz','standing_pose.npz']:shutil.copyfile(reference/name,folder/name)
        for name in ['control_summary.json','body_names.json','joint_names.json','control_model.mjb']:shutil.copyfile(source/name,folder/name)
        names=read(source/'body_names.json');assert names==base_meta['body_names'];p=r['parameters'];meta=copy.deepcopy(base_meta);meta.update(movement='cmj_replay',display_platform_only=False,landing_simulated=True,body_names=names,phase_times_s=dict(bottom=p['ready_s']+p['crouch_s'],start_ascent=p['ready_s']+p['crouch_s'],end_ascent=r['takeoff_s'],landing=r['touchdown_s']),takeoff_velocity_m_s=r['takeoff_velocity_m_s'],control_pilot=True,neutral_display_not_tension=True);atomic(folder/'summary.json',meta)
        x=-z['body_xpos'][:,1:,1]-.0306183966;lo=float(x.max())+.22-2.3;hi=float(x.min())-.22+2.3;assert lo<=hi,(c['key'],lo,hi);center=float(np.clip(1.7,lo,hi));c.update(folder=str(folder),duration_s=float(z['times'][-1]),render_start_s=0. if c['role']=='final_stage'else max(0.,p['ready_s']-.1),render_dir='fixed_side_render_v2',camera_center_forward_m=center)
        # Native hinge centres and native qpos indices for the torque overlays.
        model=mujoco.MjModel.from_binary_path(str(source/'control_model.mjb'));data=mujoco.MjData(model);joint_names=read(source/'joint_names.json');ids=[model.joint(n).id for n in joint_names];anchors=[]
        for q in z['qpos']:data.qpos[:]=q;mujoco.mj_fwdPosition(model,data);anchors.append(data.xanchor[ids].copy())
        np.savez_compressed(folder/'joint_overlay.npz',times=z['times'],anchors=np.array(anchors),angles_rad=z['qpos'][:,model.jnt_qposadr[ids]])
    m=dict(snapshot_at=time.time(),run=str(RUN),actual_completed_PPO_updates=50,actual_training_rollouts_used=2400,cases=selected,training_history=history,actor=[9,64,64,58],critic=[9,64,64,1],final_four_keys=[x[0]for x in stages],final_four_distances_m=[x[2]for x in stages],policy_example_updates=policy_updates,individual_muscle_tensions_available=False,com_marker_visible=False,airborne_path_revealed_only_after_touchdown=True,ghosts_per_montage=3,main_camera=dict(world_fixed_side=True,width_m=4.6,case_static_centres=True),final_result_replaceable=True,frozen_best_is_global_optimum=False,active_search_separate=True,restarts=[dict(after_trial=768,algorithm='Same-domain CEM-assisted re-center, not PPO')],language='English',resolution=[1080,1920],footer_reserved_px=384)
    atomic(OUT/'source_manifest.json',m);shutil.copyfile(ROOT/'assets/fullbody/CREDITS.md',OUT/'ANATOMY_CREDITS.md');atomic(OUT/'draft_status.json',dict(status='sources_frozen',cases=len(selected)));print('V2_SOURCES',[(c['key'],c['record']['training_trial'])for c in selected],m['final_four_distances_m'],policy_updates,flush=True)

if __name__=='__main__':main()
