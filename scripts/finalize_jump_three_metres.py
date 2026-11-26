"""Finish the authorized video/example after the qualified3m stop, never fake it."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy,sys,time,subprocess,shutil,os,hashlib,json
import numpy as np
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v5';OLD=ROOT/'deliverables/jump_rl_study_draft_v4';WEB=ROOT/'deliverables/jump_landing_live';RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3'

def status(stage,**kw):atomic(OUT/'finalization_status.json',dict(stage=stage,heartbeat=time.time(),**kw));print('FINALIZATION',stage,flush=True)
def command(script,*args):subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

def freeze():
    import mujoco
    validation=read(RUN/'verified_best.json');assert validation['passed']and all(r['stable']and r['eligible']and r['retained_distance_m']>=3. for r in validation['cases']);assert {r['dt_s']for r in validation['cases']}=={.0005,.00025,.000125,.0000625};record=copy.deepcopy(min(validation['cases'],key=lambda r:r['retained_distance_m']));history=[json.loads(line)for line in(RUN/'history.jsonl').read_text(encoding='utf-8').splitlines()if line.strip()];record['training_trial']=sum(r['kind']=='ppo_train'and r['finished_at']<=record['finished_at']for r in history);source=Path(record['folder']);m=copy.deepcopy(read(OLD/'source_manifest.json'));case=next(c for c in m['cases']if c['key']=='best_checked');folder=OUT/'scenes/best_checked';folder.mkdir(parents=True,exist_ok=True);old=Path(case['multiview_folder']);z=dict(np.load(source/'control_motion.npz'));z['mode']=z['phase'];np.savez_compressed(folder/'motion.npz',**z)
    for name in ['control_summary.json','body_names.json','joint_names.json','control_model.mjb']:shutil.copyfile(source/name,folder/name)
    for name in ['reference_pose.npz','standing_pose.npz']:shutil.copyfile(old/name,folder/name)
    # Spread the four comparison stages over the now wider achieved distance.
    middle=copy.deepcopy(next(c for c in read(OLD/'source_manifest.json')['cases']if c['key']=='best_checked'));middle['key']='middle_two';middle['retained_analysis_source_key']='best_checked';m['cases']=[middle if c['key']=='middle_two'else c for c in m['cases']]
    r=read(source/'control_summary.json');p=r['parameters'];summary=read(old/'summary.json');summary.update(takeoff_velocity_m_s=r['takeoff_velocity_m_s'],phase_times_s=dict(bottom=p['ready_s']+p['crouch_s'],start_ascent=p['ready_s']+p['crouch_s'],end_ascent=r['takeoff_s'],landing=r['touchdown_s']));atomic(folder/'summary.json',summary)
    native=mujoco.MjModel.from_binary_path(str(source/'control_model.mjb'));data=mujoco.MjData(native);names=read(source/'joint_names.json');ids=[native.joint(n).id for n in names];anchors=[]
    for q in z['qpos']:data.qpos[:]=q;mujoco.mj_fwdPosition(native,data);anchors.append(data.xanchor[ids].copy())
    np.savez_compressed(folder/'joint_overlay.npz',times=z['times'],anchors=np.array(anchors),angles_rad=z['qpos'][:,native.jnt_qposadr[ids]])
    x=-z['body_xpos'][:,1:,1]-.0306183966;lo=float(x.max())+.22-2.3;hi=float(x.min())-.22+2.3;assert lo<=hi;center=float(np.clip(1.7,lo,hi));case.update(source=str(source),folder=str(folder),multiview_folder=str(folder),record=record,checked_distance_m=min(c['retained_distance_m']for c in validation['cases']),duration_s=float(z['times'][-1]),render_fps=24,render_start_s=0.,render_dir='fullbody_side_render',camera_center_forward_m=center)
    m.update(version=5,stable_three_metre_complete=True,final_four_distances_m=[next(c['checked_distance_m']for c in m['cases']if c['key']==k)for k in m['final_four_keys']],best_checked_min_retained_m=case['checked_distance_m'],active_search_separate=False,final_result_replaceable=False,qualified_final_validation=str(RUN/'verified_best.json'));atomic(OUT/'source_manifest.json',m)
    atomic(OUT/'qualified_final_validation.json',validation);m['qualified_final_validation']=str(OUT/'qualified_final_validation.json');atomic(OUT/'source_manifest.json',m)
    for name in ['policy_definition.json','ANATOMY_CREDITS.md','parameter_catalogue.json','training_compatibility.json']:shutil.copyfile(OLD/name,OUT/name)
    for key in m['final_four_keys']:
        if key=='best_checked':continue
        source_key='best_checked'if key=='middle_two'else key
        for suffix in ['muscles.npz','muscles.json','metrics.npz']:shutil.copyfile(OLD/f'{source_key}_{suffix}',OUT/f'{key}_{suffix}')
        if key=='middle_two':
            info=read(OUT/f'{key}_muscles.json');info['key']=key;atomic(OUT/f'{key}_muscles.json',info)
    catalogue=read(OUT/'parameter_catalogue.json')
    for item in catalogue['parameters']:
        for c in m['cases']:
            if c['key']not in m['final_four_keys']:continue
            params=read(Path(c['source'])/'control_summary.json')['parameters'];key=item['key']
            if key in params:item['values_by_stage'][c['key']]=params[key]
            else:
                action=next(a for a in read(RUN/'definition.json')['actions']if a['key']==key)
                item['values_by_stage'][c['key']]=params['propulsion_effort_knots_fraction'][action['stem']][action['knot']]
    atomic(OUT/'parameter_catalogue.json',catalogue);state=read(RUN/'live.json');m.update(opening_snapshot_training_trials=m['actual_training_rollouts_used'],opening_snapshot_policy_updates=m['actual_completed_PPO_updates'],finalized_training_trials=state['training_episodes'],finalized_policy_updates=state['policy_updates']);atomic(OUT/'source_manifest.json',m)
    return m,case

def produce():
    status('freeze_qualified_inputs');manifest,case=freeze();command('build_reduced_jump_muscles.py','--output',OUT/'native_muscles','--manifest',OUT/'source_manifest.json','--case','best_checked');command('build_fullbody_jump_muscles_v4.py','--output',OUT,'--manifest',OUT/'source_manifest.json','--native-analysis',OUT/'native_muscles','--case','best_checked');analysis=read(OLD/'muscle_analysis.json');analysis['cases']['middle_two']=copy.deepcopy(analysis['cases']['best_checked']);analysis['cases']['middle_two']['key']='middle_two';analysis['cases']['best_checked']=read(OUT/'best_checked_muscles.json');atomic(OUT/'muscle_analysis.json',analysis);folder=Path(case['folder'])
    for source,dest in [(OUT/'best_checked_muscles.npz',folder/'muscles.npz'),(OUT/'best_checked_muscles.json',folder/'muscles.json')]:shutil.copyfile(source,dest)
    import prepare_jump_rl_draft_v2_metrics as metrics
    metrics.OUT=OUT;result=metrics.one(case);oldmetrics=read(OLD/'metrics.json');oldmetrics['cases']['middle_two']=copy.deepcopy(oldmetrics['cases']['best_checked']);oldmetrics['cases']['middle_two']['key']='middle_two';oldmetrics['cases']['best_checked']=result;atomic(OUT/'metrics.json',oldmetrics)
    status('render_final_model')
    def render(kind):
        script='render_control_jump_blender.py'if kind=='front_rear'else'render_fixed_side_jump_blender.py';args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts'/script),'--','--kg','0','--folder',str(folder),'--analysis-study','--hide-com','--fullbody-muscle-contours','--fps','24','--skip-existing']
        args+=['--matched-person-scale','--paired-rear','--render-dir','front_rear_render']if kind=='front_rear'else['--world-side','--ortho-width','4.6','--camera-forward',str(case['camera_center_forward_m']),'--render-dir','fullbody_side_render']
        with(folder/(kind+'_render.log')).open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    with ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(render,['front_rear','side']))
    status('compose_video');command('compose_jump_rl_draft_v5.py');status('verify_video');verify();command('export_registered_example.py','--destination',ROOT/'github_release/examples/long_jump');verify_registered_example();command('prepare_github_release.py');status('complete',video=str(OUT/'standing_long_jump_rl_final.mp4'))

def verify_registered_example():
    example=ROOT/'github_release/examples/long_jump';meta=read(example/'prepared_model.json');output=OUT/'registered_example_replay';command('analyze_example.py','--example',example,'--dt',meta['source_dt_s'],'--output',output);validation=read(OUT/'qualified_final_validation.json');source=next(r for r in validation['cases']if r['id']==meta['source_id']);a=np.load(Path(source['folder'])/'control_motion.npz');b=np.load(output/'control_motion.npz');error=float(abs(a['qpos']-b['qpos']).max());assert error<1e-10 and read(output/'analysis_score.json')['stable'];atomic(example/'reproduction_check.json',dict(passed=True,qpos_max_abs_error=error,source_retained_m=meta['expected_retained_m'],model_sha256=meta['model_sha256'],historical_rollout_data_required=False))

def verify():
    import imageio_ffmpeg
    from PIL import Image
    from compose_jump_rl_draft_v4 import BG
    delivery=read(OUT/'draft_delivery.json');m=read(OUT/'source_manifest.json');analysis=read(OUT/'muscle_analysis.json');source_checks=[]
    for key in m['final_four_keys']:
        z=np.load(OUT/f'{key}_muscles.npz');error=float(abs(np.einsum('tqn,tn->tq',z['R'],z['all_force_N'])-z['target']).max());assert error<.01;source_checks.append(dict(key=key,moment_balance_error_Nm=error))
    stream=imageio_ffmpeg.read_frames(delivery['video'],pix_fmt='rgb24');metadata=next(stream);assert tuple(metadata['size'])==(1080,1920)and metadata['fps']==30;count=0;error=0
    for raw in stream:
        frame=np.frombuffer(raw,np.uint8).reshape(1920,1080,3);error=max(error,int(abs(frame[1536:].astype(int)-np.array(BG)).max()));count+=1
    assert count==delivery['frames']and error<=4
    check=dict(passed=True,frames=count,footer_error=error,source_checks=source_checks,stable_three_metres_qualified=True,trial_count_definition=delivery['trial_provenance']['count_definition'],video_sha256=hashlib.sha256(Path(delivery['video']).read_bytes()).hexdigest());atomic(OUT/'delivery_check.json',check)
    provenance=delivery['trial_provenance'];trial=provenance['cases']['best_checked']['discovered_at_ppo_trial'];distance=m['final_four_distances_m'][-1];state=read(RUN/'live.json');extra=state.get('supplemental_search');method=f'{state["training_episodes"]:,} PPO trials'+(f' + {extra["candidate_number"]} local proposals'if extra else'');(OUT/'caption_en.txt').write_text(f'Can AI learn a better standing long jump?\nSame virtual torque limits. {method}.\n{distance:.2f} m with stable landing across four time steps.\nA model experiment, not a measured human performance.\n',encoding='utf-8')
    (OUT/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stable3m jump · final</title><style>body{{background:#0c1827;color:#e9eff5;font:18px/1.6 system-ui;max-width:1000px;margin:auto;padding:24px}}video{{width:100%;max-height:85vh}}a{{color:#77dac7}}</style><h1>{distance:.2f} m · stable landing</h1><p>Discovered at V3 PPO trial {trial:,}. Training stopped after {state['training_episodes']:,} completed trials / {state['policy_updates']} policy updates. Verification, CEM and replays are counted separately.</p><video controls playsinline src="/rl_draft_v5.mp4"></video><p><a href="/rl_draft_v5.mp4" download>Download video</a> · <a href="/rl_draft_v5/trial_provenance.json">Four-case trial counts</a> · <a href="/rl_draft_v5/delivery_check.json">Checks</a></p><p>The approved opening is followed by whole-body conditional tension, four synchronized fixed-side jumps and four oblique views. Phase alignment aids comparison; it does not alter native motion. Muscle-force allocation remains a conditional model result.</p>''',encoding='utf-8')
    if extra:
        page=OUT/'index.html';text=page.read_text(encoding='utf-8');text=text.replace(f'Discovered at V3 PPO trial {trial:,}.',f'Qualified by robust local proposal {extra["candidate_number"]} after PPO.');page.write_text(text,encoding='utf-8')
    for name in ['standing_long_jump_rl_final.mp4','trial_provenance.json','delivery_check.json','caption_en.txt','preview_four_side.png']:
        dest=ROOT/'github_release/examples/long_jump'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/name,dest)

def main():
    OUT.mkdir(parents=True,exist_ok=True);status('waiting_for_qualified_three_metres')
    while True:
        s=read(WEB/'status.json',{});v=s.get('verified')or{}
        if s.get('status')=='goal_found'and v.get('min_fine_retained_m',0.)>=3. and set(v.get('timesteps_s',[]))=={.0005,.00025,.000125,.0000625}:break
        if s.get('status')in ['failed','budget_finished']:status('goal_not_yet_qualified',training_status=s['status'],checked_m=v.get('min_fine_retained_m'));return
        atomic(OUT/'finalization_status.json',dict(stage='waiting_for_qualified_three_metres',heartbeat=time.time(),checked_m=v.get('min_fine_retained_m'),trials=s.get('training_episodes')));time.sleep(5)
    try:produce()
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
