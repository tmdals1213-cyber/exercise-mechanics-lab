"""Use another real late failure that fits a fixed camera through the full fall."""
import copy,subprocess,shutil
from pathlib import Path
import numpy as np
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v1'

def main():
    m=read(OUT/'source_manifest.json');taken={c['record']['id']for c in m['cases']};options=[]
    for h in m['training_history']:
        if not h['fell'] or not h['eligible'] or h['id']in taken or not 2.2<h['distance_m']<2.85 or h['training_trial']>768:continue
        worker,ep=h['id'].split('_episode_');worker='worker_'+worker.rsplit('worker_',1)[-1];source=Path(m['run'])/worker/('episode_'+ep);r=read(source/'control_summary.json');delay=r['first_body_floor_contact_s']-r['touchdown_s']
        if .8<delay<1.9:options.append((delay,h,source,r))
    for delay,h,source,r in sorted(options,key=lambda x:-x[0]):
        z=dict(np.load(source/'control_motion.npz'));pos=-z['body_xpos'][:,1:,1];minimum=float(pos.min());maximum=float(pos.max())
        if maximum-minimum>4.05 or z['times'][-1]-r['first_body_floor_contact_s']<.5:continue
        break
    else:raise RuntimeError('No fitting distinct native late failure')
    c=next(c for c in m['cases']if c['key']=='late_fall');old=copy.deepcopy(c);folder=OUT/'scenes/late_fall_fitted';folder.mkdir(exist_ok=True)
    for name in ['reference_pose.npz','standing_pose.npz','summary.json']:shutil.copyfile(Path(c['folder'])/name,folder/name)
    for name in ['control_summary.json','body_names.json','joint_names.json','control_model.mjb']:shutil.copyfile(source/name,folder/name)
    z['mode']=z['phase'];np.savez_compressed(folder/'motion.npz',**z);meta=read(folder/'summary.json');p=r['parameters'];meta.update(takeoff_velocity_m_s=r['takeoff_velocity_m_s'],phase_times_s=dict(bottom=p['ready_s']+p['crouch_s'],start_ascent=p['ready_s']+p['crouch_s'],end_ascent=r['takeoff_s'],landing=r['touchdown_s']));atomic(folder/'summary.json',meta)
    record=read(source/'score.json');record['training_trial']=h['training_trial'];center=round((minimum+maximum)/2-.0306183966,3)
    c.clear();c.update(key='late_fall',source=str(source),record=record,render_fps=15,folder=str(folder),duration_s=float(z['times'][-1]),render_start_s=max(0.,p['ready_s']-.1),render_dir='fixed_side_render_v2',camera_center_forward_m=center)
    args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts/render_fixed_side_jump_blender.py'),'--','--kg','0','--folder',str(folder),'--analysis-study','--fps','15','--world-side','--ortho-width','4.6','--camera-forward',str(center),'--frame-start',str(round(c['render_start_s']*15)),'--skip-existing']
    with(folder/'render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    atomic(OUT/'late_failure_framing_audit.json',dict(replaced=old,selected_trial=h['training_trial'],unchanged_native_source=True,reason='Keep the entire fall inside a world-fixed camera at the same human scale.',body_center_forward_bounds=[minimum,maximum],camera_center_forward_m=center))
    atomic(OUT/'source_manifest.json',m);print('FITTED_NATIVE_FAILURE',h['training_trial'],minimum,maximum,center,flush=True)

if __name__=='__main__':main()
