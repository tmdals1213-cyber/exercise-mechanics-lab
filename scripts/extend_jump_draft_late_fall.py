"""Display-only continuation of a trial that fell at its evaluation deadline."""
import copy,subprocess
from pathlib import Path
import numpy as np
from jump_monitor_data import ROOT,read,atomic
from controlled_broad_jump import run
OUT=ROOT/'deliverables/jump_rl_study_draft_v1'

def main():
    m=read(OUT/'source_manifest.json');c=next(c for c in m['cases']if c['key']=='late_fall');source=Path(c['source']);folder=Path(c['folder']);r=read(source/'control_summary.json');p=copy.deepcopy(r['parameters'])
    p.update(stop_after_touchdown_s=3.1,recovery_evaluation_s=3.1)
    result,z=run(p,OUT/'display_continuations/late_fall',dt=r['physics_dt_s']);original=np.load(source/'control_motion.npz');idx=np.searchsorted(z['times'],original['times']);assert np.allclose(z['times'][idx],original['times'],atol=1e-9);error=float(abs(z['qpos'][idx]-original['qpos']).max());assert error<1e-10
    assert z['times'][-1]-result['first_body_floor_contact_s']>.5
    z['mode']=z['phase'];np.savez_compressed(folder/'motion.npz',**z);atomic(folder/'control_summary.json',result)
    c.update(duration_s=float(z['times'][-1]),render_dir='fixed_side_render_continued',display_continuation=dict(original_prefix_qpos_error=error,original_evaluation_end_s=float(original['times'][-1]),reason='Fall began at the evaluation deadline; same controller continued only for display. Training score unchanged.'))
    args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts/render_fixed_side_jump_blender.py'),'--','--kg','0','--folder',str(folder),'--analysis-study','--fps',str(c['render_fps']),'--world-side','--ortho-width','4.6','--camera-forward','1.85','--render-dir',c['render_dir'],'--frame-start',str(round(c['render_start_s']*c['render_fps'])),'--skip-existing']
    with(folder/'continuation_render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    atomic(OUT/'source_manifest.json',m);print('LATE_FALL_CONTINUED',error,c['duration_s'],flush=True)

if __name__=='__main__':main()
