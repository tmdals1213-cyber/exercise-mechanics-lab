"""Keep the same world scale, fit the two far-reaching fall endings."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v1'

def main():
    manifest=read(OUT/'source_manifest.json');cases=[c for c in manifest['cases']if c['key']in ['late_fall','short_fall']]
    def render(case):
        folder=Path(case['folder']);args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts/render_fixed_side_jump_blender.py'),'--','--kg','0','--folder',str(folder),'--analysis-study','--fps',str(case['render_fps']),'--world-side','--ortho-width','4.6','--camera-forward','1.85','--render-dir','fixed_side_render_fit','--frame-start',str(round(case['render_start_s']*case['render_fps'])),'--skip-existing']
        with(folder/'fit_render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        case.update(render_dir='fixed_side_render_fit',camera_center_forward_m=1.85)
    with ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(render,cases))
    manifest['main_camera']['case_framing_note']='All clips retain4.6m orthographic width and fixed world cameras. Two far-reaching fall clips use1.85m centre instead of1.7m; rulers use each actual camera. No tracking or zoom changes.';atomic(OUT/'source_manifest.json',manifest);print('DRAFT_FALL_FRAMING_FITTED',flush=True)

if __name__=='__main__':main()
