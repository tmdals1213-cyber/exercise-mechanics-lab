"""Render the checked motion at48fps for a smooth0.65x second replay."""
from pathlib import Path
import subprocess
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v1'

def main():
    manifest=read(OUT/'source_manifest.json');case=next(c for c in manifest['cases']if c['key']=='best_checked');folder=Path(case['folder']);args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts/render_fixed_side_jump_blender.py'),'--','--kg','0','--folder',str(folder),'--analysis-study','--fps','48','--world-side','--ortho-width','4.6','--camera-forward','1.7','--render-dir','fixed_side_render_slow','--skip-existing']
    with(folder/'slow_hero_render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    # Other framing jobs may have updated the same manifest meanwhile.
    manifest=read(OUT/'source_manifest.json');case=next(c for c in manifest['cases']if c['key']=='best_checked');case.update(render_fps=48,render_dir='fixed_side_render_slow');atomic(OUT/'source_manifest.json',manifest);print('DRAFT_SLOW_HERO_READY',flush=True)

if __name__=='__main__':main()
