"""Render real selected trials using the established anatomical mesh pipeline."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess,sys
from jump_monitor_data import ROOT,read,atomic

OUT=ROOT/'deliverables/jump_rl_study_draft_v1'

def main():
    manifest=read(OUT/'source_manifest.json');blender=ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe';jobs=[]
    for case in manifest['cases']:jobs.append((case,'side'))
    jobs.append((next(c for c in manifest['cases']if c['key']=='best_checked'),'hook_iso'))
    done=[]
    def render(job):
        case,kind=job;folder=Path(case['folder']);fps=case['render_fps'];script='render_fixed_side_jump_blender.py'if kind=='side'else'render_control_jump_blender.py';args=[str(blender),'--background','--threads','4','--python',str(ROOT/'scripts'/script),'--','--kg','0','--folder',str(folder),'--analysis-study','--skip-existing','--fps',str(fps)]
        if kind=='side':args+=['--world-side','--ortho-width','4.6','--camera-forward','1.7','--frame-start',str(max(0,round(case['render_start_s']*fps)))]
        else:args+=['--matched-person-scale','--floor-review','--frame-end','90']
        with(folder/f'{kind}_render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return dict(case=case['key'],kind=kind)
    atomic(OUT/'draft_status.json',dict(status='rendering',completed=0,total=len(jobs)))
    with ThreadPoolExecutor(max_workers=2)as pool:
        pending=[pool.submit(render,job)for job in jobs]
        for future in as_completed(pending):
            result=future.result();done.append(result);atomic(OUT/'draft_status.json',dict(status='rendering',completed=len(done),total=len(jobs),done=done));print('DRAFT_SCENE_RENDERED',result,flush=True)
    atomic(OUT/'draft_status.json',dict(status='rendered',completed=len(done),total=len(jobs)));print('JUMP_RL_DRAFT_RENDERED',flush=True)

if __name__=='__main__':main()
