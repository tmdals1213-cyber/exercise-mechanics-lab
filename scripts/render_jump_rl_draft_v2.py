"""Render actual anatomy without any COM marker; frozen training snapshot."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess
from pathlib import Path
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v2'

def main():
    m=read(OUT/'source_manifest.json');jobs=[(c,'side')for c in m['cases']]+[(next(c for c in m['cases']if c['key']=='best_checked'),'iso')];done=[]
    def render(job):
        c,kind=job;folder=Path(c['folder']);script='render_fixed_side_jump_blender.py'if kind=='side'else'render_control_jump_blender.py';fps=c['render_fps']if kind=='side'else 30
        args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts'/script),'--','--kg','0','--folder',str(folder),'--analysis-study','--hide-com','--skip-existing','--fps',str(fps)]
        if kind=='side':args+=['--world-side','--ortho-width','4.6','--camera-forward',str(c['camera_center_forward_m']),'--frame-start',str(round(c['render_start_s']*fps))]
        else:args+=['--matched-person-scale','--floor-review','--frame-end','90']
        with(folder/f'{kind}_render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return dict(case=c['key'],kind=kind)
    atomic(OUT/'draft_status.json',dict(status='rendering',done=[],total=len(jobs)))
    with ThreadPoolExecutor(max_workers=2)as pool:
        pending=[pool.submit(render,j)for j in jobs]
        for future in as_completed(pending):
            result=future.result();done.append(result);atomic(OUT/'draft_status.json',dict(status='rendering',done=done,total=len(jobs)));print('V2_SCENE',result,flush=True)
    atomic(OUT/'draft_status.json',dict(status='rendered',done=done,total=len(jobs)));print('V2_ANATOMY_READY',flush=True)

if __name__=='__main__':main()
