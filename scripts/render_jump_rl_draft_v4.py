from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v4'
def main():
    m=read(OUT/'source_manifest.json');cases=[c for c in m['cases']if c['key']in m['final_four_keys']];jobs=[(c,kind)for c in cases for kind in['front_rear','side']];done=[]
    def one(job):
        c,kind=job;folder=Path(c['multiview_folder']);script='render_control_jump_blender.py'if kind=='front_rear'else'render_fixed_side_jump_blender.py';args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts'/script),'--','--kg','0','--folder',str(folder),'--analysis-study','--hide-com','--fullbody-muscle-contours','--fps','24','--skip-existing']
        if kind=='front_rear':args+=['--matched-person-scale','--paired-rear','--render-dir',c['multiview_dir']]
        else:args+=['--world-side','--ortho-width','4.6','--camera-forward',str(c['camera_center_forward_m']),'--render-dir',c['full_side_dir']]
        with(folder/f'{kind}_render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return dict(case=c['key'],view=kind)
    atomic(OUT/'draft_status.json',dict(status='rendering',done=[],total=8))
    with ThreadPoolExecutor(max_workers=2)as pool:
        futures=[pool.submit(one,j)for j in jobs]
        for f in as_completed(futures):done.append(f.result());atomic(OUT/'draft_status.json',dict(status='rendering',done=done,total=8));print('V4_RENDER',done[-1],flush=True)
    atomic(OUT/'draft_status.json',dict(status='rendered',done=done,total=8));print('V4_RENDER_READY',flush=True)
if __name__=='__main__':main()
