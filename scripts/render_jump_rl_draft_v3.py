"""Front-oblique and true-side muscle contours for the four unchanged cases."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v3'

def main():
    m=read(OUT/'source_manifest.json');cases=[c for c in m['cases']if c['key']in m['final_four_keys']];done=[]
    def render(c):
        folder=Path(c['multiview_folder']);args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts/render_control_jump_blender.py'),'--','--kg','0','--folder',str(folder),'--analysis-study','--matched-person-scale','--hide-com','--native-muscle-contours','--fps',str(c['multiview_fps']),'--skip-existing']
        with(folder/'multiview_render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return c['key']
    atomic(OUT/'draft_status.json',dict(status='rendering',done=[],total=4))
    with ThreadPoolExecutor(max_workers=2)as pool:
        futures=[pool.submit(render,c)for c in cases]
        for future in as_completed(futures):done.append(future.result());atomic(OUT/'draft_status.json',dict(status='rendering',done=done,total=4));print('V3_MULTIVIEW',done[-1],flush=True)
    atomic(OUT/'draft_status.json',dict(status='rendered',done=done,total=4));print('V3_MULTIVIEW_READY',flush=True)

if __name__=='__main__':main()
