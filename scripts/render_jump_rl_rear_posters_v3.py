from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v3'
def main():
    m=read(OUT/'source_manifest.json');metrics=read(OUT/'metrics.json');cases=[c for c in m['cases']if c['key']in m['final_four_keys']]
    def one(c):
        folder=Path(c['multiview_folder']);n=round(c['duration_s']*c['multiview_fps']);frame=round(metrics['cases'][c['key']]['phase_times'][2]/c['duration_s']*n);c['rear_poster_frame']=frame;args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts/render_control_jump_blender.py'),'--','--kg','0','--folder',str(folder),'--analysis-study','--matched-person-scale','--hide-com','--native-muscle-contours','--paired-rear','--render-dir','rear_posters','--fps',str(c['multiview_fps']),'--frame-start',str(frame),'--frame-end',str(frame)]
        with(folder/'rear_poster.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    with ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(one,cases))
    m['final_views']=['front_oblique_motion','true_side_motion','rear_summary_pose'];atomic(OUT/'source_manifest.json',m);print('V3_REAR_POSTERS_READY',flush=True)
if __name__=='__main__':main()
