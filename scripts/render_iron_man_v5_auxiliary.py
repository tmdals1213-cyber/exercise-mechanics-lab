from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_contours_v5';BLENDER=ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'
for key,kg in [('jump',0),('squat',240)]:
 for mode in ['exposure','joint_peak']:
  cmd=[str(BLENDER),'--background','--threads','3','--python','scripts/render_iron_man_contours_v5_blender.py','--','--kg',str(kg),'--folder',str(OUT/'render_cases'/key),'--analysis-study','--tension','--case-exposure','--matched-person-scale','--fps','30']
  if mode=='joint_peak':cmd+=['--joint-peak']
  with (OUT/(key+'_'+mode+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  assert (OUT/'render_cases'/key/('exposure_render' if mode=='exposure' else 'joint_peak_render')/'frame_0000.png').exists()
  print('V5_AUX',key,mode,flush=True)
