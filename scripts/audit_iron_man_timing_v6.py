from pathlib import Path
import json,hashlib,numpy as np,imageio_ffmpeg
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_timing_v6';OLD=ROOT/'deliverables/iron_man_contours_v5'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
 for key in ['jump','squat']:
  f=OUT/'render_cases'/key;old=OLD/'render_cases'/key
  camera=json.loads((f/'slow_render_camera.json').read_text());records={r['frame']:r for r in camera['frames']}
  prior=json.loads((old/'matched_camera.json').read_text())
  for k in range(61):
   record=prior['frames'][2*k].copy();record['frame']=3*k;records[3*k]=record
  assert sorted(records)==list(range(181));camera['frames']=[records[k] for k in range(181)];camera['reuse_provenance']='Every third frame exactly reuses every second frame from v5; identical physical time and fixed camera.'
  (f/'slow_render_camera.json').write_text(json.dumps(camera,indent=2))

def scan(path,keep):
 r=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24');meta=next(r);assert meta['size']==(1080,1920) and meta['fps']==30;n=0;foot=0
 for raw in r:
  im=np.frombuffer(raw,np.uint8).reshape(1920,1080,3);foot=max(foot,int(abs(im[1540:].astype(int)-[13,23,38]).max()))
  if n in keep:Image.fromarray(im).save(OUT/f'verified_{path.stem}_{n:03d}.png')
  n+=1
 return n,foot

def main():
 m=json.loads((OUT/'metrics.json').read_text());old=json.loads((OLD/'metrics.json').read_text())
 assert m['comparisons']==old['comparisons'] and m['joint_comparisons']==old['joint_comparisons']
 assert m['intro_s']==3.5 and m['repeats']==2 and m['pass_durations_s']==[4.,6.] and m['summary_duration_s']==5 and m['combined_muscle_joint_summary']
 checks={}
 for key in ['jump','squat']:
  f=OUT/'render_cases'/key;prior=OLD/'render_cases'/key
  for name in ['tensions.npz','tension_summary.json','visualization/data.json','visualization/joint_reactions.npz']:assert sha(f/'analysis'/name)==sha(prior/'analysis'/name)
  assert sha(f/'motion.npz')==sha(prior/'motion.npz')
  camera=json.loads((f/'slow_render_camera.json').read_text());assert camera['ortho_width_m']==5.6 and camera['zoom_is_pose_independent'] and len(camera['frames'])==181
  for k in range(181):
   path=f/'slow_render'/f'frame_{k:04d}.png'
   with Image.open(path) as im:assert im.size==(1600,860);im.verify()
   assert abs(camera['frames'][k]['time_s']-m['cases'][key]['T']*k/180)<1e-8
  for k in range(61):assert sha(f/'slow_render'/f'frame_{3*k:04d}.png')==sha(prior/'study_render'/f'frame_{2*k:04d}.png')
  checks[key]=dict(source_physics_unchanged=True,normal_frames=121,slow_frames=181,speeds=m['cases'][key]['speeds'])
 assert len((OUT/'caption_en.txt').read_text(encoding='utf-8').splitlines())==4
 hn,hf=scan(OUT/'iron_man_contour_hook_3_5s.mp4',{0,35,63,104});n,foot=scan(OUT/m['file'],{105,165,225,315,405,554})
 assert hn==105 and n==555 and max(hf,foot)<=5,(hn,n,hf,foot)
 audit=dict(status='passed',duration_s=18.5,full_frames=n,hook_frames=hn,footer_max_RGB_error=max(hf,foot),intro_stretch=1.75,pass_durations_s=[4,6],combined_summary_s=5,source_forces_and_statistics_unchanged=True,cases=checks,not_empirical_validation=True,video_sha256=sha(OUT/m['file']))
 (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2))
 manifest=dict(video=m['file'],hook='iron_man_contour_hook_3_5s.mp4',viewer='index.html',summary='summary.png',caption='caption_en.txt',audit='completion_audit.json',duration_s=18.5,frames=555,sha256=audit['video_sha256'])
 (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2));print('TIMING_V6_PASS',n,hn,max(hf,foot),flush=True)

if __name__=='__main__':
 import sys
 prepare() if '--prepare' in sys.argv else main()
