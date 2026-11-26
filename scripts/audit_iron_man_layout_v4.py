from pathlib import Path
import json,hashlib,numpy as np,imageio_ffmpeg
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_layout_v4';STUDY=ROOT/'deliverables/iron_man_comparison_final'
def read(path,keep):
 r=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24');meta=next(r);assert meta['size']==(1080,1920);n=0;footer=0;frames={}
 for raw in r:
  im=np.frombuffer(raw,np.uint8).reshape(1920,1080,3);footer=max(footer,int(abs(im[1540:].astype(int)-np.array([13,23,38])).max()))
  if n in keep:frames[n]=im.copy()
  n+=1
 return meta,n,footer,frames

def main():
 m=json.loads((OUT/'metrics.json').read_text());old=json.loads((STUDY/'metrics.json').read_text());assert m['duration_s']==18 and m['intro_s']==2 and m['repeats']==3 and m['summary_duration_s']==4 and not m['intro_process_labels'];assert m['comparisons']==old['comparisons'];assert m['common_crops']==old['common_crops'];assert m['common_camera_ortho_width_m']==old['common_camera_ortho_width_m']
 for key in ['jump','squat']:
  assert m['cases'][key]['force_sha256']==old['cases'][key]['force_sha256']==hashlib.sha256((STUDY/'render_cases'/key/'analysis/tensions.npz').read_bytes()).hexdigest();assert abs(m['cases'][key]['speed']/old['cases'][key]['speed']-1.5)<1e-10
 assert len((OUT/'caption_en.txt').read_text(encoding='utf-8').splitlines())==4
 hm,hn,hf,H=read(OUT/'iron_man_layout_hook_2s.mp4',{0,20,36,59});fm,fn,ff,F=read(OUT/m['file'],{60,120,180,300,420,539});assert hn==60 and fn==540 and max(hf,ff)<=5,(hn,fn,hf,ff)
 assert np.mean(abs(H[0][20:690,560:1060].astype(float)-H[36][20:690,560:1060]))>2,'Model generation should be visible'
 for k in [0,36,59]:Image.fromarray(H[k]).save(OUT/f'verified_hook_{k:02d}.png')
 for k in [60,180,420]:Image.fromarray(F[k]).save(OUT/f'verified_video_{k:03d}.png')
 audit=dict(status='passed',hook_frames=hn,full_frames=fn,duration_s=18,footer_max_RGB_error=max(hf,ff),comparison_statistics_and_force_files_unchanged=True,playback_rate_relative_to_previous=1.5,summary_duration_s=4,caption_lines=4,physics_audit_reference='../iron_man_comparison_final/completion_audit_hook_v2.json',intro_layout=['upper-left athlete cutout','upper-right standing model generation','lower-left squat','lower-right box jump'],intro_process_text=False,frame_sampling=m['frame_sampling'],intro_action_common_pixels_per_m=(1600/5.6)*(505/730),main_common_camera_width_m=5.6,hook_sha256=hashlib.sha256((OUT/'iron_man_layout_hook_2s.mp4').read_bytes()).hexdigest(),video_sha256=hashlib.sha256((OUT/m['file']).read_bytes()).hexdigest(),source_code_sha256={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in ['compose_iron_man_layout_v4.py','render_iron_man_model_v4_blender.py']})
 (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2));print('LAYOUT_V4_PASS',hn,fn,max(hf,ff),flush=True)
if __name__=='__main__':main()
