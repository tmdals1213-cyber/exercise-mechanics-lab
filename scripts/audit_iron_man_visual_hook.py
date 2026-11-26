from pathlib import Path
import json,hashlib,numpy as np,imageio_ffmpeg
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_visual_hook_v3';OLD=ROOT/'deliverables/iron_man_comparison_final'
def decode(p,want):
 r=imageio_ffmpeg.read_frames(str(p),pix_fmt='rgb24');meta=next(r);assert meta['size']==(1080,1920);n=0;footer=0;frames={}
 for raw in r:
  im=np.frombuffer(raw,np.uint8).reshape(1920,1080,3);footer=max(footer,int(abs(im[1540:].astype(int)-np.array([13,23,38])).max()))
  if n in want:frames[n]=im.copy()
  n+=1
 return meta,n,footer,frames

def main():
 m=json.loads((OUT/'visual_hook_metrics.json').read_text());source=ROOT/m['source_video'];assert hashlib.sha256(source.read_bytes()).hexdigest()==m['source_video_sha256'];hook=OUT/m['hook_video'];full=OUT/m['video'];hm,hn,hf,H=decode(hook,{0,12,24,36,48,59});fm,fn,ff,F=decode(full,{60,300,839});sm,sn,sf,S=decode(source,{60,300,839});assert hn==60 and fn==840 and sn==840;assert max(hf,ff)<=5
 errors={k:float(abs(F[k].astype(float)-S[k]).mean()) for k in S};assert max(errors.values())<2,errors
 assert len((OUT/'caption_en.txt').read_text(encoding='utf-8').splitlines())==4
 for name in ['tensions.npz','tension_summary.json']:
  assert hashlib.sha256((OUT/'hero_case/analysis'/name).read_bytes()).hexdigest()==hashlib.sha256((OLD/'render_cases/squat/analysis'/name).read_bytes()).hexdigest()
 assert np.mean(abs(H[24].astype(float)-H[48]))>3
 for k in [0,24,48]:Image.fromarray(H[k]).save(OUT/f'verified_hook_{k:02d}.png')
 audit=dict(status='passed',hook_frames=hn,hook_duration_s=2,full_frames=fn,full_duration_s=28,footer_max_RGB_error=max(hf,ff),main_analysis_reencoding_mean_RGB_errors=errors,source_motion_and_tensions_unchanged=True,caption_lines=4,hook_sha256=hashlib.sha256(hook.read_bytes()).hexdigest(),full_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),physics_validation_reference='../iron_man_comparison_final/completion_audit_hook_v2.json',source_code_sha256={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in ['render_iron_man_intro_blender.py','compose_iron_man_visual_hook.py']},visual_checks=['Actual athlete cutout visible at opening.','Moving generic squat body with procedural model cells, then computed muscle-group colours.','Reaction-force vectors use saved resultants/COP; display scaling is declared in metrics.','No new force inference from the athlete photograph.'])
 (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2));print('VISUAL_HOOK_PASS',hn,fn,max(hf,ff),errors,flush=True)
if __name__=='__main__':main()
