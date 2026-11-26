"""Verify the editorial revision without re-running unchanged mechanics checks."""
from pathlib import Path
import json,hashlib,numpy as np,imageio_ffmpeg
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_comparison_final'
def main():
 old=json.loads((OUT/'completion_audit.json').read_text());m=json.loads((OUT/'metrics.json').read_text());assert m['intro_s']==2 and m['duration_s']==28
 for name,digest in old['source_code_sha256'].items():
  if name!='compose_iron_man_final.py':assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest,name
 exposure=json.loads((OUT/'tension_exposure.json').read_text())
 for key in ['jump','squat']:
  digest=hashlib.sha256((OUT/'render_cases'/key/'analysis/tensions.npz').read_bytes()).hexdigest();assert digest==m['cases'][key]['force_sha256']==exposure['cases'][key]['force_sha256']
 lines=(OUT/'caption_en.txt').read_text(encoding='utf-8').splitlines();assert len(lines)==4 and all(lines)
 reader=imageio_ffmpeg.read_frames(str(OUT/m['file']),pix_fmt='rgb24');meta=next(reader);assert meta['size']==(1080,1920);count=0;footer=0;keep={}
 for raw in reader:
  im=np.frombuffer(raw,np.uint8).reshape(1920,1080,3);footer=max(footer,int(abs(im[1540:].astype(int)-np.array([13,23,38])).max()))
  if count in [0,30,59,60,600,839]:keep[count]=im.copy()
  count+=1
 assert count==840 and footer<=5,(count,footer)
 assert np.mean(abs(keep[0][202:1400].astype(float)-keep[30][202:1400]))>1,'Intro must show movement'
 for i in [0,60,839]:Image.fromarray(keep[i]).save(OUT/f'verified_hook_frame_{i:03d}.png')
 audit=dict(old);audit['source_code_sha256']['compose_iron_man_final.py']=hashlib.sha256((ROOT/'scripts/compose_iron_man_final.py').read_bytes()).hexdigest();audit.update(status='passed',video_sha256=hashlib.sha256((OUT/m['file']).read_bytes()).hexdigest(),decoded_frames=count,video=meta,footer_max_RGB_error=footer,editorial_revision=dict(intro_s=2,hook='Yun Sung-bin to virtual athlete',animated_intro_verified=True,caption_lines=len(lines),body_scale_and_physics_unchanged=True,physics_checks_reused_after_source_hash_verification=True))
 (OUT/'completion_audit_hook_v2.json').write_text(json.dumps(audit,indent=2));print('HOOK_EDIT_PASS',count,footer,len(lines),flush=True)
if __name__=='__main__':main()
