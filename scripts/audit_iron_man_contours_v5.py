from pathlib import Path
import json,hashlib,numpy as np,imageio_ffmpeg
from PIL import Image
from group_contour_style_v5 import rgb,joint_rgb
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_contours_v5';OLD=ROOT/'deliverables/iron_man_comparison_final'
def scan(path,keep):
 r=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24');meta=next(r);assert meta['size']==(1080,1920);n=0;foot=0;frames={}
 for raw in r:
  im=np.frombuffer(raw,np.uint8).reshape(1920,1080,3);foot=max(foot,int(abs(im[1540:].astype(int)-[13,23,38]).max()))
  if n in keep:frames[n]=im.copy()
  n+=1
 return n,foot,frames

def main():
 m=json.loads((OUT/'metrics.json').read_text());old=json.loads((OLD/'metrics.json').read_text());assert m['duration_s']==22 and m['comparisons']==old['comparisons'];assert m['muscle_palette']['upper_N']==30000 and m['joint_palette']['upper_N']==50000
 assert np.array_equal(rgb(np.array([0,100,1000,30000])),[[43,97,142],[48,163,169],[240,183,78],[238,82,69]])
 checks={}
 for key in ['jump','squat']:
  folder=OUT/'render_cases'/key;src=OLD/'render_cases'/key
  for name in ['tensions.npz','tension_summary.json']:
   assert hashlib.sha256((folder/'analysis'/name).read_bytes()).hexdigest()==hashlib.sha256((src/'analysis'/name).read_bytes()).hexdigest()
  z=dict(np.load(folder/'analysis/tensions.npz'));v=json.loads((folder/'analysis/visualization/data.json').read_text());F=np.array(v['group_tension_N']);err=max(float(abs(F[i]-z['force_N'][:,g['indices']].sum(1)).max()) for i,g in enumerate(v['groups']));assert err<1e-7
  j=dict(np.load(folder/'analysis/visualization/joint_reactions.npz'));J=np.array(v['joint_force_N']);jerr=float(abs(np.linalg.norm(j['joint_force_world_N'],axis=2).T-J).max());assert jerr<1e-7
  mapping=json.loads((folder/'tension_render_mapping.json').read_text());assert mapping['palette_upper_N']==30000 and mapping['joint_upper_N']==50000 and not mapping['forces_changed']
  assert len(list((folder/'study_render').glob('frame_*.png')))==121
  for f in (folder/'study_render').glob('frame_*.png'):
   with Image.open(f) as im:assert im.size==(1600,860);im.verify()
  for item in m['joint_comparisons']:
   ids=[i for i,g in enumerate(v['joints']) if g['key'].startswith(item['joint'].lower()+'_')];series=J[ids].mean(0);t=np.array(v['times']);a,b=item[key]['window_s'];tt=np.r_[a,t[(t>a)&(t<b)],b];ff=np.interp(tt,t,series);assert abs(ff.max()-item[key]['peak_N'])<1e-7
  checks[key]=dict(group_force_max_error_N=err,joint_resultant_max_error_N=jerr,force_files_unchanged=True,main_render_frames=121)
 assert len((OUT/'caption_en.txt').read_text(encoding='utf-8').splitlines())==4
 a=np.asarray(Image.open(OUT/'model_case/hero_solid/frame_0059.png'))[:,:,3]>16;b=np.asarray(Image.open(OUT/'render_cases/squat/intro_contour/frame_0033.png'))[:,:,3]>16;iou=float((a&b).sum()/(a|b).sum());assert iou>.99,iou
 hn,hf,H=scan(OUT/'iron_man_contour_hook_2s.mp4',{0,20,36,59});n,foot,V=scan(OUT/m['file'],{60,120,420,540,659});assert hn==60 and n==660 and max(hf,foot)<=5,(hn,n,hf,foot)
 for k in [0,36,59]:Image.fromarray(H[k]).save(OUT/f'verified_hook_{k:02d}.png')
 for k in [120,420,540]:Image.fromarray(V[k]).save(OUT/f'verified_video_{k:03d}.png')
 audit=dict(status='passed',hook_frames=hn,full_frames=n,duration_s=22,footer_max_RGB_error=max(hf,foot),cases=checks,neutral_to_contour_alpha_mask_IoU=iou,main_camera_width_m=5.6,source_forces_and_existing_comparison_statistics_unchanged=True,muscle_scale='Per-side group sum, common 0-30 kN logarithmic',exposure_scale='Per-side group integral, common 0-30 kN s logarithmic',joint_scale='Resultant magnitude, common 0-50 kN linear black/red',joint_summary='Bilateral-mean per-side sampled peak resultant; same active windows. Not contact pressure or stress.',not_empirical_validation=True,physics_scope_reference='../iron_man_comparison_final/model_scope.json',video_sha256=hashlib.sha256((OUT/m['file']).read_bytes()).hexdigest(),source_code_sha256={f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in ['group_contour_style_v5.py','render_iron_man_contours_v5_blender.py','render_iron_man_intro_contour_v5_blender.py','compose_iron_man_contours_v5.py']})
 (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2));print('CONTOURS_V5_PASS',hn,n,max(hf,foot),iou,flush=True)
if __name__=='__main__':main()
