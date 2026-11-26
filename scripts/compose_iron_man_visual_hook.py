"""2 s visual-only reference -> mesh -> physical muscle-group load reveal."""
from pathlib import Path
import json,math,hashlib,html,sys
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_visual_hook_v3';OLD=ROOT/'deliverables/iron_man_comparison_final';BG=(13,23,38);FG=(237,243,251);MUTED=(139,163,183);CYAN=(83,194,223);ACC=(239,149,88);FPS=30
@lru_cache(maxsize=24)
def font(s,b=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if b else 'C:/Windows/Fonts/arial.ttf',s)
def smooth(u):u=np.clip(u,0,1);return u*u*(3-2*u)
def put(d,xy,s,size=25,c=FG,b=False):d.text(xy,s,font=font(size,b),fill=c)
def main():
 preview='--preview' in sys.argv;case=OUT/'hero_case';photo=Image.open(OUT/'assets/yun_cutout.png').convert('RGBA');assert photo.getextrema()[3][0]==0;photo=photo.crop(photo.getchannel('A').getbbox());source=dict(np.load(case/'motion.npz'));meta=json.loads((case/'summary.json').read_text());idx={n:i for i,n in enumerate(meta['body_names'])};dt=float(np.median(np.diff(source['times'])));centres=source['body_xpos'][:,idx['pelvis']].copy();centres[:,2]=(source['contact_positions_m'][:,:,2].min(1)+source['body_xpos'][:,idx['head'],2]+.13)/2;sigma=.065/dt;radius=int(np.ceil(3*sigma));ker=np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2);ker/=ker.sum();centres=np.column_stack([np.convolve(np.pad(centres[:,j],(radius,radius),mode='edge'),ker,mode='valid') for j in range(3)]);yaw=np.deg2rad(25);el=np.deg2rad(15);right=np.array([np.cos(yaw),np.sin(yaw),0]);up=np.array([-np.sin(yaw)*np.sin(el),np.cos(yaw)*np.sin(el),np.cos(el)]);images={}
 for mode,begin,end in [('hero_wire',10,43),('hero_solid',30,59)]:
  images[mode]={}
  for k in range(begin,end+1):
   p=case/mode/f'frame_{k:04d}.png'
   with Image.open(p) as im:assert im.size==(1080,1536);images[mode][k]=im.convert('RGBA')
 def project(k,points):
  ht=k/30;blend=smooth((ht-.35)/.70);physical=.75+1.35*k/59;i=int(abs(source['times']-physical).argmin());target=centres[i].copy();target[2]=(1-blend)*(source['body_xpos'][i,idx['head'],2]-.20)+blend*target[2];target-=right*.25*(1-blend);world=np.array(points)-np.r_[centres[i,:2],0];delta=world-target;pp=1536/(1.6+1.8*blend);return np.c_[540+pp*(delta@right),768-pp*(delta@up)]
 def arrow(draw,start,end,c,alpha):
  delta=end-start;L=np.linalg.norm(delta)
  if L<4:return
  n=delta/L;side=np.array([-n[1],n[0]]);draw.line([tuple(start),tuple(end)],fill=(*c,int(alpha)),width=4);base=end-13*n;draw.polygon([tuple(end),tuple(base+6*side),tuple(base-6*side)],fill=(*c,int(alpha)))
 def frame(k):
  t=k/FPS;im=Image.new('RGBA',(1080,1920),(*BG,255));d=ImageDraw.Draw(im)
  # Restrained construction grid, drawn in the real model camera.
  if t>.65:
   grid=Image.new('RGBA',im.size);gd=ImageDraw.Draw(grid);opacity=int(70*smooth((t-.65)/.45))
   for x in np.linspace(-.75,.75,7):
    p=project(k,[[x,-.45,0],[x,.45,0]]);gd.line([tuple(a) for a in p],fill=(57,89,111,opacity),width=1)
   for y in np.linspace(-.45,.45,5):
    p=project(k,[[-.75,y,0],[.75,y,0]]);gd.line([tuple(a) for a in p],fill=(57,89,111,opacity),width=1)
   im=Image.alpha_composite(im,grid)
  # Face/appearance is a source-frame cutout, never a generated new performance.
  pa=1-smooth((t-.28)/.55)
  if pa>0:
   h=960*(1+.035*t);w=round(photo.width*h/photo.height);pic=photo.resize((w,round(h)),Image.Resampling.LANCZOS);a=np.array(pic.getchannel('A'),float);fade=np.clip((np.arange(pic.height)[::-1])[:,None]/90,0,1);pic.putalpha(Image.fromarray((a*fade*pa).astype(np.uint8)));im.alpha_composite(pic,(round(540-w/2),270))
  wa=smooth((t-.30)/.25)*(1-smooth((t-1.10)/.36))
  if wa>0:
   body=images['hero_wire'][int(np.clip(k,10,43))].copy();body.putalpha(body.getchannel('A').point(lambda a:round(a*wa)));im.alpha_composite(body,(0,0))
  sa=smooth((t-1.00)/.42)
  if sa>0:
   body=images['hero_solid'][max(k,30)].copy();body.putalpha(body.getchannel('A').point(lambda a:round(a*sa)));im.alpha_composite(body,(0,0))
  # Actual foot reaction and bar-on-body force vectors, scaled only for display.
  if t>1.15:
   layer=Image.new('RGBA',im.size);ld=ImageDraw.Draw(layer);physical=.75+1.35*k/59;i=int(abs(source['times']-physical).argmin());P=source['contact_positions_m'][i];F=source['contact_forces_N'][i];alpha=190*smooth((t-1.15)/.30)
   for ids in [np.arange(4),np.arange(4,8)]:
    f=F[ids].sum(0);p=np.sum(P[ids]*F[ids,2,None],axis=0)/max(F[ids,2].sum(),1e-9);uv=project(k,[p,p+f/10000]);arrow(ld,*uv,CYAN,alpha)
   uv=project(k,[P[-1],P[-1]+F[-1]/10000]);arrow(ld,*uv,ACC,alpha);im=Image.alpha_composite(im,layer)
  # One scan sweep signals reconstruction, instead of text describing it.
  if .30<t<1.12:
   sy=360+900*(t-.30)/.82;scan=Image.new('RGBA',im.size);sd=ImageDraw.Draw(scan);a=int(120*np.sin(np.pi*(t-.30)/.82));sd.line((120,sy,960,sy),fill=(*CYAN,a),width=2);im=Image.alpha_composite(im,scan)
  d=ImageDraw.Draw(im);put(d,(64,42),'@biomechanics_ksm',22,MUTED);put(d,(64,112),'IRON MAN',64,FG,True);put(d,(67,196),'YUN SUNG-BIN',26,ACC,True)
  phase=0 if t<.55 else 1 if t<1.2 else 2
  for j,(x,title) in enumerate([(96,'REFERENCE'),(432,'MODEL'),(756,'TENSION')]):
   put(d,(x,1380),title,28,ACC if j==phase else MUTED,j==phase)
   if j<2:put(d,(x+220,1380),'→',28,MUTED)
  d.line((96,1443,984,1443),fill=(39,61,77),width=2);progress=96+888*min(t/2,1);d.line((96,1443,progress,1443),fill=ACC,width=3)
  assert np.all(np.array(im)[1536:,:,:3]==BG)
  return im.convert('RGB')
 for k in [0,12,24,36,48,59]:frame(k).save(OUT/f'hook_preview_{k:02d}.png')
 frame(0).save(OUT/'cover.png');sourcevideo=OLD/'iron_man_squat_vs_box_jump_hook_v2.mp4';dest=OUT/'iron_man_squat_vs_box_jump_visual_v3.mp4';data=dict(source_video=str(sourcevideo.relative_to(ROOT)),source_video_sha256=hashlib.sha256(sourcevideo.read_bytes()).hexdigest(),video=dest.name,hook_video='iron_man_visual_hook_2s.mp4',intro_s=2,frames=840,duration_s=28,cutout='assets/yun_cutout.png',actual_source_frame_s=149.10,reference='https://www.youtube.com/watch?v=ngyj9rSEjLc',reference_publisher='JTBC Entertainment',generic_model_motion_not_personalized=True,hero_physical_times_s=[.75,2.10],hero_colour='Calculated muscle-group force, common absolute logarithmic palette; the main study retains its individual-element absolute palette.',vectors='Saved foot reaction resultants at per-foot COP and saved bar-on-body force; world vectors displayed with scale 10000 N per metre.',main_analysis_reused_after_first_2s=True)
 (OUT/'visual_hook_metrics.json').write_text(json.dumps(data,indent=2))
 if preview:print('VISUAL_HOOK_PREVIEWS_READY',flush=True);return
 writer=imageio_ffmpeg.write_frames(str(dest),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None);hookwriter=imageio_ffmpeg.write_frames(str(OUT/'iron_man_visual_hook_2s.mp4'),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','2','-movflags','+faststart']);hookwriter.send(None);reader=imageio_ffmpeg.read_frames(str(sourcevideo),pix_fmt='rgb24');vm=next(reader);assert vm['size']==(1080,1920);count=0
 try:
  for k in range(60):
   arr=np.array(frame(k));writer.send(arr);hookwriter.send(arr);count+=1
  hookwriter.close()
  for k,raw in enumerate(reader):
   if k<60:continue
   writer.send(np.frombuffer(raw,np.uint8).reshape(1920,1080,3));count+=1
 finally:reader.close();writer.close()
 assert count==840,count
 caption=(OUT/'caption_en.txt').read_text(encoding='utf-8');page='<html><meta charset="utf-8"><style>body{background:#0d1726;color:#edf3fb;font:18px system-ui;max-width:900px;margin:30px auto}video{max-height:86vh;max-width:100%}pre{white-space:pre-wrap}</style><video controls playsinline src="'+dest.name+'" poster="cover.png"></video><pre>'+html.escape(caption)+'</pre><a href="production_notes.json">Production notes</a></html>';(OUT/'index.html').write_text(page,encoding='utf-8');print('VISUAL_HOOK_VIDEO_READY',dest,count,flush=True)
if __name__=='__main__':main()
