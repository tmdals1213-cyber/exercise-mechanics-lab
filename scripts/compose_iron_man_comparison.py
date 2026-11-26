"""Simple athlete-inspired portrait edit; technical scope stays in caption."""
from pathlib import Path
import json,hashlib,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from build_motion_study_reel import anatomy_icon
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_comparison'
BG=(13,23,38);FG=(237,243,251);MUTED=(166,188,209);ACC=(239,149,88);COLORS=[ACC,(80,198,186),(139,158,245)]
FPS=30;CYCLE=5.2;MOTION=3*CYCLE;DURATION=4+MOTION+4
GROUPS=[('QUADS',('vasint_','vaslat_','vasmed_','recfem_'),1),('GLUTES',('glmax',),0),('CALVES',('soleus_','gasmed_','gaslat_'),3)]
def font(n,b=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if b else 'C:/Windows/Fonts/arial.ttf',n)
def text(d,xy,s,n=28,c=FG,b=False):
 box=d.textbbox((0,0),s,font=font(n,b));assert box[2]<=1080-xy[0]-24,(xy,s,box)
 d.text(xy,s,font=font(n,b),fill=c)
def center(d,y,s,n=40,c=FG,b=False):
 w=d.textbbox((0,0),s,font=font(n,b))[2];d.text(((1080-w)/2,y),s,font=font(n,b),fill=c)
def main():
 cases={};checks={}
 for key,speed,end in [('jump',.35,1.799),('squat',.70,3.5)]:
  folder=OUT/'render_cases'/key;meta=json.loads((folder/'summary.json').read_text());a=folder/'analysis';z=dict(np.load(a/'tensions.npz'));v=json.loads((a/'visualization/data.json').read_text());names=v['names'];files=sorted((folder/'review_render').glob('frame_*.png'));expected=round(meta['phase_times_s']['end']*30)+1;assert len(files)==expected,(key,len(files),expected)
  last=min(len(files)-1,round(end*30));bounds=[[],[]]
  for f in files[:last+1]:
   pixels=np.asarray(Image.open(f));alpha=pixels[:,:,3]
   for j,(lo,hi) in enumerate([(0,800),(800,1600)]):
    yy,xx=np.where(alpha[:,lo:hi]>8);assert len(xx)>0;bounds[j].append((int(xx.min()+lo),int(yy.min()),int(xx.max()+lo+1),int(yy.max()+1)))
  crops=[];sizes=[]
  for j,rows in enumerate(bounds):
   b=np.array(rows);lo=j*800;hi=lo+800;crop=(max(lo,int(b[:,0].min())-5),max(0,int(b[:,1].min())-5),min(hi,int(b[:,2].max())+5),min(860,int(b[:,3].max())+5));ratio=min([490,200][j]/(crop[2]-crop[0]),430/(crop[3]-crop[1]));crops.append(crop);sizes.append((round((crop[2]-crop[0])*ratio),round((crop[3]-crop[1])*ratio)))
  images=[]
  for f in files[:last+1]:
   im=Image.open(f).convert('RGBA');images.append([im.crop(c).resize(s,Image.Resampling.LANCZOS) for c,s in zip(crops,sizes)])
  traces=[];metrics=[]
  for label,stems,_ in GROUPS:
   ids=[i for i,n in enumerate(names) if n.startswith(stems) and n.endswith(('_r','_l'))];trace=z['force_N'][:,ids].sum(1)/2;traces.append(trace);metrics.append(dict(group=label,peak_N=float(trace.max()),integral_Ns=float(np.trapezoid(trace,z['times'])),window_s=[float(z['times'][0]),float(z['times'][-1])],definition='Mean per leg of simultaneous group sums'))
  cases[key]=dict(images=images,sizes=sizes,speed=speed,end=end,times=z['times'],traces=np.array(traces),metrics=metrics,meta=meta)
  checks[key]=dict(render_frame_count=len(files),motion_sha256=hashlib.sha256((folder/'motion.npz').read_bytes()).hexdigest(),tension_sha256=hashlib.sha256((a/'tensions.npz').read_bytes()).hexdigest(),speed=speed,clip_end_s=end,metrics=metrics)
 refs=np.maximum(cases['jump']['traces'].max(1),cases['squat']['traces'].max(1));peak=cases['jump']['metrics'][0]['peak_N']/cases['squat']['metrics'][0]['peak_N'];total=cases['squat']['metrics'][0]['integral_Ns']/cases['jump']['metrics'][0]['integral_Ns']
 def render_person(im,key,source,y):
  c=cases[key];index=min(len(c['images'])-1,round(source*30))
  for j,(p,size) in enumerate(zip(c['images'][index],c['sizes'])):im.paste(p,([285,615][j]-size[0]//2,y+(430-size[1])//2),p)
 def graphs(d,key,source,y):
  c=cases[key];ts=c['times'];phase=float(np.clip((source-ts[0])/(ts[-1]-ts[0]),0,1))
  for j,(label,_,kind) in enumerate(GROUPS):
   yy=y+j*139;anatomy_icon(d,743,yy-3,kind,scale=.40);text(d,(778,yy+7),label,27,c=COLORS[j],b=True);x0=755;w=270;top=yy+53;h=62
   d.line((x0,top+h,x0+w,top+h),fill=(54,71,91),width=2);pts=[(x0+(t-ts[0])/(ts[-1]-ts[0])*w,top+h-f/refs[j]*h) for t,f in zip(ts,c['traces'][j])];d.line(pts,fill=COLORS[j],width=4);x=x0+w*phase;value=np.interp(source,ts,c['traces'][j]);py=top+h-value/refs[j]*h;d.line((x,top-3,x,top+h+3),fill=FG,width=2);d.ellipse((x-5,py-5,x+5,py+5),fill=FG)
 def motion(t):
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);text(d,(55,65),'SQUAT vs BOX JUMP',52,b=True);text(d,(58,150),'MUSCLE TENSION',29,c=MUTED)
  for key,y,title in [('jump',295,'160 cm BOX JUMP'),('squat',910,'240 kg SQUAT')]:
   c=cases[key];source=min(t%CYCLE*c['speed'],c['end']);text(d,(55,y-60),title,36,b=True);text(d,(612,y-51),f"{c['speed']:g}x",25,c=MUTED);render_person(im,key,source,y);d=ImageDraw.Draw(im);graphs(d,key,source,y+5)
  d.line((55,815,1025,815),fill=(48,65,83),width=2)
  return im
 def hook(t):
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);center(d,75,'YUN SUNG-BIN',43,c=MUTED,b=True);center(d,155,'IRON MAN',102,b=True);center(d,325,'SQUAT vs BOX JUMP',43,c=ACC,b=True)
  for key,y,source in [('jump',460,1.55),('squat',970,1.6)]:
   c=cases[key];p=c['images'][min(len(c['images'])-1,round(source*30))][0];ratio=min(650/p.width,425/p.height);p=p.resize((round(p.width*ratio),round(p.height*ratio)),Image.Resampling.LANCZOS);im.paste(p,((1080-p.width)//2,y),p)
  d=ImageDraw.Draw(im);text(d,(70,860),'160 cm',43,c=ACC,b=True);text(d,(70,1365),'240 kg',43,c=ACC,b=True)
  return im
 def method(t):
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);center(d,160,'HOW MUCH TENSION?',55,b=True)
  # Simple external-force, joint-moment and muscle-path icons.
  for j,(x,label) in enumerate([(190,'MOTION'),(540,'TORQUE'),(890,'TENSION')]):
   if j==0:
    d.ellipse((x-22,530,x+22,574),outline=FG,width=6);d.line([(x,584),(x-12,710),(x+50,785),(x+15,865)],fill=FG,width=12);d.line([(x-12,710),(x-75,785),(x-100,865)],fill=FG,width=12);d.line([(x,610),(x+90,655)],fill=FG,width=10);d.line((x-120,895,x+110,895),fill=MUTED,width=4);d.line((x-95,1000,x-95,910),fill=ACC,width=6);d.polygon([(x-105,925),(x-85,925),(x-95,905)],fill=ACC)
   elif j==1:
    d.line([(x-85,590),(x,730),(x+100,850)],fill=FG,width=16);d.ellipse((x-20,710,x+20,750),fill=ACC);d.arc((x-90,640,x+90,820),190,340,fill=ACC,width=8);d.polygon([(x+80,775),(x+57,773),(x+79,751)],fill=ACC)
   else:
    d.line((x-50,555,x+50,875),fill=MUTED,width=9);d.ellipse((x-45,610,x+45,825),fill=ACC);d.line((x-50,555,x-50,500),fill=FG,width=5);d.polygon([(x-60,520),(x-40,520),(x-50,495)],fill=FG);d.line((x+50,875,x+50,930),fill=FG,width=5);d.polygon([(x+40,910),(x+60,910),(x+50,935)],fill=FG)
   w=d.textbbox((0,0),label,font=font(35,True))[2];d.text((x-w/2,1060),label,font=font(35,True),fill=FG)
  for x in [365,715]:d.line((x-38,735,x+38,735),fill=MUTED,width=5);d.polygon([(x+22,722),(x+42,735),(x+22,748)],fill=MUTED)
  return im
 def conclusion():
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);center(d,95,'QUADS',65,b=True);anatomy_icon(d,477,235,1,scale=2.0)
  for x,title,winner,ratio in [(55,'PEAK','JUMP',peak),(565,'TOTAL','SQUAT',total)]:
   d.rounded_rectangle((x,610,x+460,1230),radius=24,fill=(22,37,56));text(d,(x+45,665),title,39,c=MUTED,b=True);text(d,(x+45,780),winner,57,c=ACC if winner=='JUMP' else COLORS[1],b=True);text(d,(x+45,910),f'{ratio:.1f}x',100,b=True)
  return im
 count=round(DURATION*FPS);path=OUT/'iron_man_squat_vs_box_jump.mp4';writer=imageio_ffmpeg.write_frames(str(path),(1080,1920),fps=FPS,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None)
 static=[hook(0),method(0),conclusion()];static[0].save(OUT/'cover.png');static[1].save(OUT/'method.png');static[2].save(OUT/'conclusion.png')
 try:
  for k in range(count):
   t=k/FPS;im=static[0] if t<2 else static[1] if t<4 else motion(t-4) if t<4+MOTION else static[2]
   assert np.all(np.asarray(im)[1536:]==BG)
   if k in [round(5*FPS),round(7*FPS),round(12*FPS)]:im.save(OUT/f'preview_{k:04d}.png')
   writer.send(np.asarray(im))
 finally:writer.close()
 actual,seconds=imageio_ffmpeg.count_frames_and_secs(str(path));assert actual==count
 caption=f'''Iron Man power: a 160 cm box jump or a 240 kg squat—which asks more of your quads?

Inspired by Yun Sung-bin, we put both movements through the same 178 cm / 87 kg musculoskeletal model.

In this rough calculation:
Peak quad tension: JUMP ≈ {peak:.1f}x
Total quad tension over time: SQUAT ≈ {total:.1f}x

Movement + external forces → joint moments → estimated muscle tension. A higher peak and a longer load tell different stories.

TOTAL = tension integrated over time (N·s), not an average or mechanical work. Compared phases: jump propulsion (0.17 s) and squat ascent (2 s). Charts show mean group tension per leg; each chart pair shares its group scale. Contours use one common scale for individual muscle tensions during those phases. The 160 cm figure is box height.

These are illustrative model estimates inspired by the performance conditions. Anatomy, movement timing and force sharing affect the values. AI-agent-assisted analysis; no muscle-strength ceiling imposed.

Models: adapted MyoSim / MuJoCo; Seth–RMR shoulder paths; Catelli patella kinematics.
Method reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC4321112/
Squat reference: https://pubmed.ncbi.nlm.nih.gov/40275729/
Movement reference: https://www.youtube.com/watch?v=ngyj9rSEjLc&t=155s

Anatomy: BodyParts3D © DBCLS (CC BY 4.0); Z-Anatomy—Gauthier Kervyn, Marcin Zielinski, Lluis Vinent (CC BY-SA 4.0); supplementary BodyParts3D (CC BY-SA 2.1 JP). Geometry, posing and colors adapted.
https://huggingface.co/datasets/myohub/myosuite-assets
https://github.com/Z-Anatomy/Models-of-human-anatomy

#Biomechanics #YunSungBin #Squat #BoxJump
'''
 (OUT/'caption_en.txt').write_text(caption,encoding='utf-8');credits=ROOT/'results/power_comparison/CREDITS.md'
 if credits.exists():(OUT/'CREDITS.md').write_bytes(credits.read_bytes())
 manifest=dict(status='rough_model_concept_edit',video=path.name,width=1080,height=1920,fps=FPS,frames=count,duration_s=count/FPS,bottom_reserved_px=384,on_screen_language='English',hook_s=2,principle_s=2,motion_s=MOTION,conclusion_s=4,repetitions=3,quads_jump_to_squat_peak_ratio=peak,quads_squat_to_jump_integral_ratio=total,source_cases=checks,athlete_theme='Yun Sung-bin / Iron Man',individual_anthropometry_identified=False,force_accuracy_claim=False,video_notes='Existing checked motions reused. Jump clip cuts after initial box contact; late standing recovery is outside the edit. No newly solved landing is claimed.',notes_and_references='caption_en.txt')
 (OUT/'video_manifest.json').write_text(json.dumps(manifest,indent=2));(OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Iron Man comparison</title><style>body{background:#0d1726;color:#edf3fb;max-width:960px;margin:24px auto;padding:24px;font:17px/1.6 system-ui}video{display:block;height:80vh;max-width:100%}pre{white-space:pre-wrap}a{color:#50c6ba}</style><h1>Squat vs box jump</h1><video controls playsinline src="'+path.name+'" poster="cover.png"></video><details><summary>Post caption</summary><pre>'+html.escape(caption)+'</pre></details><p><a href="caption_en.txt">English caption</a> · <a href="video_manifest.json">Build record</a></p></html>',encoding='utf-8');print('IRON_MAN_VIDEO_READY',json.dumps(manifest),flush=True)
if __name__=='__main__':main()
