"""Final portrait: 3.5 s hook, normal and slower replay, combined summary.
Camera crops are identical between cases; force clocks stay in physical time.
"""
from pathlib import Path
import json,math,hashlib,csv,html,sys
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from build_motion_study_reel import anatomy_icon,font,BG,FG,MUTED,ACC
from functools import lru_cache
font=lru_cache(maxsize=64)(font)
from group_contour_style_v5 import rgb as group_rgb,joint_rgb,MUSCLE_UPPER_N,EXPOSURE_UPPER_NS,JOINT_UPPER_N
from case_contours import overview
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT/'deliverables/iron_man_contours_v5';OUT=ROOT/'deliverables/iron_man_timing_v6';OUT.mkdir(exist_ok=True)
FPS=30;CYCLE=4.;PASS_DURATIONS=[4.,6.];REPEATS=2;INTRO=3.5;MOTION=10.;DURATION=18.5
GROUPS=[('QUADS',1),('GLUTES',0),('HAMSTRINGS',2),('CALVES',3)]
CROPS=[(40,100,800,760),(995,100,1345,760)]
SIZES=[(475,413),(223,420)]

def text(d,xy,s,size=24,c=FG,b=False):
 assert d.textbbox((0,0),s,font=font(size,b))[2]<=1080-xy[0]-20,(xy,s)
 d.text(xy,s,font=font(size,b),fill=c)

def main():
 preview='--preview' in sys.argv;cases={}
 for key,title in [('jump','160 CM BOX JUMP'),('squat','240 KG SQUAT')]:
  folder=STUDY/'render_cases'/key;ap=folder/'analysis';z=dict(np.load(ap/'tensions.npz'));s=json.loads((ap/'tension_summary.json').read_text());v=json.loads((ap/'visualization/data.json').read_text());meta=json.loads((folder/'summary.json').read_text());ids=[s['names'].index('recfem_'+side) for side in ['r','l']];rf=z['force_N'][:,ids].mean(1);ts=z['times'];T=float(ts[-1]);files=sorted((folder/'study_render').glob('frame_*.png'));expected=121
  assert len(files)==expected,(key,len(files),expected)
  images=[[],[]]
  for file in files:
   with Image.open(file) as im:
    for view in range(2):images[view].append(im.crop(CROPS[view]).resize(SIZES[view],Image.Resampling.LANCZOS).convert('RGBA'))
  slow_images=[[],[]]
  slow_files=sorted((OUT/'render_cases'/key/'slow_render').glob('frame_*.png'));assert len(slow_files)==181,(key,len(slow_files))
  for file in slow_files:
   with Image.open(file) as im:
    for view in range(2):slow_images[view].append(im.crop(CROPS[view]).resize(SIZES[view],Image.Resampling.LANCZOS).convert('RGBA'))
  camera=json.loads((folder/'matched_camera.json').read_text());assert camera['ortho_width_m']==5.6 and camera['zoom_is_pose_independent']
  cases[key]=dict(title=title,times=ts,rf=rf,T=T,groups=overview(v)[[1,0,2,3]],group_times=np.array(v['times']),joint_times=np.array(v['times']),joint_names=v['joints'],joint_values=np.array(v['joint_force_N']),images=images,slow_images=slow_images,meta=meta,folder=folder,force_sha256=hashlib.sha256((ap/'tensions.npz').read_bytes()).hexdigest(),motion_sha256=hashlib.sha256((folder/'motion.npz').read_bytes()).hexdigest())
 ymax=max(1000,math.ceil(max(c['rf'].max() for c in cases.values())/1000)*1000);group_ref=np.maximum(*[c['groups'].max(1) for c in cases.values()]);exposure=json.loads((STUDY/'tension_exposure.json').read_text());ec=exposure['cases'];comparisons=[]
 for j,(name,_) in enumerate(GROUPS):
  stats={}
  for key,c in cases.items():
   a,b=ec[key]['start_s'],ec[key]['end_s'];mask=(c['group_times']>=a)&(c['group_times']<=b);stats[key]=dict(peak_N=float(c['groups'][j,mask].max()),integral_Ns=ec[key]['group_integral_mean_per_side_Ns'][j])
  comparisons.append(dict(group=name,**stats))
 def log_legend(im,d,x,y,w,upper,unit,tick_y=None):
  values=np.expm1(np.linspace(0,np.log1p(upper),w));ramp=group_rgb(values,upper)[None];im.paste(Image.fromarray(ramp).resize((w,10)),(x,y));ty=y+18 if tick_y is None else tick_y
  for value,label in [(0,'0'),(100,'0.1'),(1000,'1'),(upper,f'{upper/1000:.0f}')]:
   xx=x+w*np.log1p(value)/np.log1p(upper);d.line((xx,y+10,xx,y+14),fill=MUTED,width=1);text(d,(max(x,xx-8),ty),label,16,MUTED)
  text(d,(x+w+34,y-7),unit,19,MUTED)
 def graph(d,c,t,base,key):
  x0=765;y0=base+100;w=260;h=235;anatomy_icon(d,698,base+18,4,.52);text(d,(735,base+20),'RECTUS FEMORIS',25,FG,True)
  phase=c['meta']['phase_times_s'];a=phase['start_ascent'];b=phase['end_ascent'];d.rectangle((x0+w*a/c['T'],y0,x0+w*b/c['T'],y0+h),fill=(29,45,58))
  for u in [0,.5,1]:
   yy=y0+h*(1-u);d.line((x0,yy,x0+w,yy),fill=(47,65,82));text(d,(716,yy-10),f'{ymax*u/1000:.0f}',18,MUTED)
  text(d,(730,y0-27),'kN',18,MUTED);points=[(x0+w*tt/c['T'],y0+h*(1-f/ymax)) for tt,f in zip(c['times'],c['rf'])];d.line(points,fill=(105,126,143),width=3);past=[p for p,tt in zip(points,c['times']) if tt<=t];cur=float(np.interp(t,c['times'],c['rf']));cx=x0+w*t/c['T'];cy=y0+h*(1-cur/ymax)
  if len(past)>1:d.line([*past,(cx,cy)],fill=ACC,width=4)
  d.line((cx,y0,cx,y0+h),fill=(223,234,244),width=2);d.ellipse((cx-5,cy-5,cx+5,cy+5),fill=FG);text(d,(x0,y0+h+12),'0',18,MUTED);text(d,(x0+w-48,y0+h+12),f'{c["T"]:.1f}s',18,MUTED)
  if key=='jump':label='PREPARATION' if t<a else 'PUSH-OFF' if t<b else 'FLIGHT' if t<phase['landing'] else 'LANDING' if t<phase['landing']+.35 else 'RECOVERY'
  else:label='READY' if t<.25 else 'LOWERING' if t<a else 'ASCENT' if t<b else 'LOCKOUT'
  text(d,(745,base+365),label,20,MUTED);text(d,(745,base+405),f'{cur/1000:.2f} kN',33,ACC,True);text(d,(745,base+451),f'Peak {c["rf"].max()/1000:.2f} kN',22,MUTED)
 def frame(t):
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);text(d,(45,28),'@biomechanics_ksm',23,MUTED);text(d,(45,72),'160 CM BOX JUMP vs 240 KG SQUAT',38,FG,True);cycle=PASS_DURATIONS[0] if t<PASS_DURATIONS[0] else PASS_DURATIONS[1];local=t if t<PASS_DURATIONS[0] else t-PASS_DURATIONS[0];p=min(local/cycle,1);slow=t>=PASS_DURATIONS[0]
  for key,base in [('jump',202),('squat',820)]:
   c=cases[key];physical=p*c['T'];text(d,(45,base),'BOX JUMP' if key=='jump' else 'SQUAT',30,FG,True);text(d,(735,base-15),f'{c["T"]/cycle:.2f}x speed',22,MUTED);images=c['slow_images'] if slow else c['images'];fi=p*(180 if slow else 120)
   for view in range(2):
    body=images[view][min(int(round(fi)),len(images[view])-1)];x=(20+(475-SIZES[view][0])//2) if view==0 else (490+(225-SIZES[view][0])//2);y=base+80+(420-SIZES[view][1]);im.paste(body,(x,y),body)
   graph(d,c,physical,base,key)
   for j,(name,icon) in enumerate(GROUPS):
    x=45+j*250;y=base+513;force=float(np.interp(physical,c['group_times'],c['groups'][j]));u=force/max(group_ref[j],1);anatomy_icon(d,x,y-5,icon,.72);text(d,(x+50,y),name,19,FG,True);d.rounded_rectangle((x+50,y+56,x+191,y+65),radius=4,fill=(41,60,77))
    if u>0:d.rounded_rectangle((x+50,y+56,x+50+max(1,141*u),y+65),radius=4,fill=ACC)
    text(d,(x+50,y+27),f'{force/1000:.2f} kN',21,MUTED)
  d.line((45,800,1035,800),fill=(47,65,82));text(d,(45,1425),'MUSCLE GROUP TENSION',19,MUTED);log_legend(im,d,45,1460,280,MUSCLE_UPPER_N,'kN',tick_y=1474);text(d,(395,1452),'LOG SCALE',18,MUTED)
  text(d,(45,1500),'JOINT REACTION',18,MUTED);ramp=joint_rgb(np.linspace(0,JOINT_UPPER_N,135))[None];im.paste(Image.fromarray(ramp).resize((135,8)),(235,1509));text(d,(390,1500),'0-50 kN',18,MUTED)
  return im
 def conclusion():
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);text(d,(45,48),'TOTAL',52,FG,True);text(d,(45,115),'TENSION x TIME',25,MUTED);text(d,(45,156),'',22,MUTED)
  for key,offset,title in [('jump',20,'160 CM BOX JUMP'),('squat',550,'240 KG SQUAT')]:
   text(d,(offset+25,205),title,29,FG,True);source=Image.open(cases[key]['folder']/'exposure_render/frame_0000.png').convert('RGBA')
   for view,cx in enumerate([offset+125,offset+355]):
    crop=(250,145,610,690) if view==0 else (990,145,1350,690);body=source.crop(crop);body=body.resize((round(360*540/545),540),Image.Resampling.LANCZOS);im.paste(body,(round(cx-body.width/2),285),body)
  text(d,(45,855),'GROUP TOTAL / kN s',22,MUTED);log_legend(im,d,45,903,350,EXPOSURE_UPPER_NS,'kN s');text(d,(580,889),'LOG / ONE REPETITION',20,MUTED);text(d,(425,968),'PEAK',28,FG,True);text(d,(760,968),'TOTAL',28,FG,True)
  for j,c in enumerate(comparisons):
   y=1025+j*78;anatomy_icon(d,44,y-5,GROUPS[j][1],.48);text(d,(92,y),c['group'],23,FG,True)
   for metric,x in [('peak_N',425),('integral_Ns',760)]:
    jf=c['jump'][metric];sf=c['squat'][metric];winner='JUMP' if jf>sf else 'SQUAT';ratio=max(jf,sf)/max(min(jf,sf),1e-8);text(d,(x,y),f'{winner} {ratio:.2f}x',27,ACC if winner=='JUMP' else FG,True)
  text(d,(45,1390),'PEAK = INSTANT    /    TOTAL = TENSION x TIME',24,MUTED)
  text(d,(45,1440),'178 CM / 87 KG',21,MUTED)
  return im
 joint_comparisons=[]
 for stem in ['hip','knee','ankle']:
  item=dict(joint=stem.upper())
  for key,c in cases.items():
   ids=[i for i,j in enumerate(c['joint_names']) if j['key'].startswith(stem+'_')];assert len(ids)==2;force=c['joint_values'][ids].mean(0);ts=c['joint_times'];a,b=ec[key]['start_s'],ec[key]['end_s'];tt=np.r_[a,ts[(ts>a)&(ts<b)],b];ff=np.interp(tt,ts,force);i=int(ff.argmax());integral=float(np.trapezoid(ff,tt));item[key]=dict(peak_N=float(ff[i]),peak_time_s=float(tt[i]),mean_N=integral/(b-a),integral_Ns=integral,window_s=[a,b])
  joint_comparisons.append(item)
 (OUT/'joint_comparison.json').write_text(json.dumps(dict(metric='Joint reaction resultant magnitude, bilateral mean per side; sampled model estimates, not contact pressure.',joints=joint_comparisons),indent=2))
 def joint_conclusion():
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);text(d,(45,48),'JOINT REACTION',48,FG,True);text(d,(45,115),'PEAK / kN',25,MUTED)
  for key,cx,title in [('jump',270,'BOX JUMP'),('squat',810,'SQUAT')]:
   text(d,(cx-125,188),title,29,FG,True);raw=Image.open(cases[key]['folder']/'joint_peak_render/frame_0000.png').convert('RGBA');body=raw.crop((240,145,620,690)).resize((418,600),Image.Resampling.LANCZOS);im.paste(body,(cx-209,230),body)
  ramp=joint_rgb(np.linspace(0,JOINT_UPPER_N,300))[None];im.paste(Image.fromarray(ramp).resize((300,12)),(45,864));text(d,(45,890),'0',18,MUTED);text(d,(300,890),'50 kN',18,MUTED);text(d,(435,855),'SAME SCALE',21,MUTED)
  text(d,(400,950),'BOX JUMP',25,FG,True);text(d,(755,950),'SQUAT',25,FG,True)
  for j,c in enumerate(joint_comparisons):
   y=1032+120*j;text(d,(70,y+4),c['joint'],31,FG,True)
   for key,x in [('jump',400),('squat',755)]:
    val=c[key]['peak_N'];text(d,(x,y),f'{val/1000:.1f}',43,FG,True);d.rounded_rectangle((x,y+58,x+190,y+65),radius=3,fill=(35,49,63));w=190*min(val/JOINT_UPPER_N,1);d.rounded_rectangle((x,y+58,x+max(w,1),y+65),radius=3,fill=tuple(joint_rgb(val).tolist()))
  text(d,(45,1400),'PER SIDE',22,MUTED);text(d,(45,1440),'178 CM / 87 KG',21,MUTED);return im
 # Four-tile opening: appearance upper-left, generated model upper-right,
 # two actual movement replays below. No workflow labels or progress bar.
 photo=Image.open(OUT/'yun_cutout.png').convert('RGBA');photo=photo.crop(photo.getchannel('A').getbbox())
 model_wire=Image.open(STUDY/'model_case/hero_wire/frame_0059.png').convert('RGBA');model_solid=Image.open(STUDY/'model_case/hero_solid/frame_0059.png').convert('RGBA')
 boxes=np.array([model_wire.getchannel('A').getbbox(),model_solid.getchannel('A').getbbox()]);crop=(int(boxes[:,0].min())-8,int(boxes[:,1].min())-8,int(boxes[:,2].max())+8,int(boxes[:,3].max())+8)
 model_wire=model_wire.crop(crop);model_solid=model_solid.crop(crop);ms=min(350/model_solid.width,610/model_solid.height);model_size=(round(model_solid.width*ms),round(model_solid.height*ms));model_wire=model_wire.resize(model_size,Image.Resampling.LANCZOS);model_solid=model_solid.resize(model_size,Image.Resampling.LANCZOS)
 contour_images={}
 for i in range(27,60):
  with Image.open(STUDY/'render_cases/squat/intro_contour'/f'frame_{i:04d}.png') as ci:contour_images[i]=ci.convert('RGBA').crop(crop).resize(model_size,Image.Resampling.LANCZOS)
 def ease(u):u=np.clip(u,0,1);return u*u*(3-2*u)
 @lru_cache(maxsize=140)
 def action_image(key,fi):
  path=OUT/'render_cases'/key/'slow_render'/f'frame_{fi:04d}.png'
  with Image.open(path) as image:
   crop=(65,125,795,755) if key=='squat' else (870,125,1600,755)
   return image.convert('RGBA').crop(crop).resize((505,436),Image.Resampling.LANCZOS)
 def intro_frame_base(t):
  im=Image.new('RGBA',(1080,1920),(*BG,255))
  # Preserve the familiar face; let the virtual body assemble next to it.
  panel=Image.new('RGBA',(500,650));h=610*(1+.015*t);w=round(photo.width*h/photo.height);pic=photo.resize((w,round(h)),Image.Resampling.LANCZOS);alpha=np.array(pic.getchannel('A'),float);fade=np.clip(np.arange(pic.height)[::-1,None]/60,0,1);pic.putalpha(Image.fromarray((alpha*fade).astype(np.uint8)));panel.alpha_composite(pic,(-150,8));im.alpha_composite(panel,(20,45))
  blend=ease((t-.35)/.45);body=Image.blend(model_wire,model_solid,float(blend));
  if t>=.85:
   ci=float(np.clip(t*30,27,59));lo=int(ci);hi=min(lo+1,59);colour=Image.blend(contour_images[lo],contour_images[hi],ci-lo);body=Image.blend(body,colour,float(ease((t-.85)/.50)))
  a=np.array(body.getchannel('A'),float);reveal=ease((t-.05)/.55);edge=body.height*reveal;mask=np.clip((edge-np.arange(body.height)[:,None])/20,0,1);body.putalpha(Image.fromarray((a*mask).astype(np.uint8)));x=round(800-body.width/2);y=62;im.alpha_composite(body,(x,y))
  if .05<t<.60:
   d=ImageDraw.Draw(im);sy=y+edge;d.line((x-8,sy,x+body.width+8,sy),fill=(83,194,223,130),width=2)
  # Equal source-pixel scaling in both lower action views.
  u=float(np.clip((t-.08)/1.70,0,1))
  for key,x,title,start,end in [('squat',20,'240 KG SQUAT',.25,3.25),('jump',555,'160 CM BOX JUMP',.20,2.628)]:
   physical=start+u*(end-start);fi=physical/cases[key]['T']*180;image=action_image(key,min(int(round(fi)),180));im.alpha_composite(image,(x,825));d=ImageDraw.Draw(im);text(d,(x+18,755),title,28,FG,True)
  assert np.all(np.asarray(im)[1536:,:,:3]==BG)
  return im.convert('RGB')
 def intro_frame(t):return intro_frame_base(t/1.75)
 def combined_conclusion():
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
  text(d,(45,48),'MUSCLES + JOINTS',43,FG,True)
  for key,offset,title in [('jump',0,'160 CM BOX JUMP'),('squat',540,'240 KG SQUAT')]:
   text(d,(offset+45,150),title,29,FG,True)
   for kind,cx,filename,crop in [('TOTAL',offset+150,'exposure_render',(250,145,610,690)),('PEAK',offset+390,'joint_peak_render',(240,145,620,690))]:
    with Image.open(cases[key]['folder']/filename/'frame_0000.png') as raw:body=raw.convert('RGBA').crop(crop)
    body=body.resize((round(body.width*480/545),480),Image.Resampling.LANCZOS);im.paste(body,(round(cx-body.width/2),210),body);text(d,(cx-34,695),kind,20,MUTED)
  text(d,(45,750),'MUSCLE TOTAL',21,FG,True);log_legend(im,d,45,789,280,EXPOSURE_UPPER_NS,'kN s')
  text(d,(590,750),'JOINT PEAK',21,FG,True);ramp=joint_rgb(np.linspace(0,JOINT_UPPER_N,280))[None];im.paste(Image.fromarray(ramp).resize((280,10)),(590,789));text(d,(590,807),'0',16,MUTED);text(d,(846,807),'50',16,MUTED);text(d,(904,782),'kN',19,MUTED)
  d.line((550,880,550,1320),fill=(47,65,82),width=1)
  text(d,(45,877),'MUSCLES',24,FG,True);text(d,(233,922),'PEAK',21,MUTED,True);text(d,(389,922),'TOTAL',21,MUTED,True)
  for j,c in enumerate(comparisons):
   y=985+j*86;anatomy_icon(d,36,y-4,GROUPS[j][1],.35);text(d,(75,y),c['group'],18,FG,True)
   for metric,x in [('peak_N',226),('integral_Ns',389)]:
    jf=c['jump'][metric];sf=c['squat'][metric];winner='JUMP' if jf>sf else 'SQUAT';ratio=max(jf,sf)/max(min(jf,sf),1e-8);text(d,(x,y),f'{winner} {ratio:.1f}x',20,ACC if winner=='JUMP' else FG,True)
  text(d,(590,877),'JOINT PEAK / kN',24,FG,True);text(d,(770,922),'JUMP',21,MUTED,True);text(d,(923,922),'SQUAT',21,MUTED,True)
  for j,c in enumerate(joint_comparisons):
   y=984+j*123;text(d,(590,y+3),c['joint'],25,FG,True)
   for key,x in [('jump',770),('squat',923)]:text(d,(x,y),f'{c[key]["peak_N"]/1000:.1f}',32,FG,True)
  text(d,(45,1380),'PEAK = INSTANT / TOTAL = TENSION x TIME',23,MUTED);text(d,(45,1432),'178 CM / 87 KG',21,MUTED)
  assert np.all(np.asarray(im)[1536:]==BG)
  return im
 info=combined_conclusion();info.save(OUT/'summary.png');frame(2.45).save(OUT/'motion_preview.png');frame(7.675).save(OUT/'slow_motion_preview.png');intro_frame(2.10).save(OUT/'cover.png')
 metrics=dict(file='iron_man_squat_vs_box_jump_final_v6.mp4',duration_s=DURATION,width=1080,height=1920,fps=FPS,intro_s=INTRO,hook='Four-tile intro; model creation then calculated squat group-contour transition. No process explanation text.',joint_summary_duration_s=5,combined_muscle_joint_summary=True,pass_durations_s=PASS_DURATIONS,intro_stretch=1.75,joint_comparisons=joint_comparisons,muscle_palette=dict(aggregation='per-side muscle-group scalar tension',scale='logarithmic',upper_N=MUSCLE_UPPER_N),joint_palette=dict(metric='resultant magnitude',scale='linear black/red',upper_N=JOINT_UPPER_N),exposure_palette=dict(metric='muscle-group force-time integral',scale='logarithmic',upper_Ns=EXPOSURE_UPPER_NS),intro_contour_source='Saved squat group force at the same phase as the lower-left action, on the neutral display pose; not a new standing-load solve.',editorial_rate_relative_to_previous=1.0,summary_duration_s=5,display_retiming_only=True,intro_process_labels=False,frame_sampling='Exact 121-frame normal and 181-frame slow display-cycle sampling; no moving-pose cross-fades.',repeats=REPEATS,bottom_reserved_px=384,common_camera_ortho_width_m=5.6,common_crops=CROPS,common_output_pixels_per_m=[1600/5.6*SIZES[v][0]/(CROPS[v][2]-CROPS[v][0]) for v in range(2)],camera_zoom_pose_independent=True,source_force_times_preserved=True,all_phases_have_solved_tensions=True,force_semantics='Scalar muscle-tendon tension; group bilateral sum / 2; not activation.',comparisons=comparisons,tension_exposure=exposure,cases={k:dict(T=c['T'],speeds=[c['T']/v for v in PASS_DURATIONS],force_sha256=c['force_sha256'],motion_sha256=c['motion_sha256'],RF_peak_N=float(c['rf'].max())) for k,c in cases.items()});(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
 for key,c in cases.items():
  with (OUT/(key+'_forces.csv')).open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['physical_time_s','rectus_femoris_mean_per_side_N',*[n+'_mean_per_side_N' for n,_ in GROUPS]])
   for j,tt in enumerate(c['times']):w.writerow([tt,c['rf'][j],*[np.interp(tt,c['group_times'],row) for row in c['groups']]])
 if preview:print('FINAL_PREVIEWS_READY',flush=True);return
 path=OUT/metrics['file'];writer=imageio_ffmpeg.write_frames(str(path),(1080,1920),fps=FPS,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None)
 hookwriter=imageio_ffmpeg.write_frames(str(OUT/'iron_man_contour_hook_3_5s.mp4'),(1080,1920),fps=FPS,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','2','-movflags','+faststart']);hookwriter.send(None)
 try:
  for k in range(round(FPS*DURATION)):
   t=k/FPS;im=intro_frame(t) if t<INTRO else frame(t-INTRO) if t<INTRO+MOTION else info;arr=np.asarray(im);assert np.all(arr[1536:]==BG);writer.send(arr)
   if t<INTRO:hookwriter.send(arr)
   if k==round(FPS*INTRO)-1:hookwriter.close()
   if k%90==0:print('FINAL_REEL',k,round(FPS*DURATION),flush=True)
 finally:writer.close()
 caption=(OUT/'caption_en.txt').read_text(encoding='utf-8') if (OUT/'caption_en.txt').exists() else ''
 page='<html><meta charset="utf-8"><style>body{background:#0d1726;color:#edf3fb;font:18px system-ui;max-width:900px;margin:30px auto}video{max-height:85vh;max-width:100%}pre{white-space:pre-wrap}</style><video controls playsinline src="'+path.name+'" poster="cover.png"></video><pre>'+html.escape(caption)+'</pre><p><a href="production_notes.json">Production notes and credits</a></p></html>';(OUT/'index.html').write_text(page,encoding='utf-8');print('FINAL_REEL_READY',path,flush=True)
if __name__=='__main__':main()
