"""English portrait comparison with synchronized repeated complete cycles."""
from pathlib import Path
import json,sys,math
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/grip_comparison'
CYCLES='--full-cycle' in sys.argv;PHYSIOLOGY='--physiology' in sys.argv or CYCLES
CASE_ROOTS={g:OUT/f'grip_{g}'/('cycle' if CYCLES else '')/('physiology' if PHYSIOLOGY else '') for g in [65,80]}
CASES={g:np.load(CASE_ROOTS[g]/'grip_traces.npz') for g in [65,80]}
REPORTS={g:json.loads((CASE_ROOTS[g]/'grip_traces.json').read_text('utf-8')) for g in [65,80]}
FPS=30;REPEATS=3;INTRO=30;ENDING=120
DURATION=float(CASES[65]['times'][-1]) if CYCLES else 2.
STEPS=round(DURATION*FPS)
COLORS={65:(114,190,199),80:(214,174,119)};BG=(13,18,25);FG=(231,236,240);MUTED=(144,158,172)
LABELS={65:'STANDARD GRIP',80:'WIDE GRIP'}
CAPTION='Model-estimated force · Muscles: bilateral / Joint reactions: right'
PANELS=[('delt_front_force_N','Anterior deltoid','kN',1000),('delt_middle_force_N','Middle deltoid','kN',1000),
 ('upper_pec_force_N','Clavicular pec','kN',1000),('triceps_force_N','Triceps','kN',1000),('biceps_force_N','Biceps','kN',1000),
 ('latissimus_force_N','Latissimus dorsi','N',1),('rotator_cuff_force_N','Rotator cuff','kN',1000),
 ('shoulder_resultant_N','Shoulder reaction','kN',1000),('elbow_resultant_N','Elbow reaction','kN',1000),('wrist_resultant_N','Wrist reaction','kN',1000)]

@lru_cache(maxsize=40)
def font(size,bold=False):
 return ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf',size)

def centered(d,text,cx,y,size,color=FG,bold=False):
 f=font(size,bold);box=d.textbbox((0,0),text,font=f);d.text((cx-(box[2]-box[0])/2,y),text,font=f,fill=color)

def upper_limit(value):
 if value<1e-8:return 1.
 step=10**math.floor(math.log10(value))/4
 return math.ceil(value*1.08/step)*step

LIMITS={key:upper_limit(max(float(CASES[g][key][:STEPS+1].max())/div for g in CASES)) for key,_,_,div in PANELS}

def curve(draw,points,color,width=3,dashed=False):
 for i,(a,b) in enumerate(zip(points,points[1:])):
  if not dashed or i%4<2:draw.line([a,b],fill=color,width=width)

def chart(image,key,label,unit,div,index,box):
 d=ImageDraw.Draw(image);x,y,w,h=box;left=x+43;right=x+w-15;top=y+29;bottom=y+h-17;maximum=LIMITS[key]
 d.text((x+9,y),label,font=font(20,True),fill=FG);d.text((x+w-46,y+3),unit,font=font(14),fill=MUTED)
 d.text((x+3,top-8),f'{maximum:g}',font=font(12),fill=MUTED);d.text((x+21,bottom-9),'0',font=font(12),fill=MUTED)
 for fraction in [0,.5,1]:
  yy=bottom-fraction*(bottom-top);d.line((left,yy,right,yy),fill=(36,46,57),width=1)
 if CYCLES:
  for t in [2,2.3,4.3]:
   xx=left+t/DURATION*(right-left);d.line((xx,top,xx,bottom),fill=(40,49,59),width=1)
 cursor=min(float(CASES[65]['times'][index]),DURATION)
 for grip,data in CASES.items():
  t=data['times'][:STEPS+1];values=data[key][:STEPS+1]/div;components=[(values,False)]
  if key.endswith('_force_N'):
   opposing=data[key.replace('_force_N','_opposing_force_N')][:STEPS+1]/div
   components=[(np.maximum(values-opposing,0),False),(opposing,True)]
  for values,dashed in components:
   points=[(left+float(tt)/DURATION*(right-left),bottom-float(v)/maximum*(bottom-top)) for tt,v in zip(t,values)]
   curve(d,points,tuple(int(c*.24) for c in COLORS[grip]),2,dashed)
   past=points[:min(index+1,STEPS+1)];curve(d,past,COLORS[grip],3,dashed)
   if past:
    px,py=past[-1];d.ellipse((px-2,py-2,px+2,py+2),fill=COLORS[grip])
 xx=left+cursor/DURATION*(right-left);d.line((xx,top,xx,bottom),fill=(186,196,207),width=1)
 if key in ['elbow_resultant_N','wrist_resultant_N']:
  for fraction,label in [(0,'0%'),(.5,'50%'),(1,'100%')]:
   tx=left+fraction*(right-left);width=d.textlength(label,font=font(12));d.text((tx-width/2,bottom+2),label,font=font(12),fill=MUTED)

def compose(index,rep=1,preview=False):
 image=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(image)
 d.text((36,14),'EXERCISE MECHANICS',font=font(15,True),fill=MUTED)
 right='GRIP COMPARISON / 01';d.text((1044-d.textlength(right,font=font(15)),14),right,font=font(15),fill=MUTED)
 centered(d,'OVERHEAD PRESS',540,36,48,FG,True)
 centered(d,'How does grip width shift the load?',540,96,22,MUTED)
 centered(d,'vs.',540,141,17,MUTED)
 for grip,x in [(65,28),(80,560)]:
  label=f'{LABELS[grip]}  /  {grip} cm';centered(d,label,x+246,137,24,COLORS[grip],True)
  path=CASE_ROOTS[grip]/'comparison_frames'/('preview.png' if preview else f'frame_{index+1:04d}.png')
  image.paste(Image.open(path).convert('RGB'),(x,174));d=ImageDraw.Draw(image)
  d.line((x,174,x+491,174),fill=COLORS[grip],width=2)
 phase='ASCENT' if index<=60 else 'TOP HOLD' if index<=69 else 'DESCENT' if index<=129 else 'BOTTOM HOLD'
 d.text((37,823),f'100 kg   /   {phase}',font=font(17,True),fill=FG)
 for left,dashed,text in [(350,False,'Assisting'),(523,True,'Opposing')]:
  for s in range(0,35,10):
   if dashed:d.line((left+s,836,left+s+5,836),fill=MUTED,width=2)
  if not dashed:d.line((left,836,left+35,836),fill=MUTED,width=2)
  d.text((left+45,823),text,font=font(16),fill=MUTED)
 counter=f'{rep:02d} / {REPEATS:02d}';d.text((1035-d.textlength(counter,font=font(18,True)),822),counter,font=font(18,True),fill=FG)
 for i,(key,label,unit,div) in enumerate(PANELS):chart(image,key,label,unit,div,index,(28+(i%2)*532,866+(i//2)*117,492,113))
 centered(d,'CYCLE PHASE',540,1450,12,MUTED)
 d.text((43,1475),CAPTION,font=font(16),fill=MUTED)
 d.text((43,1511),'BodyParts3D © DBCLS · Z-Anatomy © Kervyn et al. · CC BY / BY-SA',font=font(13),fill=(111,129,145))
 return image

def summary_image():
 image=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(image)
 d.text((52,45),'GRIP WIDTH / FORCE DISTRIBUTION',font=font(18,True),fill=MUTED)
 d.text((49,79),'OVERHEAD PRESS',font=font(46,True),fill=FG)
 d.text((53,147),'100 kg   ·   MEAN / PEAK   ·   ONE FULL CYCLE' if CYCLES else '100 kg   ·   MEAN / PEAK   ·   ASCENT',font=font(23),fill=MUTED)
 for grip,x in [(65,548),(80,883)]:
  centered(d,LABELS[grip],x,231,24,COLORS[grip],True);centered(d,f'{grip} cm',x,267,19,MUTED)
 for i,(key,label,unit,div) in enumerate(PANELS):
  y=327+i*99;d.line((52,y+76,1024,y+76),fill=(39,50,61),width=1)
  d.text((55,y),label,font=font(25,True),fill=FG);d.text((55,y+36),unit,font=font(17),fill=MUTED)
  for grip,x in [(65,420),(80,755)]:
   s=REPORTS[grip]['summaries'][key];text=f"{s['mean_ascent']/div:.1f} / {s['max_ascent']/div:.1f}"
   d.text((x,y+14),text,font=font(28,True),fill=COLORS[grip])
 d.text((55,1398),'Repeated identical cycles · No fatigue simulation',font=font(20),fill=MUTED)
 d.text((55,1431),'Active + passive muscle force / GH directional stability',font=font(19),fill=MUTED)
 d.text((43,1475),CAPTION,font=font(16),fill=MUTED)
 d.text((43,1511),'BodyParts3D © DBCLS · Z-Anatomy © Kervyn et al. · CC BY / BY-SA',font=font(13),fill=(111,129,145))
 return image

def main():
 summary=summary_image();summary.save(OUT/'grip_comparison_summary.png')
 if '--preview' in sys.argv:
  compose(30,1,False).save(OUT/'grip_comparison_preview.png');return
 timeline=[{'kind':'intro','source_index':0,'rep':1} for _ in range(INTRO)]
 timeline += [{'kind':'cycle','source_index':i,'rep':r} for r in range(1,REPEATS+1) for i in range(STEPS)]
 timeline += [{'kind':'summary'} for _ in range(ENDING)]
 writer=imageio.get_writer(OUT/'ohp_grip_comparison_portrait.mp4',fps=FPS,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart'])
 try:
  intro=compose(0)
  for frame in timeline:
   picture=summary if frame['kind']=='summary' else intro if frame['kind']=='intro' else compose(frame['source_index'],frame['rep'])
   writer.append_data(np.asarray(picture))
 finally:writer.close()
 compose(30).save(OUT/'grip_comparison_poster.png');compose(0).save(OUT/'grip_comparison_start.png');compose(60).save(OUT/'grip_comparison_lockout.png')
 if CYCLES:compose(99).save(OUT/'grip_comparison_descent.png')
 (OUT/'playback_timeline.json').write_text(json.dumps(timeline),encoding='utf-8')
 (OUT/'layout.json').write_text(json.dumps({'canvas':[1080,1920],'bottom_reserved_px':384,'fps':FPS,'frames':len(timeline),
  'language':'en','title':'Overhead Press','comparison':'Standard Grip vs. Wide Grip','repeats':REPEATS,'cycle_duration_s':DURATION,
  'intro_frames':INTRO,'summary_frames':ENDING,'same_camera':True,'complete_cycles':CYCLES,
  'backend':'active_passive_gh' if PHYSIOLOGY else 'legacy_minimum_torque','strength_ceiling_enforced':False,
  'mean_max_interval_s':REPORTS[65]['statistics_interval_s'],'force_display':'solid assisting component and dashed opposing component; nonnegative tensions',
  'main_indicator':'force distribution; no work or power readouts','graph_panels':[{'key':k,'label':l,'unit':u,'axis_max':LIMITS[k]} for k,l,u,_ in PANELS]},indent=2),encoding='utf-8')
 print('GRIP_COMPARISON_VIDEO_COMPLETE',len(timeline),'frames',flush=True)

if __name__=='__main__':main()
