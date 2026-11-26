"""Render a genuine 1080x1920 MP4 from model paths and calculated force demand.

The muscle envelopes are illustrative fusiform shapes, not tissue segmentation.
Bottom 20% stays empty for social-video account/caption overlays.
"""
from pathlib import Path
import json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ohp_100kg_2s'
WIDTH,HEIGHT=1080,1920
FONT=Path('C:/Windows/Fonts/malgun.ttf')
BOLD=Path('C:/Windows/Fonts/malgunbd.ttf')


def font(size,bold=False):return ImageFont.truetype(str(BOLD if bold else FONT),size)


def palette(ratio):
 stops=[(0,np.array([96,61,65])),(.35,np.array([166,89,63])),(.75,np.array([239,151,68])),(1.2,np.array([255,91,67])),(2.5,np.array([255,66,83]))]
 for (a,ca),(b,cb) in zip(stops,stops[1:]):
  if ratio<=b:return tuple(np.uint8(ca+(cb-ca)*max(0,(ratio-a)/(b-a))))
 return tuple(stops[-1][1])


def project(p):
 x,y,z=p; angle=math.radians(19)
 return np.array([480+650*(x*math.cos(angle)+z*math.sin(angle)),1100-650*(y-1.476)])


def mirrored(points,sign):
 a=np.asarray(points,dtype=float).copy();a[:,0]*=sign;return a


def stroke(draw,points,color,width):
 p=[tuple(project(v)) for v in points]
 draw.line(p,fill=color,width=round(width),joint='curve')
 r=width/2
 for x,y in [p[0],p[-1]]:draw.ellipse((x-r,y-r,x+r,y+r),fill=color)


def belly(draw,points,radius,color):
 if len(points)<2:return
 points=np.asarray(points)
 lengths=np.linalg.norm(np.diff(points,axis=0),axis=1);cumulative=np.r_[0,np.cumsum(lengths)]
 if cumulative[-1]<1e-6:return
 sample=np.linspace(0,cumulative[-1],42)
 sampled=np.stack([np.interp(sample,cumulative,points[:,j]) for j in range(3)],axis=1)
 screen=np.array([project(p) for p in sampled]);direction=np.gradient(screen,axis=0)
 direction/=np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-5)
 normal=np.stack([-direction[:,1],direction[:,0]],axis=1)
 # The narrow ends indicate tendon-like attachments, not a measured tendon area.
 profile=(.12+.88*np.sin(np.linspace(0,math.pi,len(screen)))**.7)*radius*650
 left=screen+normal*profile[:,None];right=screen-normal*profile[:,None]
 edge=tuple(int(c*.72) for c in color)
 draw.polygon([tuple(p) for p in np.r_[left,right[::-1]]],fill=edge)
 for width,brightness in [(.78,.9),(.48,1.07),(.20,1.15)]:
  a=screen+normal*(profile*width)[:,None];b=screen-normal*(profile*width)[:,None]
  shade=tuple(min(255,int(c*brightness)) for c in color)
  draw.polygon([tuple(p) for p in np.r_[a,b[::-1]]],fill=shade)
 shadow=tuple(int(c*.62) for c in color)
 draw.line([tuple(p) for p in right],fill=shadow,width=2)
 highlight=tuple(min(255,int(c*.8+55)) for c in color)
 for fraction in [-.45,0,.45]:
  fiber=screen+normal*(profile*fraction)[:,None]
  draw.line([tuple(p) for p in fiber[4:-4]],fill=highlight,width=1)


def render(data,index,overlay_safe_guide=False):
 f=data['frames'][index];summary=data['summary'];names=data['muscle_names']
 im=Image.new('RGB',(WIDTH,HEIGHT),(13,18,25));d=ImageDraw.Draw(im)
 # Deliberately no content below y=1536.
 d.text((66,64),'EXERCISE MECHANICS',font=font(22),fill=(94,219,196))
 d.text((62,105),'OHP 100 kg',font=font(64,True),fill=(242,245,249))
 d.text((66,184),'180 cm · 90 kg  /  2초 상승 → 락아웃',font=font(27),fill=(166,181,197))
 exceeds=not summary['all_frames_bounded_feasible']
 if exceeds:d.text((68,222),'가정된 동작 · 원본 모델 기준력 초과 구간 포함',font=font(18),fill=(225,171,116))
 d.rounded_rectangle((58,247,925,532),radius=20,fill=(24,33,44),outline=(48,63,79),width=2)
 d.text((85,261),'근육 / 한쪽 기준',font=font(21),fill=(154,172,190))
 d.text((440,261),'요구 장력',font=font(21),fill=(154,172,190))
 d.text((710,261),'누적 부하',font=font(21),fill=(154,172,190))
 display=['deltoid','triceps','pectoralis']
 for row,key in enumerate(display):
  group=data['groups'][key];y=310+row*72
  ids=[names.index(n) for n in group['muscles']]
  ratio=max(f['reference_force_ratio'][i] for i in ids)
  d.rounded_rectangle((84,y+7,91,y+44),radius=3,fill=palette(ratio))
  d.text((109,y),group['label'],font=font(32,True),fill=(227,233,239))
  d.text((109,y+39),f"동작 최고 {group['peak_force_N']:,.0f} N",font=font(17),fill=(158,174,190))
  number_color=palette(ratio) if ratio>.2 else (173,187,202)
  d.text((432,y-3),f"{group['force_N'][index]:,.0f}",font=font(39,True),fill=number_color)
  d.text((618,y+12),'N',font=font(23),fill=(154,172,190))
  d.text((709,y),f"{group['impulse_Ns'][index]/1000:.2f}",font=font(33,True),fill=(224,231,239))
  d.text((817,y+12),'kN·s',font=font(20),fill=(154,172,190))
 t=f['time_s'];progress=min(t/2,1)
 d.rounded_rectangle((67,553,916,559),radius=3,fill=(43,57,72))
 if progress>0:d.rounded_rectangle((67,553,67+849*progress,559),radius=3,fill=(94,219,196))
 phase='락아웃 유지' if t>=2 else '가속' if t<.15 else '감속' if t>1.85 else '정속 상승'
 d.text((70,574),f'{t:.2f} s / 2.50 s',font=font(23),fill=(167,183,199))
 d.text((720,574),phase,font=font(23),fill=(94,219,196))
 # Anatomical context, stylized body shell; center stage stays inside the safe area.
 torso=[[-.075,1.58,0],[-.20,1.51,0],[-.24,1.39,0],[-.22,1.22,0],[-.15,1.06,0],[.15,1.06,0],[.22,1.22,0],[.24,1.39,0],[.20,1.51,0],[.075,1.58,0]]
 d.polygon([tuple(project(p)) for p in torso],fill=(65,74,85))
 # Rib and abdominal contours are visual context, not an additional mechanical model.
 for sign in [-1,1]:
  for y in [1.39,1.34,1.29]:stroke(d,[[sign*.01,y,0.015],[sign*.17,y-.025,.015]],(96,104,112),2)
  for y in [1.21,1.15,1.09]:
   p=project([sign*.065,y,.025]);d.rounded_rectangle((p[0]-24,p[1]-17,p[0]+24,p[1]+17),radius=12,fill=(83,88,95))
 stroke(d,[[0,1.55,0],[0,1.64,0]],(134,140,145),42)
 head=project([0,1.705,0]);d.ellipse((head[0]-49,head[1]-68,head[0]+49,head[1]+60),fill=(177,180,180),outline=(218,218,210),width=2)
 # Nose / face contour, deliberately neutral.
 d.arc((head[0]-23,head[1]-29,head[0]+35,head[1]+35),-35,65,fill=(141,149,151),width=3)
 joints=f['joints']
 for sign in [-1,1]:
  arm=mirrored([joints['humerus'],joints['ulna'],joints['hand'],f['grip']],sign)
  stroke(d,arm[:2],(92,96,101),57);stroke(d,arm[1:3],(101,105,108),43);stroke(d,arm[2:],(171,174,171),28)
  stroke(d,arm[:3],(205,202,179),10)
 # Thin forearm paths first, large shoulder/chest/arm bellies last.
 order=sorted(range(len(names)),key=lambda i:(0 if i>=24 else 1, np.mean(np.asarray(f['paths'][i])[:,2]) if f['paths'][i] else 0))
 f0=[m['max_isometric_force_N'] for m in summary['muscles']]
 for i in order:
  name=names[i]
  if name in ['INFSP','SUBSC','TMIN','TMAJ','CORB'] or name.startswith('LAT'):continue
  if name.startswith('DELT'):radius=.033
  elif name.startswith('TRI'):radius=.025
  elif name.startswith('PECM'):radius=.040
  elif name.startswith('LAT'):radius=.027
  elif name.startswith('BIC'):radius=.026
  elif i>=24:radius=.007
  else:radius=.014
  color=palette(f['reference_force_ratio'][i])
  for sign in [-1,1]:belly(d,mirrored(f['paths'][i],sign),radius,color)
 # Hands and actual bar center; sleeves are cropped to the portrait composition.
 gy,gz=f['bar_y_m'],f['bar_z_m']
 left=project([-.78,gy,gz]);right=project([.78,gy,gz])
 d.line([tuple(left),tuple(right)],fill=(202,213,221),width=12)
 for sign in [-1,1]:
  p=project([sign*.64,gy,gz]);d.rounded_rectangle((p[0]-22,p[1]-66,p[0]+22,p[1]+66),radius=8,fill=(73,91,111),outline=(139,159,177),width=3)
  p=project([sign*.275,gy,gz]);d.ellipse((p[0]-14,p[1]-15,p[0]+14,p[1]+15),fill=(209,205,186))
 # Compact reading key and model scope are kept ABOVE the reserved bottom area.
 d.text((69,1417),'색상  =  요구 장력 / 모델 기준 등척성 힘',font=font(24),fill=(167,183,199))
 for k in range(220):d.line([(70+k*2,1460),(70+k*2,1473)],fill=palette(k/110),width=2)
 d.text((550,1446),'어깨·팔꿈치 2축 추정',font=font(23),fill=(174,188,202))
 d.text((69,1492),'근육 경로 도식 · 기준력 초과 ≠ 실제 근활성도',font=font(23),fill=(167,183,199))
 # Nothing is drawn below the 80% line (1536 px).
 if overlay_safe_guide:
  d.line([(0,1536),(WIDTH,1536)],fill=(108,118,130),width=2)
  d.text((66,1580),'플랫폼 UI 영역 — 핵심 정보 없음',font=font(25),fill=(95,106,120))
 return im


def main():
 data=json.loads((OUT/'muscle_motion.json').read_text(encoding='utf-8'))
 # Freeze the first/last data frames for reading, without accumulating extra load.
 fps=30;indices=[0]*15+list(range(0,len(data['frames']),2))+[len(data['frames'])-1]*30
 writer=imageio.get_writer(OUT/'ohp_100kg_portrait.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=1,
                          ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'])
 try:
  for n,index in enumerate(indices):
   im=render(data,index);writer.append_data(np.asarray(im))
   if n%30==0:print(f'Rendering {n}/{len(indices)}',flush=True)
 finally:writer.close()
 render(data,60).save(OUT/'portrait_poster.png')
 render(data,60,True).save(OUT/'portrait_safe_area.png')
 (OUT/'portrait_preview.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OHP · 세로 영상</title><style>body{margin:0;background:#0d1219;color:#dce4ef;font:15px Malgun Gothic,sans-serif;display:grid;justify-items:center;padding:20px}video{height:85vh;max-width:100%;aspect-ratio:9/16}p{max-width:650px;line-height:1.8}</style><video src="ohp_100kg_portrait.mp4" controls loop playsinline></video><p>1080×1920 · 30fps · 하단 20% 정보 제외. 근육 형상은 OpenSim 경로 기반 도식입니다. 장력은 어깨 거상·팔꿈치 굴곡 두 축의 역학적 요구값이며, 누적 부하는 힘–시간 적분입니다. 실제 근활성도·근비대·손상 예측이 아닙니다.</p></html>''',encoding='utf-8')
 print('Saved 1080x1920 portrait MP4 and preview images',flush=True)


if __name__=='__main__':main()
