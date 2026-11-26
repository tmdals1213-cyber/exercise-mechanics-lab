"""Compose real 3D full-body frames into portrait video with a 20% UI reserve."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ohp_fullbody'
DATA=np.load(OUT/'fullbody_motion.npz');META=json.loads((OUT/'fullbody_metadata.json').read_text(encoding='utf-8'))
FONT='C:/Windows/Fonts/malgun.ttf';BOLD='C:/Windows/Fonts/malgunbd.ttf'

def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)

def group(prefixes,index):
 ids=[i for i,n in enumerate(META['muscle_names']) if any(n.startswith(p) for p in prefixes)]
 return float(DATA['force_N'][index,ids].sum())/1000

def compose(index,body_image=None):
 image=Image.new('RGB',(1080,1920),(13,18,25));d=ImageDraw.Draw(image)
 if body_image is None:body_image=Image.open(OUT/'body_frames'/f'body_{index+1:04d}.png').convert('RGB')
 image.paste(body_image,(0,260));d=ImageDraw.Draw(image)
 d.text((58,35),'EXERCISE MECHANICS',font=font(19),fill=(91,213,194))
 d.text((55,69),'전신 장력 · OHP 100 kg',font=font(49,True),fill=(235,241,247))
 d.text((60,137),'180 cm · 90 kg 설정  /  2초 상승',font=font(25),fill=(164,181,198))
 t=float(DATA['times'][index]);phase='락아웃' if t>=2 else '상승'
 d.text((759,140),f'{t:.2f} s  {phase}',font=font(24),fill=(93,215,195))
 values=[group(['DELT'],index),group(['TRI'],index),group(['IL_','LTp','MF_'],index)]
 xpositions=[60,360,665]
 for x,label,value in zip(xpositions,['삼각근','삼두근','척추기립근'],values):
  d.text((x,191),label,font=font(24,True),fill=(182,198,214));d.text((x+145,189),f'{value:.1f} kN',font=font(29,True),fill=(238,241,247))
 d.text((61,229),'근육군 양측 장력 합 · 측정값이 아닌 모델 요구값',font=font(18),fill=(145,163,181))
 # Only the legend overlays the ground below the feet, above the platform UI area.
 d.rounded_rectangle((35,1405,956,1531),radius=16,fill=(19,27,37))
 d.text((57,1414),'색상: 요구 장력 / 모델 기준 등척성 힘',font=font(22),fill=(203,216,229))
 stops=[(0,np.array([36,81,125])),(.25,np.array([30,151,185])),(.65,np.array([74,188,141])),(1.,np.array([246,190,50])),(1.6,np.array([237,67,44])),(2.5,np.array([156,18,35]))]
 for k in range(550):
  ratio=k/549*2.5
  for (a,ca),(b,cb) in zip(stops,stops[1:]):
   if ratio<=b:color=tuple(np.uint8(ca+(cb-ca)*max(0,(ratio-a)/(b-a))));break
  d.line([(57+k,1451),(57+k,1462)],fill=color)
 d.text((57,1467),'0',font=font(17),fill=(172,190,207));d.text((256,1467),'1×',font=font(17),fill=(172,190,207));d.text((557,1467),'2.5×+',font=font(17),fill=(172,190,207))
 d.rectangle((657,1452,673,1468),fill=(115,124,132));d.text((684,1447),'회색: 계산 미대응',font=font(21),fill=(174,189,204))
 d.text((57,1500),'BodyParts3D © DBCLS / CC BY 4.0 · MyoSim 역학 모델',font=font(16),fill=(131,150,168))
 # Repaint the reserved region to guarantee no text/model leaks into it.
 d.rectangle((0,1536,1080,1920),fill=(13,18,25))
 return image

def main():
 indices=[0]*15+list(range(76))+[75]*30
 writer=imageio.get_writer(OUT/'ohp_fullbody_portrait.mp4',fps=30,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart'])
 try:
  for n,index in enumerate(indices):writer.append_data(np.asarray(compose(index)))
 finally:writer.close()
 compose(30).save(OUT/'fullbody_poster.png')
 compose(0).save(OUT/'fullbody_start.png');compose(60).save(OUT/'fullbody_lockout.png')
 (OUT/'preview.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OHP 전신 근육 장력</title><style>body{margin:0;background:#0d1219;color:#ccd8e5;font:15px Malgun Gothic;display:grid;justify-items:center;padding:18px}video{height:88vh;max-width:100%;aspect-ratio:9/16}p{max-width:680px;line-height:1.7}</style><video src="ohp_fullbody_portrait.mp4" controls loop playsinline></video><p>실제 해부학 근육 표면을 사용한 전신 모델. 색상은 기준 등척성 힘 대비 역학적 요구 장력이며 근활성도·EMG·손상 지표가 아닙니다. 회색은 계산 모델에 대응되지 않은 구조입니다. 하단 20%는 플랫폼 UI를 위해 비워 두었습니다.</p></html>''',encoding='utf-8')
 print('Saved full-body portrait video',flush=True)

if __name__=='__main__':main()
