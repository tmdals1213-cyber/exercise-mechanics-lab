"""Portrait composition: primary front ISO and three synchronized helper views."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
from compose_fullbody_video import font,group,DATA,META

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ohp_fullbody';MV=OUT/'multiview'
METRICS=np.load(OUT/'bar_joint_metrics.npz')
PANELS={'iso':(32,282,660,1116),'front':(718,282,262,320),'side':(718,676,262,320),'back':(718,1070,262,320)}

def compose(index,preview=False):
 image=Image.new('RGB',(1080,1920),(13,18,25));d=ImageDraw.Draw(image)
 d.text((58,34),'EXERCISE MECHANICS',font=font(19),fill=(91,213,194))
 d.text((54,72),'OHP 100 kg · 추정 근육 장력',font=font(44,True),fill=(236,242,248))
 d.text((59,133),'180 cm · 90 kg / 그립 65 cm · 2초 상승',font=font(22),fill=(164,181,198))
 t=float(DATA['times'][index]);d.text((758,136),f'{t:.2f} s  '+('락아웃' if t>=2 else '상승'),font=font(23),fill=(93,215,195))
 for x,label,value in zip([59,299,539,779],['상부 가슴','삼각근','삼두근','척추기립근'],[group(['PECM1'],index),group(['DELT'],index),group(['TRI'],index),group(['IL_','LTp','MF_'],index)]):
  d.text((x,177),label,font=font(21,True),fill=(182,198,214));d.text((x,200),f'{value:.1f} kN',font=font(27,True),fill=(238,241,247))
 d.text((60,234),'양측 합 · 역동역학/힘 분담 추정 · 기준 근력 상한 미적용',font=font(16),fill=(145,163,181))
 labels={'iso':'전면 ISO · 메인','front':'정면','side':'측면 · 원판 숨김','back':'후면'}
 for key,(x,y,w,h) in PANELS.items():
  filename=MV/key/('preview.png' if preview else f'frame_{index+1:04d}.png')
  panel=Image.open(filename).convert('RGB');assert panel.size==(w,h)
  mask=Image.new('L',(w,h));md=ImageDraw.Draw(mask);md.rounded_rectangle((0,0,w-1,h-1),radius=12,fill=255)
  image.paste(panel,(x,y),mask)
  d.rounded_rectangle((x,y,x+w-1,y+h-1),radius=12,outline=(43,59,76),width=2)
  d.text((x+9,y-31),labels[key],font=font(22,True),fill=(93,215,195) if key=='iso' else (193,207,221))
 d.text((726,1005),'바벨 경로: 뒤로 약 18 cm',font=font(17),fill=(93,215,195))
 values=[f"{METRICS['bar_height_gain_m'][index]*100:.1f} cm",f"{np.linalg.norm(METRICS['bar_velocity_m_s'][index]):.2f} m/s",f"{METRICS['bar_hand_work_J'][index]:.0f} J",f"{METRICS['bar_power_W'][index]:.0f} W"]
 for x,label,value in zip([55,213,371,529],['상승 높이','바 속도','바벨에 한 일','기계적 출력'],values):
  d.text((x,302),label,font=font(17),fill=(157,180,199));d.text((x,330),value,font=font(25,True),fill=(93,215,195))
 d.text((55,367),'손→바벨: 위치에너지 + 운동에너지 / 출력 = 힘 · 속도',font=font(15),fill=(133,155,176))
 external=np.linalg.norm(METRICS['bar_external_joint_moment_world_Nm'][index],axis=1)
 for x,label,value in [(55,'팔꿈치 · 바벨 외력모멘트',external[0]),(355,'손목 · 바벨 외력모멘트',external[1])]:
  d.text((x,400),label,font=font(17),fill=(157,180,199));d.text((x,426),f'{value:.1f} N·m',font=font(25,True),fill=(230,237,243))
 d.text((55,462),'우측 기준 · 관절 중심에서의 모멘트 크기 · 관절면 압력과 구분',font=font(14),fill=(133,155,176))
 d.rounded_rectangle((35,1415,980,1531),radius=14,fill=(19,27,37))
 d.text((57,1422),'색상: 추정 장력 / 모델 기준 등척성 힘',font=font(22),fill=(203,216,229))
 stops=[(0,np.array([36,81,125])),(.25,np.array([30,151,185])),(.65,np.array([74,188,141])),(1.,np.array([246,190,50])),(1.6,np.array([237,67,44])),(2.5,np.array([156,18,35]))]
 for k in range(550):
  ratio=k/549*2.5
  for (a,ca),(b,cb) in zip(stops,stops[1:]):
   if ratio<=b:color=tuple(np.uint8(ca+(cb-ca)*max(0,(ratio-a)/(b-a))));break
  d.line([(57+k,1458),(57+k,1469)],fill=color)
 d.text((57,1472),'0',font=font(16),fill=(172,190,207));d.text((256,1472),'1×',font=font(16),fill=(172,190,207));d.text((557,1472),'2.5×+',font=font(16),fill=(172,190,207))
 d.rectangle((656,1457,673,1474),fill=(145,125,112));d.text((685,1452),'회갈색: 미계산 근육',font=font(19),fill=(174,189,204))
 d.text((57,1502),'BodyParts3D © DBCLS · Z-Anatomy © Kervyn et al. · CC BY / BY-SA',font=font(15),fill=(131,150,168))
 d.rectangle((0,1536,1080,1920),fill=(13,18,25))
 return image

def main():
 if '--preview' in sys.argv:
  compose(30,True).save(OUT/'multiview_layout_preview.png');print('Layout preview saved');return
 indices=[0]*15+list(range(76))+[75]*30
 writer=imageio.get_writer(OUT/'ohp_multiview_portrait.mp4',fps=30,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart'])
 try:
  for index in indices:writer.append_data(np.asarray(compose(index)))
 finally:writer.close()
 compose(30).save(OUT/'multiview_poster.png');compose(0).save(OUT/'multiview_start.png');compose(60).save(OUT/'multiview_lockout.png')
 (OUT/'multiview_layout.json').write_text(json.dumps({'canvas':[1080,1920],'bottom_reserved_px':384,'panels':PANELS,'physics_data':'fullbody_motion.npz','same_data_in_all_views':True},indent=2),encoding='utf-8')
 (OUT/'multiview_preview.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OHP · 장력과 일·출력</title><style>body{margin:0;background:#0d1219;color:#ccd8e5;font:15px Malgun Gothic;display:grid;justify-items:center;padding:18px}video{height:88vh;max-width:100%;aspect-ratio:9/16}p{max-width:680px;line-height:1.7}</style><video src="ohp_multiview_portrait.mp4" controls loop playsinline></video><p>그립 65 cm. 상단 장력은 역동역학·힘 분담 추정이며 실제 EMG나 수행 가능성의 측정값이 아닙니다. 바벨에 한 일은 위치에너지 증가와 운동에너지 증가의 합, 출력은 손이 바벨에 가하는 힘과 속도의 내적입니다. 정지해서 버틸 때도 힘은 남지만 바벨 기계적 출력은 0입니다. 관절 수치는 바벨 외력의 모멘트이며 관절면 접촉압력이 아닙니다. 하단 20%는 플랫폼 UI를 위해 비웠습니다.</p><p><a href="bar_joint_metrics.json">모델 관절 합력·순토크 및 계산 가정</a></p></html>''',encoding='utf-8')
 print('Synchronized four-view video saved')

if __name__=='__main__':main()
