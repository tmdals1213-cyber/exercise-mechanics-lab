"""Portrait educational draft: baseline full-body motion and 3 height policies."""
from pathlib import Path
import json,csv,sys
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from compose_fullbody_video import font

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ohp_fullbody'
REPORT=json.loads((OUT/'height_comparison.json').read_text('utf-8'))
POLICIES=['geometric','same_bmi','same_mass']
BG=(13,18,25);WHITE=(235,241,247);MUTED=(160,180,197);ACCENT=(93,215,195)
NOTES={
 'geometric':['체형 전체를 비례 축소하면','체중과 근육 단면적도 작아집니다.'],
 'same_bmi':['BMI가 같아도 근육량·근력은 가정입니다.','여기서는 근육 질량 비율을 같게 두었습니다.'],
 'same_mass':['체중을 같게 두면 체형의 굵기가 달라집니다.','키의 효과와 근육 단면적 가정을 구분합니다.']}

def compose(policy,index,conclusion=False):
 image=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(image)
 d.text((54,34),'EXERCISE MECHANICS · 비교 초안',font=font(19),fill=ACCENT)
 d.text((51,74),'키가 작으면',font=font(48,True),fill=WHITE)
 d.text((51,134),'웨이트에 유리할까?',font=font(48,True),fill=WHITE)
 cases=[c for c in REPORT['cases'] if c['policy']==policy]
 label=cases[0]['label'];d.text((55,207),f'{POLICIES.index(policy)+1}/3  {label} · 같은 100 kg',font=font(26,True),fill=ACCENT)
 panel=Image.open(OUT/'multiview/iso'/f'frame_{index+1:04d}.png').convert('RGB').resize((590,998),Image.Resampling.LANCZOS)
 image.paste(panel,(30,282));d=ImageDraw.Draw(image)
 d.text((46,291),'180 cm 기준 동작 시연',font=font(21),fill=ACCENT)
 d.text((46,324),'쇄골 높이 출발 · 2초 상승',font=font(19),fill=MUTED)
 for k,c in enumerate(cases):
  x=642;y=283+k*325
  d.rounded_rectangle((x,y,1040,y+307),radius=14,fill=(23,32,43),outline=(47,66,82),width=2)
  d.text((x+18,y+13),f"{c['height_cm']:.0f} cm",font=font(34,True),fill=WHITE)
  d.text((x+206,y+24),f"{c['mass_kg']:.1f} kg",font=font(24),fill=MUTED)
  d.text((x+18,y+64),f"바 이동  {c['bar_travel_m']*100:.1f} cm",font=font(24),fill=MUTED)
  d.text((x+18,y+102),f"바에 한 일  {c['bar_gravitational_work_J']:.0f} J",font=font(27,True),fill=ACCENT)
  ratio=c['bar_gravitational_work_J']/max(r['bar_gravitational_work_J'] for r in cases)
  d.rounded_rectangle((x+18,y+144,x+18+350*ratio,y+153),radius=4,fill=ACCENT)
  d.text((x+18,y+173),'모델 추정 · 양측 최대 장력 합',font=font(16),fill=MUTED)
  d.text((x+18,y+200),f"삼각근  {c['peak_deltoid_sum_N']/1000:.1f} kN",font=font(23),fill=WHITE)
  d.text((x+18,y+234),f"삼두근  {c['peak_triceps_sum_N']/1000:.1f} kN",font=font(23),fill=WHITE)
  d.text((x+18,y+271),f"가정한 근력 기준  {c['relative_reference_strength']*100:.0f}%",font=font(17),fill=MUTED)
 d.text((51,1303),'짧아지면 이동 거리와 바에 한 일은 감소',font=font(29,True),fill=WHITE)
 d.text((51,1346),'근육의 지렛대도 짧아져 장력은 비례 감소하지 않음',font=font(24),fill=ACCENT)
 lines=NOTES[policy] if not conclusion else ['유리함은 체중·근육량·비교 중량 조건에 따라 달라집니다.','키 하나로 수행 능력이나 부상 위험을 단정할 수 없습니다.']
 for j,line in enumerate(lines):d.text((52,1393+j*30),line,font=font(21),fill=MUTED)
 d.text((52,1463),'같은 관절각·템포 / 그립·경로는 키에 비례 / 근력 상한 미적용',font=font(16),fill=MUTED)
 d.text((52,1488),'근육 힘은 역동역학·최적화 추정. 일은 바벨의 순중력 일(mgΔh).',font=font(16),fill=MUTED)
 d.text((52,1512),'BodyParts3D © DBCLS · Z-Anatomy © Kervyn et al. · CC BY / BY-SA',font=font(14),fill=MUTED)
 return image

def main():
 for policy in POLICIES:compose(policy,30).save(OUT/f'height_{policy}_poster.png')
 with (OUT/'height_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(REPORT['cases'][0]));writer.writeheader();writer.writerows(REPORT['cases'])
 if '--preview' in sys.argv:return
 writer=imageio.get_writer(OUT/'height_comparison_portrait.mp4',fps=30,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart'])
 try:
  for policy in POLICIES:
   # Nine seconds per policy: 1 s initial pause, 2 s ascent, remaining lockout.
   for video_frame in range(270):
    index=int(np.clip(video_frame-30,0,75));writer.append_data(np.asarray(compose(policy,index)))
  still=np.asarray(compose('same_mass',75,True))
  for _ in range(90):writer.append_data(still)
 finally:writer.close()
 print('HEIGHT_VIDEO_COMPLETE',flush=True)

if __name__=='__main__':main()
