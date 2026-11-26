from pathlib import Path
import json
from PIL import Image,ImageDraw
from compose_fullbody_video import font

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ohp_fullbody';QA=OUT/'anatomy_check'
manifest=json.loads((QA/'views.json').read_text('utf-8'))
for view,title in [('front','정면'),('back','후면')]:
 image=Image.new('RGB',(1080,1920),(13,18,25));d=ImageDraw.Draw(image)
 d.text((43,36),f'전신 근육 형태 확인 · {title}',font=font(41,True),fill=(235,241,247))
 d.text((45,104),'형태 확인용 단색 · 장력 색상 제외 · 봉 숨김',font=font(24),fill=(165,185,205))
 d.text((45,148),'금색 점: 광배근 정지점의 화면 투영(상완골 앞쪽)',font=font(22),fill=(240,195,90))
 for k,(phase,label) in enumerate([('rest','팔 내린 기본 자세'),('start','쇄골 높이 출발'),('middle','상승 중간'),('lockout','락아웃')]):
  x=25+(k%2)*530;y=240+(k//2)*780
  tile=Image.open(QA/f'{view}_{phase}.png').convert('RGB');image.paste(tile,(x,y))
  d.text((x+12,y-43),label,font=font(26,True),fill=(93,215,195))
  for px,py in manifest[f'{view}_{phase}']['lat_insertion_projected_px']:
   d.ellipse((x+px-5,y+py-5,x+px+5,y+py+5),fill=(245,190,65),outline=(25,20,12),width=1)
 d.text((43,1770),'광배근의 좁은 건 끝 → 상완골 근위부 결절간고랑',font=font(24),fill=(226,235,243))
 d.text((43,1813),'몸통 근육은 팔꿈치·전완 뼈에 연결하지 않습니다.',font=font(22),fill=(165,185,205))
 d.text((43,1865),'BodyParts3D © DBCLS · Z-Anatomy © Kervyn et al. · 출처/라이선스: CREDITS',font=font(15),fill=(138,160,180))
 image.save(OUT/f'anatomy_{view}_check.png')
print('ANATOMY_CONTACT_SHEETS_COMPLETE')
