"""Check synchronized camera outputs, portrait format, and reserved overlay area."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ohp_fullbody';MV=OUT/'multiview'
views=json.loads((MV/'views.json').read_text(encoding='utf-8'));layout=json.loads((OUT/'multiview_layout.json').read_text())
assert set(views['views'])=={'iso','front','side','back'}
assert views['frame_count']==76 and views['same_physics_and_colour_data']
for name,info in views['views'].items():
 for index in range(1,77):
  filename=MV/name/f'frame_{index:04d}.png'
  with Image.open(filename) as image:assert image.size==tuple(info['size']),(name,index)
poster=np.asarray(Image.open(OUT/'multiview_poster.png').convert('RGB'))
assert poster.shape==(1920,1080,3)
assert np.all(poster[1536:]==np.array([13,18,25]))
for name,(x,y,w,h) in layout['panels'].items():assert x>=0 and y>=0 and x+w<=1080 and y+h<=1536
front=np.array(views['views']['front']['direction']);back=np.array(views['views']['back']['direction']);side=np.array(views['views']['side']['direction'])
assert np.dot(front,back)<-.9999 and abs(np.dot(front,side))<1e-6
reader=imageio.get_reader(OUT/'ohp_multiview_portrait.mp4');info=reader.get_meta_data()
assert info['size']==(1080,1920) and abs(info['fps']-30)<.01
for index in [15,45,75,120]:assert reader.get_data(index).shape==(1920,1080,3)
reader.close()
report={'status':'passed','views':list(views['views']),'synchronized_frames_per_view':76,'video_resolution':list(info['size']),
 'fps':info['fps'],'bottom_reserved_px':384,
 'physics_data_sha256':hashlib.sha256((OUT/'fullbody_motion.npz').read_bytes()).hexdigest()}
(OUT/'validation_multiview.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
