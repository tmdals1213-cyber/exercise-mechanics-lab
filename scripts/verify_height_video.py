"""Encoded comparison video format, sampled decoding and UI-safe lower region."""
from pathlib import Path
import json
import imageio.v2 as imageio
import numpy as np
from PIL import Image
OUT=Path(__file__).resolve().parents[1]/'results/ohp_fullbody'
r=imageio.get_reader(OUT/'height_comparison_portrait.mp4');info=r.get_meta_data()
assert info['size']==(1080,1920) and info['fps']==30 and r.count_frames()==900
error=0
for i in [0,90,269,300,539,600,809,899]:
 frame=r.get_data(i);error=max(error,int(np.abs(frame[1536:].astype(int)-[13,18,25]).max()))
r.close();assert error<=4
for policy in ['geometric','same_bmi','same_mass']:
 image=np.asarray(Image.open(OUT/f'height_{policy}_poster.png'));assert np.all(image[1536:]==[13,18,25])
report={'status':'passed','video_resolution':[1080,1920],'fps':30,'frames':900,'duration_s':30,'bottom_reserved_px':384,
 'max_encoded_bottom_channel_error':error,'sections':['geometric','same_bmi','same_mass']}
(OUT/'validation_height_video.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
