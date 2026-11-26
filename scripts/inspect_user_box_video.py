"""Local motion-study frames; no redistribution of a source video."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=float,default=148);p.add_argument('--end',type=float,default=163);p.add_argument('--step',type=float,default=1);p.add_argument('--name',default='source_review');a=p.parse_args()
 folder=ROOT/'datasets/box_jump_reference/yun_user_video';source=folder/'source.webm';out=ROOT/'results/reference_box_jump/yun_user_video';out.mkdir(exist_ok=True)
 reader=imageio_ffmpeg.read_frames(str(source));meta=next(reader);fps=float(meta['fps']);w,h=meta['size'];ts=np.arange(a.start,a.end+1e-8,a.step);want={round(t*fps):float(t)for t in ts};frames={}
 for k,raw in enumerate(reader):
  if k in want:frames[k]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(h,w,3).copy())
  if k>round(a.end*fps):break
 reader.close();cols=4;tw,th=400,225;rh=255;sheet=Image.new('RGB',(cols*tw,((len(frames)+cols-1)//cols)*rh),(13,23,38));d=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
 for cell,(k,im)in enumerate(sorted(frames.items())):
  x=(cell%cols)*tw;y=(cell//cols)*rh;sheet.paste(im.resize((tw,th)),(x,y+30));d.text((x+8,y+4),f'{k/fps:.3f} s',font=font,fill='white')
 sheet.save(out/(a.name+'.png'))
 report=dict(source='https://www.youtube.com/watch?v=ngyj9rSEjLc',publisher='JTBC Entertainment',user_selected_time_s=155.,source_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),container_fps=fps,review_interval_s=[a.start,a.end],physical_playback_rate_verified=False,box_height_independently_measured=False,force_analysis_allowed=False,analysis_only=True)
 (out/(a.name+'_manifest.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
