from pathlib import Path
import json,subprocess
from PIL import Image,ImageDraw,ImageFont,ImageOps
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'datasets/standing_long_jump_reference'
def main():
 ff=imageio_ffmpeg.get_ffmpeg_exe();font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for name,start,length,fps in [('ymca_analysis',12.,9.,3),('pass_knsu_technique',548.,19.,3)]:
  folder=BASE/name;ref=json.loads((folder/'reference.json').read_text(encoding='utf-8'));dest=folder/'demo_frames';dest.mkdir(exist_ok=True);subprocess.run([ff,'-hide_banner','-loglevel','error','-ss',str(start),'-i',ref['local_file'],'-t',str(length),'-vf',f'fps={fps}','-y',str(dest/'frame_%03d.png')],check=True,capture_output=True);files=sorted(dest.glob('frame_*.png'))
  for page in range((len(files)+15)//16):
   im=Image.new('RGB',(1920,1200),(15,23,35));d=ImageDraw.Draw(im)
   for j,file in enumerate(files[page*16:(page+1)*16]):
    with Image.open(file)as pic:thumb=ImageOps.contain(pic.convert('RGB'),(478,270))
    x=j%4*480;y=j//4*300;im.paste(thumb,(x+(480-thumb.width)//2,y));d.text((x+8,y+274),f'Approx. {start+(page*16+j)/fps:.2f}s',font=font,fill='white')
   im.save(folder/f'demo_sequence_{page}.png')
  print('DEMO_SEGMENT_READY',name,len(files),flush=True)
if __name__=='__main__':main()
