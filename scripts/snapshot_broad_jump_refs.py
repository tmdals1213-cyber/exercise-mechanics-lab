from pathlib import Path
import json,subprocess
from PIL import Image,ImageDraw,ImageFont,ImageOps
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/standing_long_jump_reference'
def main():
 ff=imageio_ffmpeg.get_ffmpeg_exe();font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for name in ['ymca_analysis','pass_knsu_technique','koong_basic']:
  folder=OUT/name;ref=json.loads((folder/'reference.json').read_text(encoding='utf-8'));T=ref['duration_s'];times=[round((i+.5)*T/16,2)for i in range(16)];im=Image.new('RGB',(1280,880),(15,23,35));d=ImageDraw.Draw(im);frames=folder/'reference_frames';frames.mkdir(exist_ok=True)
  for i,t in enumerate(times):
   dest=frames/f'{t:07.2f}.png'
   subprocess.run([ff,'-hide_banner','-loglevel','error','-ss',str(t),'-i',ref['local_file'],'-frames:v','1','-y',str(dest)],check=True,capture_output=True)
   with Image.open(dest)as pic:thumb=ImageOps.contain(pic.convert('RGB'),(318,190))
   x=i%4*320;y=i//4*220;im.paste(thumb,(x+(320-thumb.width)//2,y));d.text((x+8,y+194),f'{t:.2f}s',font=font,fill='white')
  im.save(folder/'reference_contact_sheet.png');print('REFERENCE_SHEET',name,flush=True)
if __name__=='__main__':main()
