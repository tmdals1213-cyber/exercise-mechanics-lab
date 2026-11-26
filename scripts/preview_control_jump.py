from pathlib import Path
import argparse,json,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder');a=ap.parse_args();p=Path(a.folder);z=np.load(p/'control_motion.npz');names=json.loads((p/'body_names.json').read_text());rep=json.loads((p/'control_summary.json').read_text());idx={n:i for i,n in enumerate(names)};im=Image.new('RGB',(1800,900),(13,23,38));d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
 to=rep.get('takeoff_s') or 1.4;td=rep.get('touchdown_s') or 2.
 times=[.5,1.15,to,(to+td)/2,td,min(td+.35,float(z['times'][-1]))]
 for k,t in enumerate(times):
  i=int(np.argmin(abs(z['times']-t)));x=k%3*600;y=k//3*450;C=z['com'][i]
  def screen(v):return(x+300+(-v[1]+C[1])*175,y+405-v[2]*175)
  d.line((x+10,y+405,x+590,y+405),fill=(80,100,120),width=2)
  for side,color in [('l',(86,105,126)),('r',(205,219,229))]:
   chains=[['pelvis','femur_'+side,'tibia_'+side,'talus_'+side,'calcn_'+side,'toes_'+side],['torso','humerus_'+side,'ulna_'+side,'lunate_'+side]]
   for chain in chains:
    pts=[screen(z['body_xpos'][i,idx[n]])for n in chain];d.line(pts,fill=color,width=8)
  d.line([screen(z['body_xpos'][i,idx[n]])for n in ['pelvis','torso','head']],fill=(235,170,83),width=12)
  head=screen(z['body_xpos'][i,idx['head']]+[0,0,.10]);d.ellipse((head[0]-15,head[1]-15,head[0]+15,head[1]+15),fill=(235,170,83))
  cp=screen(C);d.ellipse((cp[0]-5,cp[1]-5,cp[0]+5,cp[1]+5),fill=(244,76,76))
  d.text((x+20,y+15),f'{z["times"][i]:.2f}s / phase {z["phase"][i]}',font=font,fill='white');d.text((x+20,y+48),f'v = {-z["velocity"][i,1]:.2f}, {z["velocity"][i,2]:.2f} m/s',font=small,fill=(170,191,209));d.text((x+20,y+75),f'Hx = {z["angular_momentum"][i,0]:.1f} / Fz = {z["force_world_N"][i,2]:.0f} N',font=small,fill=(170,191,209))
 im.save(p/'control_contact_sheet.png');print(p/'control_contact_sheet.png')
if __name__=='__main__':main()
