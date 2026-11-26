from pathlib import Path
import argparse,json,math,html
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
from build_motion_study_reel import font,BG,FG,MUTED,ACC
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='baseline');ap.add_argument('--preview',action='store_true');a=ap.parse_args();out=ROOT/'deliverables/controlled_jump_pilot'/a.label;z=np.load(out/'motion.npz');rep=json.loads((out/'control_summary.json').read_text());ts=z['times'];T=float(ts[-1]);files=sorted((out/'study_render').glob('frame_*.png'));assert len(files)==round(T*30)+1;frames=[]
 for file in files:
  with Image.open(file)as im:frames.append([im.convert('RGBA').crop(c).resize((480,800),Image.Resampling.LANCZOS)for c in [(220,70,640,770),(970,70,1390,770)]])
 names=json.loads((out/'body_names.json').read_text());centres=z['body_xpos'][:,names.index('pelvis')].copy();centres[:,2]=(z['contact_positions_m'][:,:,2].min(1)+z['body_xpos'][:,names.index('head'),2]+.13)/2;dt=float(np.median(np.diff(ts)));sigma=.065/dt;radius=max(1,int(np.ceil(3*sigma)));kernel=np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2);kernel/=kernel.sum();centres=np.column_stack([np.convolve(np.pad(centres[:,j],(radius,radius),mode='edge'),kernel,mode='valid')for j in range(3)]);yaw=np.deg2rad(25);elev=np.deg2rad(15);right=np.array([np.cos(yaw),np.sin(yaw),0]);up=np.array([-np.sin(yaw)*np.sin(elev),np.cos(yaw)*np.sin(elev),np.cos(elev)]);ang=np.deg2rad(115);Rside=np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
 to=rep['takeoff_s'];td=rep['touchdown_s'];mask=(ts>=to)&(ts<=td);flight=z['com'][mask];forward=-flight[:,1];initialtoe=-z['contact_positions_m'][0,:,1].min();forward-=initialtoe;force_times=z['physics_times_s'] if 'physics_times_s'in z else ts;force_values=z['physics_force_world_N']if'physics_force_world_N'in z else z['force_world_N'];F=force_values[:,2]/(87*9.80665);ylim=max(4.,math.ceil(F.max()));phases=['READY','COUNTERMOVEMENT','PUSH-OFF','FLIGHT','LANDING','RECOVERY']
 def draw_frame(t,speed):
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
  def txt(x,y,s,size=24,color=FG):d.text((x,y),s,font=font(size,True),fill=color)
  txt(45,35,'CONTROLLED JUMP',42);txt(45,100,'178 CM / 87 KG',23,MUTED);txt(820,100,f'{speed:.2f}x speed',22,MUTED);idx=int(np.argmin(abs(ts-t)));fi=min(round(t/T*(len(frames)-1)),len(frames)-1)
  for v,x in enumerate([30,570]):im.paste(frames[fi][v],(x,175),frames[fi][v])
  native_time=fi/(len(frames)-1)*T;ni=int(np.argmin(abs(ts-native_time)));C=z['com'][ni];target=centres[ni];xy=np.r_[target[:2],0.]
  for v,x,cropx in [(0,30,220),(1,570,970)]:
   relative=C-xy-1.3*right-[0,0,target[2]] if v==0 else Rside@(C-xy)+1.3*right-[0,0,target[2]];sx=800+relative@right*1600/5.6;sy=430-relative@up*1600/5.6;cx=x+(sx-cropx)*480/420;cy=175+(sy-70)*800/700;d.ellipse((cx-7,cy-7,cx+7,cy+7),fill=ACC,outline=(255,230,205),width=2)
  txt(45,950,phases[int(z['phase'][idx])],27,ACC)
  if t>=td:txt(665,950,f'{rep["first_contact_heel_distance_m"]:.2f} m',34)
  for x,title in [(70,'FLIGHT / COM'),(600,'GROUND REACTION')]:txt(x,1045,title,23)
  x0,y0,w,h=85,1110,390,240;xmin=min(-.05,float(forward.min())-.05);xmax=max(.6,float(forward.max())+.10);zmax=max(1.3,float(flight[:,2].max())+.1)
  def coord(v):return(x0+w*((-v[1]-initialtoe)-xmin)/(xmax-xmin),y0+h*(1-v[2]/zmax))
  for u in [0,.5,1.]:d.line((x0,y0+h*u,x0+w,y0+h*u),fill=(43,60,77))
  d.line([coord(v)for v in flight],fill=(90,110,130),width=3);current=max(to,min(t,td));cur=np.array([np.interp(current,ts,z['com'][:,j])for j in range(3)]);cx,cy=coord(cur);d.ellipse((cx-6,cy-6,cx+6,cy+6),fill=ACC);txt(x0,1370,'HORIZONTAL DISTANCE / m',17,MUTED)
  x0,y0,w,h=620,1110,390,240
  for val in sorted({0,1,int(ylim)}):
   yy=y0+h*(1-val/ylim);d.line((x0,yy,x0+w,yy),fill=(43,60,77));txt(x0-30,yy-10,str(val),17,MUTED)
  points=[(x0+w*tt/T,y0+h*(1-f/ylim))for tt,f in zip(force_times,F)];d.line(points,fill=(104,126,146),width=3);cx=x0+w*t/T;d.line((cx,y0,cx,y0+h),fill=ACC,width=2);txt(x0,1370,'BODY WEIGHT',17,MUTED);txt(45,1450,'Orange marker: whole-body centre of mass',20,MUTED)
  assert np.all(np.asarray(im)[1536:]==BG);return im
 draw_frame((to+td)/2,.6).save(out/'cover.png');draw_frame(T,1).save(out/'ending.png')
 if a.preview:return
 path=out/'controlled_jump_review.mp4';writer=imageio_ffmpeg.write_frames(str(path),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None);durations=[T,T/.6]
 try:
  for speed,duration in zip([1.,.6],durations):
   for k in range(round(duration*30)):writer.send(np.asarray(draw_frame(min(T,k/30*speed),speed)))
 finally:writer.close()
 notes='This is the sagittal rigid-foot control pilot. Anatomy is neutral grey; no muscle tensions are encoded. Internal joint torques and native contact generate takeoff, flight and landing. Lateral motion is constrained. The short demonstration jump is not a maximum-distance solution.'
 (out/'index.html').write_text('<meta charset="utf-8"><style>body{background:#0d1726;color:#edf3fb;font:18px system-ui;max-width:900px;margin:30px auto}video{max-height:85vh;max-width:100%}</style><video controls playsinline src="'+path.name+'" poster="cover.png"></video><p>'+html.escape(notes)+'</p>',encoding='utf-8');(out/'review_manifest.json').write_text(json.dumps(dict(video=path.name,duration_s=sum(durations),fps=30,neutral_anatomy=True,muscle_tension_colours=False,bottom_reserved_px=384,scope=notes),indent=2));print('CONTROL_VIDEO_READY',path,flush=True)
if __name__=='__main__':main()
