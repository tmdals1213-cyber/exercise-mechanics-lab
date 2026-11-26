from pathlib import Path
import argparse,json,math,html
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
from build_motion_study_reel import font,BG,FG,MUTED,ACC
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='broad_jump_starter');ap.add_argument('--preview',action='store_true');a=ap.parse_args();out=ROOT/'deliverables/controlled_jump_pilot'/a.label;z=np.load(out/'motion.npz');rep=json.loads((out/'control_summary.json').read_text());cam=json.loads((out/'fixed_side_camera_v2.json').read_text());times=z['times'];T=float(times[-1]);files=sorted((out/'fixed_side_render_v2').glob('frame_*.png'));tracked=sorted((out/'study_render').glob('frame_*.png'));assert len(files)==len(tracked)==round(T*30)+1;frames=[]
 for side,track in zip(files,tracked):
  with Image.open(side)as im:mainview=im.convert('RGBA').crop((0,190,1080,980))
  with Image.open(track)as im:mini=[im.convert('RGBA').crop(c).resize(size,Image.Resampling.LANCZOS)for c,size in [((220,70,640,770),(228,380)),((900,70,1470,770),(309,380))]]
  frames.append((mainview,mini))
 toe_y=cam['initial_foremost_sole_y_m'];ppm=1080/cam['ortho_width_m'];centre=cam['camera_center_forward_m'];cz=cam['camera_center_z_m'];ground_y=140+525+cz*ppm-190;takeoff=rep['takeoff_s'];landing=rep['touchdown_s'];flight_ids=np.flatnonzero((times>=takeoff)&(times<=landing));velocity=rep['takeoff_velocity_m_s'];angle=math.degrees(math.atan2(velocity[2],-velocity[1]));distance=rep.get('valid_contact_distance_m')or rep['first_contact_heel_distance_m'];labels=['READY','COUNTERMOVEMENT','PUSH-OFF','FLIGHT','LANDING','RECOVERY']
 contact_episode=rep['parameters'].get('stop_after_touchdown_s')is not None
 hold_s=.9 if contact_episode else 0.
 ruler_max=max(2.5,math.ceil(distance*2)/2)
 if contact_episode:labels[4]='FIRST CONTACT'
 def project(p):return(540+(-p[1]+toe_y-centre)*ppm,140+525-(p[2]-cz)*ppm-190)
 def frame(t,speed,hold_progress=0.):
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
  def text(x,y,s,size=24,color=FG,b=True):d.text((x,y),s,font=font(size,b),fill=color)
  fi=min(round(t/T*(len(frames)-1)),len(frames)-1);native_time=fi/(len(frames)-1)*T;i=int(np.argmin(abs(times-native_time)));body,mini=frames[fi];im.paste(body,(0,140),body)
  text(45,22,'@biomechanics_ksm',20,MUTED);text(45,61,'STANDING LONG JUMP',40);text(45,113,labels[int(z['phase'][i])],23,ACC);text(865,113,f'{speed:.2f}x speed',21,MUTED)
  # World-coordinate ruler stays fixed while the person moves across it.
  d.line((35,ground_y,1045,ground_y),fill=(94,115,135),width=2);zero=540-centre*ppm
  for m in np.arange(0,ruler_max+.01,.25):
   x=zero+m*ppm;major=abs(m*2-round(m*2))<1e-8;d.line((x,ground_y-6,x,ground_y+(15 if major else 8)),fill=MUTED,width=2)
   if major:text(x-18,ground_y+20,f'{m:.1f}',18,MUTED)
  d.line((zero,ground_y-40,zero,ground_y+18),fill=FG,width=3);text(45,ground_y+20,'m',18,MUTED)
  contact_shown=t>=landing-.001
  if contact_shown:
   end=zero+distance*ppm;d.line((zero,ground_y,end,ground_y),fill=ACC,width=5);d.line((end,ground_y-20,end,ground_y+20),fill=ACC,width=3)
  # Reveal the airborne COM trajectory after touchdown, leaving the jump clear.
  if contact_shown:
   u=float(np.clip((max(0.,t-landing)+hold_progress)/.65,0,1));u=u*u*(3-2*u);n=max(1,round(u*len(flight_ids)));path=[project(z['com'][j])for j in flight_ids[:n]]
   if len(path)>1:d.line(path,fill=ACC,width=3)
  x,y=project(z['com'][i]);d.ellipse((x-6,y-6,x+6,y+6),fill=ACC,outline=(255,232,213),width=2)
  for v,x in enumerate([155,655]):im.paste(mini[v],(x,925),mini[v])
  text(188,1298,'FRONT OBLIQUE',18,MUTED);text(765,1298,'SIDE',18,MUTED)
  distance_label='FIRST CONTACT'if contact_episode else'DISTANCE'
  d.line((45,1343,1035,1343),fill=(45,63,80));stats=[(distance_label,f'{distance:.2f} m'if contact_shown else '--'),('H / V SPEED',f'{-velocity[1]:.2f} / {velocity[2]:.2f}'),('ANGLE',f'{angle:.1f} deg')]
  for x,(name,value)in zip([45,405,835],stats):text(x,1374,name,19,MUTED);text(x,1411,value,30,ACC if name==distance_label else FG)
  text(405,1452,'m/s',18,MUTED);text(45,1490,'178 CM / 87 KG',18,MUTED);text(715,1490,'COM',18,ACC);assert np.all(np.asarray(im)[1536:]==BG);return im
 frame((takeoff+landing)/2,.6).save(out/'fixed_side_cover.png');frame(T,1,hold_s).save(out/'fixed_side_ending.png')
 if a.preview:return
 video=out/'fixed_side_multiview_review.mp4';writer=imageio_ffmpeg.write_frames(str(video),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None);count=0
 try:
  for speed in [1.,.6]:
   for k in range(round((T/speed+hold_s)*30)):
    clock=k/30;writer.send(np.asarray(frame(min(T,clock*speed),speed,max(0.,clock-T/speed))));count+=1
 finally:writer.close()
 metrics=dict(video=video.name,frames=count,duration_s=count/30,main_camera_tracks_body=False,main_view='True world-fixed side view; immutable world scale, ground and takeoff line',insets='Same native poses in tracked front-oblique and side views',distance_m=distance,distance_metric='Nearest native sole-floor contact from touchdown through recovery when stable; first-heel projection fallback for an invalid candidate. Foot-contact control proxy, not official full-body test scoring.',takeoff_horizontal_m_s=-velocity[1],takeoff_vertical_m_s=velocity[2],takeoff_angle_deg=angle,instantaneous_force_colours=False,bottom_reserved_px=384,body_motion_changed_by_renderer=False)
 if contact_episode:metrics.update(distance_metric='Heel projection at first native foot contact; recovery is not shown or certified.',first_contact_freeze_seconds_per_replay=hold_s,contact_pose_held_only_in_display=True)
 (out/'fixed_side_metrics.json').write_text(json.dumps(metrics,indent=2));(out/'fixed_side_index.html').write_text('<meta charset="utf-8"><style>body{background:#0d1726;color:#edf3fb;font:18px system-ui;max-width:900px;margin:30px auto}video{max-height:85vh;max-width:100%}</style><video controls playsinline src="'+video.name+'" poster="fixed_side_cover.png"></video><p>World-fixed side view with tracked front-oblique and side insets. Neutral anatomy; no muscle-force colour field.</p>',encoding='utf-8');print('FIXED_SIDE_MULTIVIEW_READY',video,flush=True)
if __name__=='__main__':main()
