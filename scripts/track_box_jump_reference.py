"""Track an actual demonstration as a kinematic reference, without estimating forces.

Image observations and monocular estimated 3D are stored separately. The latter
is not calibrated 3D motion capture. Physical timing and geometry need review.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,mediapipe as mp,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
LINKS=[(11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32)]
def angle(a,b,c):
 u=a-b;v=c-b;den=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
 return np.degrees(np.arccos(np.clip((u*v).sum(-1)/np.maximum(den,1e-8),-1,1)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=float,default=9.0);p.add_argument('--end',type=float,default=17.0);a=p.parse_args()
 source=ROOT/'datasets/box_jump_reference/catalyst';out=ROOT/'results/reference_box_jump/catalyst_tracking';out.mkdir(parents=True,exist_ok=True)
 reader=imageio_ffmpeg.read_frames(str(source/'source.webm'));meta=next(reader);fps=float(meta['fps']);w,h=meta['size'];times=[];obs=[];world=[];qc=[]
 options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'.tools/pose_reference/pose_landmarker_full.task')),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1,min_pose_detection_confidence=.6,min_tracking_confidence=.6)
 with mp.tasks.vision.PoseLandmarker.create_from_options(options)as tracker:
  for k,raw in enumerate(reader):
   t=k/fps
   if t<a.start:continue
   if t>a.end:break
   pixels=np.frombuffer(raw,np.uint8).reshape(h,w,3);left,top,right,bottom=120,50,min(700,w),min(470,h)
   crop=np.ascontiguousarray(pixels[top:bottom,left:right]);result=tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop),round(t*1000))
   points=np.full((33,5),np.nan);wp=np.full((33,3),np.nan)
   if result.pose_landmarks:
    for i,lm in enumerate(result.pose_landmarks[0]):points[i]=[left+lm.x*(right-left),top+lm.y*(bottom-top),lm.z,lm.visibility,lm.presence]
    for i,lm in enumerate(result.pose_world_landmarks[0]):wp[i]=[lm.x,lm.y,lm.z]
   times.append(t);obs.append(points);world.append(wp)
   if len(times)%8==1:qc.append((t,pixels.copy(),points.copy()))
   if len(times)%30==0:print('REFERENCE_TRACKING',len(times),round(t,2),flush=True)
 reader.close();times=np.array(times);obs=np.array(obs);world=np.array(world)
 np.savez_compressed(out/'landmarks.npz',source_times_s=times,relative_times_s=times-times[0],image_landmarks_xy_z_visibility_presence=obs,monocular_estimated_world_landmarks=world)
 measures={}
 for side,hip,knee,ankle,shoulder,elbow,wrist in [('L',23,25,27,11,13,15),('R',24,26,28,12,14,16)]:
  measures['knee_flexion_2D_'+side]=180-angle(obs[:,hip,:2],obs[:,knee,:2],obs[:,ankle,:2])
  measures['elbow_flexion_2D_'+side]=180-angle(obs[:,shoulder,:2],obs[:,elbow,:2],obs[:,wrist,:2])
 np.savez_compressed(out/'image_angle_proxies.npz',source_times_s=times,**measures)
 W,H=1000,720;sheet=Image.new('RGB',(W,H),(18,25,35));draw=ImageDraw.Draw(sheet)
 for cell,index in enumerate(np.linspace(0,len(qc)-1,12).astype(int)):
  t,pixels,points=qc[index];im=Image.fromarray(pixels);d=ImageDraw.Draw(im)
  for i,j in LINKS:
   if np.isfinite(points[[i,j],:2]).all()and points[[i,j],3].min()>.5:d.line([tuple(points[i,:2]),tuple(points[j,:2])],fill=(30,245,190),width=3)
  im=im.resize((250,210));x=(cell%4)*250;y=(cell//4)*240;sheet.paste(im,(x,y+25));draw.text((x+5,y+5),f'{t:.2f} s',fill='white')
 sheet.save(out/'tracking_contact_sheet.png')
 frames=round((times[-1]-times[0])*fps)+1;writer=imageio_ffmpeg.write_frames(str(out/'image_space_motion_review.mp4'),(854,480),fps=fps,codec='libx264',quality=8,macro_block_size=2);writer.send(None)
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for t,points in zip(times,obs):
  im=Image.new('RGB',(854,480),(13,23,38));d=ImageDraw.Draw(im);d.text((25,20),'REFERENCE MOTION / IMAGE SPACE',font=font,fill='white');d.text((25,447),'Monocular tracking draft / No force calculation',font=font,fill=(165,185,205))
  for i,j in LINKS:
   if np.isfinite(points[[i,j],:2]).all():
    good=points[[i,j],3].min()>.5;d.line([tuple(points[i,:2]),tuple(points[j,:2])],fill=(75,200,195)if good else(90,90,95),width=4)
  writer.send(np.asarray(im))
 writer.close()
 missing=np.isnan(obs[:,:,0]).all(1);lower=obs[:,[23,24,25,26,27,28,29,30,31,32],3]
 report=dict(status='raw_reference_tracking_pending_visual_and_geometry_review',source_url='https://www.youtube.com/watch?v=d2z2_rRkpAo',source_page='https://www.catalystathletics.com/exercise/548/Box-Jump/',source_title='Box Jump | Olympic Weightlifting Exercise Library',source_video_SHA256=hashlib.sha256((source/'source.webm').read_bytes()).hexdigest(),source_interval_s=[float(times[0]),float(times[-1])],container_fps=fps,physical_playback_rate_verified=False,frames=len(times),missing_detection_frames=int(missing.sum()),lower_limb_min_visibility=float(np.nanmin(lower)),low_visibility_lower_limb_fraction=float(np.mean(lower<.5)),camera_calibrated=False,monocular_3D_is_measured_ground_truth=False,source_box_height_m=None,force_analysis_allowed=False,native_model_retarget_complete=False,notes=['Track actual pose and phase before any dynamics correction.','Review occluded ankles, feet and bilateral identity before accepting a reference.','Single-view angle traces are projection-dependent proxies, not validated anatomical joint angles.','The physical playback rate, metric geometry and contact events must be reviewed before quantitative inverse dynamics.'])
 (out/'reference_manifest.json').write_text(json.dumps(report,indent=2));print('REFERENCE_TRACKING_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
