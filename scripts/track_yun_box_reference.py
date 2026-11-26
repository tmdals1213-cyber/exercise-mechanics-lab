"""Track the selected athlete in the user's full-body broadcast replay.

Image-space observations only. Broadcast replay timing is not physical time.
The supplied identity anchors guide selection among background spectators.
"""
from pathlib import Path
import json
import numpy as np,mediapipe as mp,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
LINKS=[(11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32)]
def main():
 out=ROOT/'results/reference_box_jump/yun_user_video/tracking_roi_v2';out.mkdir(exist_ok=True);source=ROOT/'datasets/box_jump_reference/yun_user_video/source.webm'
 reader=imageio_ffmpeg.read_frames(str(source));meta=next(reader);w,h=meta['size'];fps=float(meta['fps']);times=[];records=[];qc=[];scores=[];candidates=[]
 anchors=np.array([[157.5,580,390],[158.,600,390],[158.35,635,470],[158.5,655,480],[158.65,675,380],[158.8,710,305],[158.95,750,300],[159.15,775,315],[159.35,790,280]])
 options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'.tools/pose_reference/pose_landmarker_full.task')),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=4,min_pose_detection_confidence=.4)
 with mp.tasks.vision.PoseLandmarker.create_from_options(options)as tracker:
  for k,raw in enumerate(reader):
   t=k/fps
   if t<157.5:continue
   if t>159.35:break
   pixels=np.frombuffer(raw,np.uint8).reshape(h,w,3);want=np.array([np.interp(t,anchors[:,0],anchors[:,j])for j in [1,2]])
   left,top,right,bottom=max(0,round(want[0]-125)),max(0,round(want[1]-180)),min(w,round(want[0]+125)),min(h,round(want[1]+180))
   crop=np.ascontiguousarray(pixels[top:bottom,left:right]);result=tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop));allpoints=[]
   for pose in result.pose_landmarks:allpoints.append(np.array([[left+lm.x*(right-left),top+lm.y*(bottom-top),lm.z,lm.visibility,lm.presence]for lm in pose]))
   distances=[np.linalg.norm(p[[23,24],:2].mean(0)-want)for p in allpoints];selected=int(np.argmin(distances))if distances else None
   points=allpoints[selected]if selected is not None and min(distances)<90 else np.full((33,5),np.nan)
   times.append(t);records.append(points);scores.append(float(min(distances))if distances else None);candidates.append(len(allpoints))
   if len(times)%4==1:qc.append((t,pixels.copy(),points.copy()))
 reader.close();times=np.array(times);points=np.array(records)
 np.savez_compressed(out/'tracked_landmarks.npz',source_times_s=times,image_landmarks_xy_z_visibility_presence=points,identity_selection_distance_px=np.array([x if x is not None else np.nan for x in scores]),candidate_counts=np.array(candidates))
 cols=4;tw,th=400,330;sheet=Image.new('RGB',(cols*tw,((len(qc)+cols-1)//cols)*th),(13,23,38));font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);d=ImageDraw.Draw(sheet)
 for cell,(t,pixels,p)in enumerate(qc):
  im=Image.fromarray(pixels);dr=ImageDraw.Draw(im)
  for a,b in LINKS:
   if np.isfinite(p[[a,b],:2]).all():dr.line([tuple(p[a,:2]),tuple(p[b,:2])],fill=(0,245,220),width=3)
  im=im.crop((440,130,1060,700)).resize((tw,300));x=(cell%cols)*tw;y=(cell//cols)*th;sheet.paste(im,(x,y+30));d.text((x+8,y+4),f'{t:.3f} s',font=font,fill='white')
 sheet.save(out/'athlete_tracking_review.png')
 shoulder=points[:,[11,12],:2].mean(1);hip=points[:,[23,24],:2].mean(1);axis=shoulder-hip;inclination=np.degrees(np.arctan2(axis[:,0],-axis[:,1]))
 np.savez_compressed(out/'trunk_image_proxy.npz',source_times_s=times,trunk_inclination_from_image_vertical_deg=inclination,shoulder_midpoint_px=shoulder,hip_midpoint_px=hip)
 report=dict(status='raw_athlete_identity_and_joint_tracking_review_pending',source='https://www.youtube.com/watch?v=ngyj9rSEjLc&t=155s',interval_s=[float(times[0]),float(times[-1])],frames=len(times),missing_frames=int(np.isnan(points[:,:,0]).all(1).sum()),identity_anchors_are_manual=True,identity_rejection_threshold_px=90,identity_distance_max_px=max(x for x in scores if x is not None),
  image_angles_are_anatomical_angles=False,physical_replay_speed_verified=False,force_analysis_allowed=False,notes=['Review identity against nearby spectators and occluded hips/knees before accepting traces.','This wide shot replays the jump shown earlier; it is not a second jump or a calibrated synchronized camera.','A torso midpoint axis is an image-space inclination proxy, not spinal flexion.'])
 (out/'tracking_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
