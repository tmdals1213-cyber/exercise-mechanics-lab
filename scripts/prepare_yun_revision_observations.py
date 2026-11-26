"""Mask unreliable observations, never replace them with a desired movement."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 folder=ROOT/'results/reference_box_jump/yun_user_video/rtmpose_motion_v2';raw=dict(np.load(folder/'tracked_landmarks.npz'));p=raw['image_landmarks_xy_z_visibility_presence'].copy();ts=raw['source_times_s'];required=list(raw['mapped_mediapipe_indices']);mask=np.zeros(p.shape[:2],bool);notes=[]
 # Reject short-lived whole-person identity switches before per-joint checks.
 hips=p[:,[23,24],:2].mean(1)
 for i in range(3,len(ts)-3):
  median=np.median(hips[i-3:i+4],axis=0)
  if np.linalg.norm(hips[i]-median)>35:
   mask[i,required]=True;notes.append(dict(time_s=float(ts[i]),reason='whole-person hip trajectory outlier',distance_px=float(np.linalg.norm(hips[i]-median))))
 for j in required:
  for i in range(3,len(ts)-3):
   median=np.median(p[i-3:i+4,j,:2],axis=0)
   if np.linalg.norm(p[i,j,:2]-median)>35:mask[i,j]=True
  mask[:,j]|=p[:,j,3]<.20
 for i in range(len(ts)):
  for ankle,heel,toe in [(27,29,31),(28,30,32)]:
   if np.linalg.norm(p[i,heel,:2]-p[i,ankle,:2])>40:mask[i,heel]=True
   if np.linalg.norm(p[i,toe,:2]-p[i,ankle,:2])>70:mask[i,toe]=True
 original=p.copy();p[mask,:2]=np.nan;p[mask,3:]=0;raw.update(image_landmarks_xy_z_visibility_presence=p,raw_image_landmarks=original,rejected_observation_mask=mask);np.savez_compressed(folder/'motion_observations.npz',**raw)
 report=dict(status='masked_observation_candidate',source_frames=len(ts),rejected_marker_samples=int(mask.sum()),whole_person_outliers=notes,method='Seven-frame image-space median outlier test (35 px), low confidence (<0.20), and gross foot marker separation. Raw observations preserved; masked samples do not contribute to pose fitting.',force_analysis_allowed=False)
 (folder/'observation_review.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 cfg=json.loads((ROOT/'scenarios/yun_box_rtmpose_kinematic_reference.json').read_text());cfg.update(
  tracking_file='results/reference_box_jump/yun_user_video/rtmpose_motion_v2/stabilized_observations.npz',interval_s=[158.,159.8],
  initial_time_warp=[[158.,158.],[159.33,159.33],[159.8,159.33]],
  frozen_coordinates=dict(shoulder_rot_r=0.,shoulder_rot_l=0.),arm_sagittal_sigma_m=.03,arm_cartesian_acceleration_weight=.025,
  flight_COM_interval_s=[158.75,159.09],flight_COM_acceleration_sigma_m_s2=[.3,.3,.8],
  ground_anchor_start_s=158.14,contact_anchor_sigma_m=.005,contact_z_sigma_m=.005,
  root_translation_upper_m=[2.5,2.],
  camera_motion_removed=True,assumed_physical_playback_rate=1.,physical_playback_rate_verified=False,
  assumption_notes=['Axial humeral rotation held at anatomical reference; sagittal arm swing remains free.','Cartesian arm trajectory is regularized rather than penalizing plane-of-elevation Euler changes.','Whole-body COM, not pelvis, follows near-zero horizontal acceleration and gravity over the conservative airborne interval.','Foot anchors apply after the final foot plant; arrival and standing are retained.'])
 (ROOT/'scenarios/yun_box_motion_v2.json').write_text(json.dumps(cfg,indent=2))
if __name__=='__main__':main()
