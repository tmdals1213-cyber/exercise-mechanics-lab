"""Preserve raw detections and mask visually identified spectator-leg outliers."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 folder=ROOT/'results/reference_box_jump/yun_user_video/rtmpose_full_body';z=dict(np.load(folder/'tracked_landmarks.npz'));p=z['image_landmarks_xy_z_visibility_presence'].copy();ts=z['source_times_s'];removed=[]
 # In the fully tucked phase the foreground white shoes are above image y=430.
 # Detections on the stationary spectator's shins below that line are not the
 # athlete. The raw estimator data remains unchanged in tracked_landmarks.npz.
 for i,t in enumerate(ts):
  if t<158.90:continue
  for ankle,heel,toe in [(27,29,31),(28,30,32)]:
   if p[i,ankle,1]>430:
    for j in [ankle,heel,toe]:removed.append(dict(source_time_s=float(t),mediapipe_compatibility_index=j,raw_xy=p[i,j,:2].tolist()));p[i,j,:2]=np.nan;p[i,j,3:]=0
 z['image_landmarks_xy_z_visibility_presence']=p;np.savez_compressed(folder/'motion_observations.npz',**z)
 note=dict(status='reviewed_tracking_with_explicit_occlusion_masks',source='https://www.youtube.com/watch?v=ngyj9rSEjLc&t=155s',mask_type='Manual image-region exclusion after visual review, not a force or joint-angle cap.',removed_observations=removed,raw_data_preserved=True,physical_time_verified=False,force_analysis_allowed=False)
 (folder/'observation_review.json').write_text(json.dumps(note,indent=2))
 cfg=json.loads((ROOT/'scenarios/yun_box_video_kinematic_reference.json').read_text());cfg.update(tracking_file='results/reference_box_jump/yun_user_video/rtmpose_full_body/motion_observations.npz',
  initial_time_warp=[[157.52,157.52],[159.33,159.33]],trunk_rotation_bounds_deg=dict(lat_bending=5,axial_rotation=10,flex_extension=15),
  trunk_bound_scope='Explicit near-sagittal, small spinal deformation assumption from visual reference review; global torso lean through pelvis/hip motion remains free.')
 (ROOT/'scenarios/yun_box_rtmpose_kinematic_reference.json').write_text(json.dumps(cfg,indent=2))
 print(json.dumps(dict(removed_observations=len(removed),source_frames=len(ts))))
if __name__=='__main__':main()
