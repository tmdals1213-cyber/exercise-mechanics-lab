"""Remove measured background camera motion before fitting athlete movement.

Uses static-field features, excludes the moving athlete and broadcast graphics,
and preserves all original keypoints and the per-frame homography.
"""
from pathlib import Path
import json,argparse
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1]
def warp(points,H):
 p=np.asarray(points);hom=np.c_[p.reshape(-1,2),np.ones(p.size//2)]@H.T
 return(hom[:,:2]/hom[:,2,None]).reshape(p.shape)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--folder',default='results/reference_box_jump/yun_user_video/rtmpose_full_body');args=parser.parse_args()
 cv2.setNumThreads(2);folder=ROOT/args.folder;raw=dict(np.load(folder/'motion_observations.npz'));ts=raw['source_times_s'];boxes=raw['manual_bounding_boxes_px'];cap=cv2.VideoCapture(str(ROOT/'datasets/box_jump_reference/yun_user_video/source.webm'));fps=cap.get(cv2.CAP_PROP_FPS);cap.set(cv2.CAP_PROP_POS_FRAMES,round(ts[0]*fps));Hs=[];qc=[];errors=[];counts=[];reference=None;tracked=None;previous=None;H=np.eye(3)
 for i,t in enumerate(ts):
  ok,img=cap.read();assert ok;gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);mask=np.zeros_like(gray);mask[100:235,20:1250]=255;mask[:125,:475]=0;mask[:150,1170:]=0
  x0,y0,x1,y1=boxes[i];mask[max(0,int(y0)-25):min(gray.shape[0],int(y1)+25),max(0,int(x0)-25):min(gray.shape[1],int(x1)+25)]=0
  if i==0:
   tracked=cv2.goodFeaturesToTrack(gray,maxCorners=800,qualityLevel=.008,minDistance=7,mask=mask).reshape(-1,2);reference=tracked.copy()
  else:
   new,status,err=cv2.calcOpticalFlowPyrLK(previous,gray,tracked.astype(np.float32).reshape(-1,1,2),None,winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));new=new.reshape(-1,2);valid=(status.ravel()>0)&(err.ravel()<25)&(new[:,0]>=0)&(new[:,0]<gray.shape[1])&(new[:,1]>=0)&(new[:,1]<gray.shape[0]);xy=new.astype(int);valid&=mask[np.clip(xy[:,1],0,gray.shape[0]-1),np.clip(xy[:,0],0,gray.shape[1]-1)]>0
   reference=reference[valid];tracked=new[valid];assert len(tracked)>20,('Background tracking lost',i,len(tracked))
   H,inliers=cv2.findHomography(reference,tracked,cv2.RANSAC,1.5);assert H is not None
   good=inliers.ravel().astype(bool);reference=reference[good];tracked=tracked[good]
  inverse=np.linalg.inv(H);residual=warp(tracked,inverse)-reference;errors.append(float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))));counts.append(len(tracked));Hs.append(inverse)
  if i in [0,15,30,40,54]:qc.append((t,img.copy(),H.copy()))
  previous=gray
 cap.release();Hs=np.array(Hs);original=raw['image_landmarks_xy_z_visibility_presence'];points=original.copy()
 for i,H in enumerate(Hs):points[i,:,:2]=warp(points[i,:,:2],H)
 raw.update(image_landmarks_xy_z_visibility_presence=points,unstabilized_image_landmarks=original,image_to_reference_H=Hs);np.savez_compressed(folder/'stabilized_observations.npz',**raw)
 center=np.array([[640.,360.]]);camera_shift=np.array([warp(center,np.linalg.inv(H))[0]-center[0]for H in Hs]);np.savez_compressed(folder/'camera_stabilization.npz',source_times_s=ts,image_to_reference_H=Hs,camera_image_center_shift_px=camera_shift,background_fit_RMS_px=errors,inliers=counts)
 report=dict(status='background_camera_motion_estimated',source='https://www.youtube.com/watch?v=ngyj9rSEjLc&t=155s',method='Long-track optical flow and RANSAC projective transform from static field features; athlete and broadcast graphics excluded.',
  start_to_end_camera_shift_at_image_center_px=camera_shift[-1].tolist(),maximum_background_fit_RMS_px=max(errors),minimum_inlier_tracks=min(counts),
  homography_is_metric_camera_calibration=False,source_observations_preserved=True,force_analysis_allowed=False)
 (folder/'camera_stabilization.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,7));axes[0,0].plot(ts,camera_shift);axes[0,0].set_title('Measured camera-induced image-centre shift');axes[0,0].legend(['x','y']);axes[0,0].set_ylabel('Pixels')
 for side,j,c in [('Left',27,'#258497'),('Right',28,'#be822f')]:
  axes[0,1].plot(ts,original[:,j,0],color=c,ls='--',label=side+' raw');axes[0,1].plot(ts,points[:,j,0],color=c,label=side+' stabilized')
 axes[0,1].set_title('Ankle image x');axes[0,1].legend(fontsize=8)
 for axis,label in [(0,'x'),(1,'y')]:
  hip=points[:,[23,24],axis].mean(1);axes[1,0].plot(ts,hip,label='Hip '+label)
 axes[1,0].set_title('Stabilized hip midpoint (image-space only)');axes[1,0].legend();axes[1,1].plot(ts,errors);axes[1,1].set_title('Static-background reprojection RMS (px)')
 for ax in axes.ravel():ax.grid(alpha=.25);ax.set_xlabel('Source time (s)')
 fig.tight_layout();fig.savefig(folder/'camera_motion_check.png',dpi=130);plt.close(fig)
if __name__=='__main__':main()
