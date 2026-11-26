"""Inspect native reprojection against source frames before dynamics or rendering."""
from pathlib import Path
import json,argparse
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/reference_box_jump/temporal_image_fit'
def main():
 global OUT
 ap=argparse.ArgumentParser();ap.add_argument('--folder');ap.add_argument('--video',default='datasets/box_jump_reference/catalyst/source.webm');ap.add_argument('--times');ap.add_argument('--crop',default='100,35,740,470');args=ap.parse_args()
 if args.folder:OUT=ROOT/args.folder
 data=np.load(OUT/'motion.npz');times=data['source_times_s'];fit=data['fit_image_points_px'];obs=data['reference_image_points_px'];names=list(data['marker_names'])
 valid=data['observed_marker_mask']if'observed_marker_mask'in data else np.ones(obs.shape[:2],bool);display_fit=fit.copy();display_obs=obs.copy()
 if'image_to_reference_H'in data:
  for i,H in enumerate(data['image_to_reference_H']):
   inverse=np.linalg.inv(H)
   for source,target in [(fit,display_fit),(obs,display_obs)]:
    hom=np.c_[source[i],np.ones(len(names))]@inverse.T;target[i]=hom[:,:2]/hom[:,2,None]
 links=[]
 for side in ['l','r']:
  for a,b in [('shoulder','hip'),('shoulder','elbow'),('elbow','wrist'),('hip','knee'),('knee','ankle'),('ankle','heel'),('heel','toe'),('ankle','toe')]:links.append((names.index(a+'_'+side),names.index(b+'_'+side)))
 for key in ['hip','shoulder']:links.append((names.index(key+'_l'),names.index(key+'_r')))
 requested=[float(x)for x in args.times.split(',')]if args.times else [12.4,12.733,12.867,12.967,13.133,13.267,13.5,14.1]
 picks=[int(np.argmin(abs(times-t)))for t in requested];crop=tuple(int(x)for x in args.crop.split(','))
 frames={}
 for i in sorted(set(picks)):
  reader=imageio_ffmpeg.read_frames(str(ROOT/args.video),input_params=['-ss',f'{times[i]:.9f}']);meta=next(reader);w,h=meta['size'];frames[i]=Image.fromarray(np.frombuffer(next(reader),np.uint8).reshape(h,w,3).copy());reader.close()
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
 sheet=Image.new('RGB',(1200,4*390+65),(13,20,30));draw=ImageDraw.Draw(sheet)
 draw.text((20,10),'Observed joints: orange | Native model reprojection: cyan | Not calibrated 3D',font=font,fill='white')
 for cell,i in enumerate(picks):
  im=frames[i].copy();d=ImageDraw.Draw(im)
  for a,b in links:d.line([tuple(display_fit[i,a]),tuple(display_fit[i,b])],fill=(0,240,235),width=3)
  for j,p in enumerate(display_obs[i]):
   if valid[i,j]:d.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=(255,135,25))
  im=ImageOps.contain(im.crop(crop),(570,355));x=(cell%2)*600+15;y=(cell//2)*390+55;sheet.paste(im,(x+(570-im.width)//2,y+25+(355-im.height)//2));draw.text((x,y),f'Source {times[i]:.3f} s',font=font,fill='white')
 sheet.save(OUT/'reprojection_review.png')
 errors=np.linalg.norm(fit-obs,axis=2);report=json.loads((OUT/'tracking_report.json').read_text())
 report['reprojection_by_marker_RMS_px']={n:float(np.sqrt(np.mean(errors[valid[:,i],i]**2)))for i,n in enumerate(names)}
 report['pixel_point_RMS']=float(np.sqrt(np.mean(errors[valid]**2)));report['pixel_component_RMS']=report['pixel_point_RMS']/np.sqrt(2)
 report['pixel_point_error_95th_percentile']=float(np.percentile(errors[valid],95));report['pixel_point_error_max']=float(errors[valid].max());report['rejected_samples_excluded_from_metrics']=int((~valid).sum())
 report['review_images']=['reprojection_review.png','tracking_trajectories.png'];report['physiology_in_scope']=False
 (OUT/'tracking_report.json').write_text(json.dumps(report,indent=2))
 # Position and source/fitted 2D angle checks disclose temporal drift.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True)
 for row,key in enumerate(['hip','wrist','ankle']):
  for col,axis in enumerate([0,1]):
   ax=axes[row,col]
   for side,color in [('l','#008899'),('r','#c76416')]:
    j=names.index(key+'_'+side);observed=obs[:,j,axis].copy();observed[~valid[:,j]]=np.nan;ax.plot(times,observed,color=color,alpha=.45,lw=3,label='Observed '+side);ax.plot(times,fit[:,j,axis],color=color,lw=1,label='Native '+side)
   ax.set_ylabel(key+' '+('x'if axis==0 else'y')+' (px)');ax.grid(alpha=.25)
 axes[0,0].legend(fontsize=8,ncol=2);axes[-1,0].set_xlabel('Source time (s)');axes[-1,1].set_xlabel('Source time (s)')
 fig.suptitle('Sequence-level image tracking | Pixel agreement does not establish 3D or dynamic accuracy');fig.tight_layout();fig.savefig(OUT/'tracking_trajectories.png',dpi=140);plt.close(fig)
 print(json.dumps(dict(pixel_point_RMS=report['pixel_point_RMS'],p95=report['pixel_point_error_95th_percentile'],maximum=report['pixel_point_error_max'],force_analysis_allowed=False)))
if __name__=='__main__':main()
