"""Portrait kinematics review; no muscle loads, physics claims or source footage."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];CASE=ROOT/'results/reference_box_jump/catalyst_native_replay';OUT=ROOT/'deliverables/reference_motion_review'
def main():
 global CASE,OUT
 parser=argparse.ArgumentParser();parser.add_argument('--folder');parser.add_argument('--output-dir');parser.add_argument('--render-dir',default='study_render');parser.add_argument('--source-name',default='Catalyst Athletics');parser.add_argument('--title',default='BOX JUMP');parser.add_argument('--label',default='KINEMATICS STUDY');parser.add_argument('--footer');parser.add_argument('--speed',type=float,default=.5);parser.add_argument('--cycles',type=int,default=2);args=parser.parse_args()
 if args.folder:CASE=ROOT/args.folder
 if args.output_dir:OUT=ROOT/args.output_dir
 OUT.mkdir(parents=True,exist_ok=True);meta=json.loads((CASE/'summary.json').read_text());assert meta['force_analysis_allowed']is False
 subject=json.loads((CASE/'setup.json').read_text()) if meta.get('measured_replay') else {}
 files=sorted((CASE/args.render_dir).glob('frame_*.png'));expected=round(meta['phase_times_s']['end']*30)+1;assert len(files)==expected,(len(files),expected);bounds=[[],[]]
 for f in files:
  pixels=np.asarray(Image.open(f));a=pixels[:,:,3]
  for view,(left,right)in enumerate([(0,800),(800,1600)]):
   yy,xx=np.where(a[:,left:right]>8);assert len(xx)>0 and yy.min()>2 and yy.max()<857;bounds[view].append((xx.min()+left,yy.min(),xx.max()+left+1,yy.max()+1))
   if meta.get('measured_replay'):
    body=(a[:,left:right]>8)&(pixels[:,left:right,0]>195)&(abs(pixels[:,left:right,0].astype(float)-pixels[:,left:right,1])<10)
    yy,xx=np.where(body);assert len(xx)>0
    bounds[view][-1]=(max(left,int(xx.min()+left)-20),max(0,int(yy.min())-10),min(right,int(xx.max()+left)+21),min(860,int(yy.max())+31))
 crops=[];sizes=[]
 for view in range(2):
  b=np.array(bounds[view]);crop=(max(view*800,int(b[:,0].min())-5),max(0,int(b[:,1].min())-5),min((view+1)*800,int(b[:,2].max())+5),min(860,int(b[:,3].max())+5));crops.append(crop);ratio=min([530,460][view]/(crop[2]-crop[0]),900/(crop[3]-crop[1]));sizes.append((round((crop[2]-crop[0])*ratio),round((crop[3]-crop[1])*ratio)))
 images=[]
 for f in files:
  im=Image.open(f).convert('RGBA');images.append([im.crop(c).resize(s,Image.Resampling.LANCZOS)for c,s in zip(crops,sizes)])
 fps=30;speed=args.speed;duration=args.cycles*meta['phase_times_s']['end']/speed;count=round(duration*fps);cycle=meta['phase_times_s']['end']/speed
 video_name='jump_motion_review.mp4' if meta.get('measured_replay') else 'box_jump_motion_replay.mp4'
 writer=imageio_ffmpeg.write_frames(str(OUT/video_name),(1080,1920),fps=fps,codec='libx264',quality=8,macro_block_size=2,output_params=['-pix_fmt','yuv420p','-movflags','+faststart']);writer.send(None)
 def font(n,b=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf'if b else'C:/Windows/Fonts/arial.ttf',n)
 for k in range(count):
  source=(k/fps%cycle)*speed;i=min(len(images)-1,round(source*30));im=Image.new('RGB',(1080,1920),(13,23,38));d=ImageDraw.Draw(im)
  d.text((55,75),args.title,font=font(64,True),fill=(237,243,251));d.text((850,170),f'{speed:g}x',font=font(28),fill=(166,188,209))
  if meta.get('measured_replay'):d.text((58,170),f"{subject['height_m']*100:g} cm / {subject['mass_kg']:g} kg",font=font(28),fill=(166,188,209))
  for view,(image,size)in enumerate(zip(images[i],sizes)):im.paste(image,([290,800][view]-size[0]//2,285+(900-size[1])//2),image)
  phase=args.label
  if meta.get('measured_replay') or meta.get('landing_simulated'):
   p=meta['phase_times_s'];phase='PREPARATION' if source<p['bottom'] else 'TAKEOFF' if source<p['end_ascent'] else 'FLIGHT' if source<p['landing'] else 'LANDING' if source<p['landing']+.35 else 'STABILIZATION'
  d=ImageDraw.Draw(im);d.text((60,1320),phase,font=font(30,True),fill=(237,243,251));d.text((60,1498),args.footer if args.footer is not None else 'Motion reference: '+args.source_name,font=font(22),fill=(166,188,209))
  if k in [0,count//4,count//2]:im.save(OUT/f'review_{k:04d}.png')
  if meta.get('landing_simulated') and k==round(meta['phase_times_s']['landing']/speed*fps):im.save(OUT/'poster.png')
  writer.send(np.asarray(im))
 tracking=json.loads((CASE/'tracking_report.json').read_text())if(CASE/'tracking_report.json').exists()else{}
 writer.close();report=dict(status='kinematic_retarget_review_only',source_motion_sha256=hashlib.sha256((CASE/'motion.npz').read_bytes()).hexdigest(),frames=count,fps=fps,duration_s=count/fps,force_analysis_allowed=False,source=tracking.get('source','https://www.catalystathletics.com/exercise/548/Box-Jump/'),platform_is_illustrative=True,physical_playback_rate_verified=False,camera_calibrated=False)
 if meta.get('physical_task'):
  report.update(status='physical_task_precontact_review',source=None,video_tracking_in_objective=False,simulation_time_known=True,physical_playback_rate_verified=True,source_camera_calibration_required=False,landing_simulated=False)
  if meta.get('landing_simulated'):
   audit=json.loads((CASE/'cycle_audit.json').read_text());report.update(status='full_cycle_motion_review_unvalidated_loads',platform_is_illustrative=False,landing_simulated=True,model_box_height_m=audit['box_height_m'],independent_dynamics_pass=audit['independent_dynamics_pass'],load_prediction_validated=False,athlete_personal_performance_identified=False)
 if meta.get('measured_replay'):
  report.update(status='measured_full_cycle_motion_review',platform_is_illustrative=False,physical_playback_rate_verified=True,landing_included=bool(meta.get('landing_included')),landing_simulated=False,load_prediction_validated=False)
 (OUT/'review_manifest.json').write_text(json.dumps(report,indent=2));print('REFERENCE_REVIEW_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()

