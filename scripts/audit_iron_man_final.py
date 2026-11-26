from pathlib import Path
import json,numpy as np,imageio_ffmpeg,hashlib
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_comparison_final'

def main():
 met=json.loads((OUT/'metrics.json').read_text());checks={};cases={}
 for key in ['jump','squat']:
  folder=OUT/'render_cases'/key;z=dict(np.load(folder/'analysis/tensions.npz'));v=json.loads((folder/'analysis/visualization/data.json').read_text());s=json.loads((folder/'analysis/tension_summary.json').read_text());motion=dict(np.load(folder/'motion.npz'));cam=json.loads((folder/'matched_camera.json').read_text());files=sorted((folder/'study_render').glob('frame_*.png'));expected=round(z['times'][-1]*60)+1
  assert len(files)==expected and len(cam['frames'])==expected,(key,len(files),expected,len(cam['frames']))
  for i,f in enumerate(files):
   assert f.name==f'frame_{i:04d}.png'
   with Image.open(f) as im:assert im.size==(1600,860);im.verify()
  assert cam['ortho_width_m']==5.6 and cam['zoom_is_pose_independent'];assert 'display_valid_window_s' not in v;assert z['times'][0]==motion['times'][0] and abs(z['times'][-1]-motion['times'][-1])<1e-6;assert np.isfinite(z['force_N']).all() and z['force_N'].min()>-1e-7
  assert hashlib.sha256((folder/'analysis/tensions.npz').read_bytes()).hexdigest()==met['cases'][key]['force_sha256'];assert s['checks']['status']=='passed'
  # Project anatomical landmarks through the actual orthographic camera.
  # Skeleton landmarks, not a claim of full surface-collision validation.
  names=json.loads((folder/'summary.json').read_text())['body_names'];ids=[names.index(n+'_'+side) for side in ['r','l'] for n in ['lunate','humerus','ulna','calcn','femur','tibia']]+[names.index('head')];yaw=np.deg2rad(25);el=np.deg2rad(15);right=np.array([np.cos(yaw),np.sin(yaw),0]);up=np.array([-np.sin(yaw)*np.sin(el),np.cos(yaw)*np.sin(el),np.cos(el)]);a=np.deg2rad(115);R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]);bounds=[[],[]]
  for rec in cam['frames']:
   i=int(abs(motion['times']-rec['time_s']).argmin());p=motion['body_xpos'][i,ids];p=np.r_[p,motion['contact_positions_m'][i],(motion['body_xpos'][i,names.index('head')]+np.array([0,0,.13]))[None]];center=np.array(rec['body_center_m'])
   for view in range(2):
    delta=(p-center) if view==0 else (p-center)@R.T;xx=800+1600/5.6*(delta@right+(-1.3 if view==0 else 1.3));yy=430-1600/5.6*(delta@up);bounds[view].append([xx.min(),yy.min(),xx.max(),yy.max()])
  bounds=[np.array(v) for v in bounds];summary_bounds=[]
  for view,bb in enumerate(bounds):
   ext=[float(bb[:,0].min()),float(bb[:,1].min()),float(bb[:,2].max()),float(bb[:,3].max())];crop=met['common_crops'][view];assert ext[0]>=crop[0] and ext[1]>=crop[1] and ext[2]<=crop[2] and ext[3]<=crop[3],(key,view,ext,crop);summary_bounds.append(ext)
  cases[key]=dict(source_frames=expected,full_force_time_range_s=[float(z['times'][0]),float(z['times'][-1])],moment_balance_error=s['checks']['moment_balance_max_error'],landmark_projection_bounds_px=summary_bounds,common_ortho_width_m=cam['ortho_width_m'])
 reader=imageio_ffmpeg.read_frames(str(OUT/met['file']),pix_fmt='rgb24');meta=next(reader);n=0;footer=0;shape=(meta['size'][1],meta['size'][0],3)
 for data in reader:
  im=np.frombuffer(data,np.uint8).reshape(shape);footer=max(footer,int(abs(im[1540:].astype(int)-np.array([13,23,38])).max()));n+=1
 assert n==round(met['duration_s']*met['fps']) and footer<=5,(n,footer)
 code_files=['compose_iron_man_final.py','render_squat_power_blender.py','build_yun_case_muscles.py','published_knee_paths_profile.py','published_patella_profile.py','prepare_yun_full_visuals.py','prepare_iron_man_exposure.py','case_absolute_render.py','case_exposure_render.py']
 audit=dict(status='passed',source_code_sha256={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in code_files},video_sha256=hashlib.sha256((OUT/met['file']).read_bytes()).hexdigest(),platform_display='Outline, visually checked to preserve full-body muscle visibility',decoded_frames=n,video=meta,footer_max_RGB_error=footer,bottom_reserved_px=384,cases=cases,path_audit=json.loads((OUT/'path_audit.json').read_text()),arm_motion=json.loads((OUT/'arm_motion_comparison.json').read_text()),not_an_empirical_muscle_force_validation=True,model_uncertainty='Conditional force sharing and deep-flexion single-obstacle tendon route approximation; source omissions are recorded in path_audit.json.')
 (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2));print('FINAL_AUDIT_PASS',n,footer,flush=True)
if __name__=='__main__':main()
