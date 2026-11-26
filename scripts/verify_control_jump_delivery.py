from pathlib import Path
import json,hashlib,numpy as np,mujoco,imageio_ffmpeg
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/controlled_broad_jump/controller_pilot_v1';OUT=ROOT/'deliverables/controlled_jump_pilot';BASE=OUT/'baseline'
def main():
 original=np.load(SRC/'baseline/control_motion.npz');replay=np.load(ROOT/'results/controlled_broad_jump/replay_consistency_v1/control_motion.npz');assert set(original.files)==set(replay.files);errors={k:float(np.max(np.abs(original[k]-replay[k])))for k in original.files};assert max(errors.values())==0.,errors
 audits={};velocities=[];frictions={};foot_guard={}
 for c in json.loads((SRC/'cases.json').read_text()):
  name=c['case'];p=SRC/name;z=np.load(p/'control_motion.npz');r=json.loads((p/'control_summary.json').read_text());a=json.loads((p/'mechanics_audit.json').read_text());assert a['status']=='passed';audits[name]=a;velocities.append(r['takeoff_velocity_m_s']);m=mujoco.MjModel.from_binary_path(str(p/'control_model.mjb'));assert m.nu==0 and m.nv==14 and not any(m.joint(j).name.startswith('mtp_angle')for j in range(m.njnt));assert np.all(m.dof_armature==0);assert abs(m.body_mass.sum()-87)<1e-9
  f=z['physics_force_world_N'];used=f[:,2]>1.;ratio=np.linalg.norm(f[used,:2],axis=1)/(.8*f[used,2]);frictions[name]=float(ratio.max());assert ratio.max()<1.0001
  minimum=[]
  for body,radius in [('head',.10),('torso',.14),('pelvis',.10)]:
   bid=m.body(body).id;centres=z['body_xpos'][:,bid]+np.einsum('nij,j->ni',z['body_xmat'][:,bid].reshape(-1,3,3),m.body_ipos[bid]);minimum.append(float((centres[:,2]-radius).min()))
  foot_guard[name]=min(minimum);assert min(minimum)>.02
 assert all(np.array_equal(v,velocities[0])for v in velocities)
 video=BASE/'controlled_jump_review.mp4';reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);assert meta['size']==(1080,1920)and meta['fps']==30;count=0;footer=0
 for data in reader:
  arr=np.frombuffer(data,np.uint8).reshape(1920,1080,3);footer=max(footer,int(np.abs(arr[1540:].astype(int)-[13,23,38]).max()))
  if count in [48,56,134,205,359]:Image.fromarray(arr).save(BASE/f'verified_{count:03d}.png')
  count+=1
 assert count==360 and footer<=5,(count,footer)
 result=dict(status='passed',video_frames=count,duration_s=count/30,footer_max_RGB_error=footer,five_stable_timing_cases=True,identical_takeoff_states=True,replay_array_max_error=max(errors.values()),no_root_actuation=True,rigid_foot_no_MTP=True,added_joint_armature=0.,sagittal_base_DOF=3,internal_joint_controls=11,friction_cone_max_ratios=frictions,minimum_nonfoot_clearance_proxy_m=foot_guard,mechanics_audits=audits,video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),empirical_validation=False,RL_trained=False,maximum_distance_optimized=False)
 (OUT/'completion_audit.json').write_text(json.dumps(result,indent=2));print('CONTROL_DELIVERY_PASS',count,footer,flush=True)
if __name__=='__main__':main()
