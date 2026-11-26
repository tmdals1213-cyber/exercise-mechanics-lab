"""Native physics worker for one-action-per-jump parameter-policy training."""
from pathlib import Path
import argparse,json,sys,contextlib,hashlib,time
import numpy as np,mujoco
from controlled_broad_jump import ROOT,DEFAULT,build,run
from jump_monitor_data import read,atomic,lines

LANDING_SPECS=[('prepare_landing_s',.025,'s'),('prepare_hip',.12,'rad'),('prepare_knee',.10,'rad'),('prepare_shoulder',.18,'rad'),('prepare_trunk',.08,'rad'),('flight_hip',.10,'rad'),('flight_knee',.12,'rad'),('flight_shoulder',.15,'rad')]
GROUND_SPECS=[('ready_shoulder',.2,'rad'),('hip_bottom',.15,'rad'),('knee_bottom',.15,'rad'),('ankle_bottom',.1,'rad'),('trunk_bottom',.1,'rad'),('lean_m',.06,'m'),('push_s',.05,'s'),('arm_delay_s',.06,'s'),('ankle_delay',.1,'fraction'),('hip_top',.15,'rad'),('pitch_top',.15,'rad')]
FULL_SPECS=[('ready_s',.08,'s'),('crouch_s',.08,'s'),('prepush_lean_s',.04,'s'),('push_s',.025,'s'),('ready_shoulder',.15,'rad'),('shoulder_bottom',.12,'rad'),('shoulder_top',.12,'rad'),('elbow',.08,'rad'),('hip_bottom',.10,'rad'),('knee_bottom',.10,'rad'),('ankle_bottom',.035,'rad'),('trunk_bottom',.06,'rad'),('hip_top',.12,'rad'),('knee_top',.035,'rad'),('ankle_top',.08,'rad'),('trunk_top',.06,'rad'),('hip_delay',.04,'fraction'),('knee_delay',.06,'fraction'),('ankle_delay',.08,'fraction'),('arm_delay_s',.025,'s'),('hip_extension_duration_fraction',.08,'duration_fraction'),('knee_extension_duration_fraction',.08,'duration_fraction'),('lean_m',.06,'m'),('pitch_bottom',.08,'rad'),('pitch_top',.08,'rad'),('flight_transition_s',.025,'s'),('flight_ankle',.04,'rad'),('flight_trunk',.05,'rad'),('prepare_transition_s',.02,'s'),('landing_hip',.08,'rad'),('landing_knee',.08,'rad'),('landing_transition_s',.015,'s'),('landing_ankle',.04,'rad'),('landing_trunk',.05,'rad'),('landing_shoulder',.12,'rad')]+LANDING_SPECS

def parameter_bounds(model,key,span,unit,centre):
 lo,hi=centre-span,centre+span
 if unit=='rad' and not key.startswith('pitch_'):
  stem='hip_flexion'if'hip'in key else'knee_angle'if'knee'in key else'ankle_angle'if'ankle'in key else'shoulder_elv'if'shoulder'in key else'elbow_flexion'if key=='elbow'else'flex_extension'if'trunk'in key else None
  if stem:
   joint=model.joint(stem if stem=='flex_extension'else stem+'_r').id;a,b=model.jnt_range[joint];assert a-1e-8<=centre<=b+1e-8,(key,centre,a,b);lo=max(lo,float(a));hi=min(hi,float(b))
 elif unit=='fraction':lo=max(0.,lo);hi=min(.8,hi)
 elif unit=='duration_fraction':lo=max(.25,lo);hi=min(1.3,hi)
 elif unit=='s' and key!='arm_delay_s':lo=max(0. if key=='prepush_lean_s'else .04,lo)
 assert lo<=centre<=hi,(key,lo,centre,hi)
 return lo,hi

def emit(kind,data):print(kind+' '+json.dumps(data,ensure_ascii=False),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--source',required=True);ap.add_argument('--stage',choices=['landing','full'],default='landing');ap.add_argument('--capacity-file');ap.add_argument('--dt',type=float,default=.0005);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);base=DEFAULT.copy();reference=json.loads((Path(a.source)/'control_summary.json').read_text());base.update(reference['parameters']);specs=LANDING_SPECS if a.stage=='landing'else GROUND_SPECS+LANDING_SPECS;capacity=None
 if a.stage=='full':
  if not a.capacity_file:raise ValueError('Full-jump training requires a fixed virtual actuator-capacity profile.')
  capacity=json.loads(Path(a.capacity_file).read_text());base['joint_torque_limits_Nm']=capacity['joint_torque_limits_Nm'];base['joint_torque_limit_basis']=capacity.get('basis','explicit_applied');specs=FULL_SPECS
  for key,value in dict(prepush_lean_s=0.,hip_delay=0.,knee_delay=0.,hip_extension_duration_fraction=1.,knee_extension_duration_fraction=1.,flight_transition_s=.25,prepare_transition_s=.20,landing_transition_s=.25,landing_ankle=.28,landing_trunk=.15,landing_shoulder=.8).items():base.setdefault(key,value)
 previous=lines(out/'episodes.jsonl');episode_offset=len(previous)
 with contextlib.redirect_stdout(sys.stderr):prepared=build(a.dt)
 velocity=reference['takeoff_velocity_m_s'];obs=np.array([.87,.89,-velocity[1]/3,velocity[2]/3,reference['takeoff_H_kg_m2_s'][0]/20,1.,0. if a.stage=='landing' else 1.],np.float32);cache={}if a.stage=='landing'else None;history=[];best=-np.inf;launch=None
 saved=read(out/'best_episode/control_summary.json',{});best=saved.get('valid_contact_distance_m')if saved.get('stable_landing')else -np.inf
 if best is None:best=-np.inf
 bounds=[parameter_bounds(prepared[1],k,h,u,base[k])if a.stage=='full'else(base[k]-h,base[k]+h)for k,h,u in specs]
 if a.stage=='full':obs=np.array([.87,.89,capacity['joint_torque_limits_Nm']['hip_flexion_r']/1000,capacity['joint_torque_limits_Nm']['knee_angle_r']/1000,capacity['joint_torque_limits_Nm']['ankle_angle_r']/1000,.8,9.80665/10],np.float32)
 definition=dict(stage=a.stage,observation=obs.tolist(),actions=[dict(key=k,center=base[k],half_range=h,unit=u,minimum=lo,maximum=hi)for(k,h,u),(lo,hi)in zip(specs,bounds)],one_action_defines_one_complete_jump=True,fixed_takeoff=a.stage=='landing',same_initial_body=True,strength_profile=capacity,physiology_model=False,dt_s=a.dt,all_posture_and_timing_parameters_jointly_learned=a.stage=='full',control_gains_and_contact_physics_fixed=True,action_mapping='Normalized deviations about a working baseline; bounded by native joint ROM and phase constraints.')
 (out/'environment.json').write_text(json.dumps(definition,indent=2));emit('RL_WORKER_READY',definition)
 def save_case(folder,report,z):
  folder.mkdir(exist_ok=True);np.savez_compressed(folder/'control_motion.tmp.npz',**z);(folder/'control_motion.tmp.npz').replace(folder/'control_motion.npz');(folder/'joint_names.json').write_text(json.dumps(prepared[3]));(folder/'body_names.json').write_text(json.dumps([prepared[1].body(i).name for i in range(prepared[1].nbody)]));mujoco.mj_saveModel(prepared[1],str(folder/'control_model.mjb'));atomic(folder/'control_summary.json',report)
 for line in sys.stdin:
  try:
   cmd=json.loads(line)
   if cmd['cmd']=='close':emit('RL_CLOSED',{});break
   action=np.asarray(cmd['action'],float);assert action.shape==(len(specs),)and np.all(np.isfinite(action));action=np.clip(action,-1,1);p=base.copy()
   for v,(key,span,unit),(lo,hi)in zip(action,specs,bounds):p[key]=float(base[key]+v*((hi-base[key])if v>=0 else(base[key]-lo)))
   with contextlib.redirect_stdout(sys.stderr):report,z=run(p,dt=a.dt,prepared=prepared,flight_cache=cache)
   late=z['times']>z['times'][-1]-.5
   rom=float(np.maximum(z['qpos'][:,3:]-prepared[1].jnt_range[3:,1],prepared[1].jnt_range[3:,0]-z['qpos'][:,3:]).max())
   to=report['takeoff_s'];td=report['touchdown_s'];flight=(z['times']>to)&(z['times']<td if td is not None else True)if to is not None else np.zeros(len(z['times']),bool)
   peakqd=float(abs(z['qvel'][flight,3:]).max())if flight.any()else 0.
   mask=z['physics_times_s']<(to if to is not None else np.inf);pushpeak=float(z['physics_force_world_N'][mask,2].max()/(87*9.80665))if mask.any()else 0.
   quality=bool(rom<.04 and report['minimum_head_torso_pelvis_clearance_proxy_m']>.1 and peakqd<23 and report['net_actuator_work_J']['flight']['positive']<650 and pushpeak<5.5)
   settled=bool(report['stable_landing']and np.linalg.norm(z['velocity'][late],axis=1).max()<.15 and(quality or a.stage=='landing'));distance=report.get('valid_contact_distance_m')if settled else None;launch_state=report.get('takeoff_velocity_m_s')
   if launch is None:launch=launch_state
   if a.stage=='landing':assert np.array_equal(launch_state,launch),'A landing-only action changed launch.'
   reward=float(distance)if distance is not None else -.5-.1*min(5.,float(np.linalg.norm(report['final_COM_velocity_m_s'])));row=dict(episode=episode_offset+len(history),kind=cmd.get('kind','train'),reward=reward,stable=settled,quality_gate=quality,max_ROM_excess_rad=rom,propulsion_peak_BW=pushpeak,contact_distance_m=distance,first_heel_distance_m=report.get('first_contact_heel_distance_m'),action=action.tolist(),parameters={k:p[k]for k,_,_ in specs},takeoff_velocity_m_s=launch_state,compute_seconds=report['compute_seconds'],peak_GRF_BW=report['peak_GRF_BW'],finished_at=time.time());history.append(row)
   with(out/'episodes.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(row)+'\n')
   if row['kind']=='baseline':save_case(out/'baseline',report,z)
   if settled and reward>best:best=reward;save_case(out/'best_episode',report,z);(out/'best_parameters.json').write_text(json.dumps(p,indent=2))
   emit('RL_RESULT',dict(**row,best_valid_distance_m=None if not np.isfinite(best)else float(best),observation=obs.tolist()))
  except Exception as e:emit('RL_ERROR',dict(error=str(e)))
if __name__=='__main__':main()
