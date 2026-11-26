"""Independent uncached replays of the overnight seed, best sample and final mean."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json, time
import numpy as np
from controlled_broad_jump import ROOT, run
from jump_monitor_data import atomic, read, lines

OUT=ROOT/'results/controlled_broad_jump/overnight_review_2026_11_25'

def replay(job):
    label,parameters,dt=job;parameters=parameters.copy();parameters['record_energy_ledger']=True
    folder=OUT/label
    report,z=run(parameters,folder,dt=dt)
    import subprocess,sys,mujoco
    for script in ['audit_control_jump.py','audit_jump_control_ledger.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(folder)],check=True,stdout=subprocess.DEVNULL)
    model=mujoco.MjModel.from_binary_path(str(folder/'control_model.mjb'))
    names=read(folder/'joint_names.json');limits=np.array([parameters['joint_torque_limits_Nm'][n]for n in names]);torques=z['physics_joint_net_torque_Nm'];saturation=np.mean(abs(torques)>=limits[None,:]*(1-1e-6),axis=0)
    rom=float(np.maximum(z['qpos'][:,3:]-model.jnt_range[3:,1],model.jnt_range[3:,0]-z['qpos'][:,3:]).max())
    t=z['times'];late=t>t[-1]-.5;speed=float(np.linalg.norm(z['velocity'][late],axis=1).max());to=report['takeoff_s'];td=report['touchdown_s'];pt=z['physics_times_s'];fz=z['physics_force_world_N'][:,2];push=(pt<to);land=(pt>=td)&(pt<td+.4)
    mechanics=read(folder/'mechanics_audit.json');ledger=read(folder/'actuator_ledger_audit.json')
    result=dict(label=label,folder=str(folder),dt_s=dt,status=report['status'],stable=bool(report['stable_landing']and speed<.15),distance_m=report.get('valid_contact_distance_m'),first_heel_m=report.get('first_contact_heel_distance_m'),last_half_second_max_speed_m_s=speed,max_ROM_excess_rad=rom,takeoff_velocity_m_s=report['takeoff_velocity_m_s'],takeoff_angle_deg=float(np.degrees(np.arctan2(report['takeoff_velocity_m_s'][2],-report['takeoff_velocity_m_s'][1]))),propulsion_peak_BW=float(fz[push].max()/(87*9.80665)),landing_peak_BW=float(fz[land].max()/(87*9.80665)),net_actuator_work_J=report['net_actuator_work_J'],joint_names=names,torque_limits_Nm=limits.tolist(),net_torque_peak_Nm=np.max(abs(torques),axis=0).tolist(),net_torque_capacity_fraction=(np.max(abs(torques),axis=0)/limits).tolist(),fraction_physics_samples_at_capacity=saturation.tolist(),mechanics=mechanics,actuator_ledger=ledger)
    atomic(folder/'review_metrics.json',result);print('REVIEW_REPLAY',label,result['distance_m'],result['stable'],mechanics['status'],ledger['status'],flush=True)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);training=ROOT/'results/controlled_broad_jump/ppo_full_overnight_v31';env=read(training/'environment.json')
    # Torch stays out of the native-physics children.
    import torch
    from train_jump_overnight import InstrumentedPPO
    torch.set_num_threads(1);model=InstrumentedPPO.load(training/'policy_final.zip',device='cpu');obs=np.array(env['observation'],np.float32);action,_=model.predict(obs,deterministic=True);action=np.clip(action,-1,1)
    baseline=read(training/'worker_00/baseline/control_summary.json')['parameters'];baseline['record_energy_ledger']=True;final=baseline.copy()
    for v,spec in zip(action,env['actions']):
        key=spec['key'];center=spec['center'];final[key]=float(center+v*((spec['maximum']-center)if v>=0 else(center-spec['minimum'])))
    choices=[(folder,read(folder/'control_summary.json'))for folder in training.glob('worker_*/best_episode')];best_folder,best=max(choices,key=lambda pair:pair[1].get('valid_contact_distance_m')or -1)
    atomic(OUT/'review_definition.json',dict(started_at=time.time(),best_source=str(best_folder),baseline_source=str(training/'worker_00/baseline'),checkpoint=str(training/'policy_final.zip'),checkpoint_num_timesteps=model.num_timesteps,final_mean_action=action.tolist(),mean_action_max_abs=float(abs(action).max()),mean_action_L2=float(np.linalg.norm(action)),all_43_outputs_active=True,body_kg=87,height_m=1.78,capacity=read(ROOT/'results/controlled_broad_jump/two_metre_initial_solution_verified/fixed_virtual_capacity.json'),cases=['initial_500us','best_500us','best_250us','best_125us','mean_500us','mean_250us','mean_125us']))
    jobs=[('initial_500us',baseline,.0005)]+[(f'{kind}_{us}us',p,us/1e6)for kind,p in [('best',best['parameters']),('mean',final)]for us in [500,250,125]]
    with ProcessPoolExecutor(max_workers=2)as pool:rows=list(pool.map(replay,jobs))
    atomic(OUT/'replay_results.json',rows);print('OVERNIGHT_REVIEW_REPLAY_READY',OUT,flush=True)

if __name__=='__main__':main()
