"""Replay two displayed cases at their native timestep for physics-rate peaks."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import copy,json
import numpy as np
from jump_monitor_data import ROOT,read,atomic

OUT=ROOT/'deliverables/jump_rl_study_draft_v1'

def one(case):
    from controlled_broad_jump import run
    folder=Path(case['folder']);source=Path(case['source']);report=read(source/'control_summary.json');p=copy.deepcopy(report['parameters']);p.update(record_energy_ledger=True,record_controller_components=False)
    result,z=run(p,OUT/'physics_metrics'/case['key'],dt=report['physics_dt_s']);original=np.load(source/'control_motion.npz');qerror=float(abs(z['qpos']-original['qpos']).max());assert qerror<1e-10,qerror
    names=read(source/'joint_names.json');ft=z['physics_times_s'];tau=z['physics_joint_net_torque_Nm'];mass=result['registration']['mass_kg'];force=np.linalg.norm(z['physics_force_world_N'],axis=1)/(mass*9.80665);bottom=p['ready_s']+p['crouch_s']+p.get('prepush_lean_s',0.);push=(ft>=bottom)&(ft<result['takeoff_s']);signals={'ground':force}
    for key,name in [('hip','hip_flexion_r'),('knee','knee_angle_r'),('ankle','ankle_angle_r')]:signals[key]=abs(tau[:,names.index(name)])
    np.savez_compressed(OUT/f'{case["key"]}_metrics.npz',times=ft,**signals)
    return dict(key=case['key'],source=str(source),dt_s=report['physics_dt_s'],source_replay_qpos_error=qerror,peak_push={key:float(values[push].max())for key,values in signals.items()},phase_times=[0.,p['ready_s'],bottom,result['takeoff_s'],result['touchdown_s'],float(z['times'][-1])],first_contact_m=result['first_contact_heel_distance_m'],retained_m=case['record']['retained_distance_m'],mass_kg=mass,torque_limits_Nm=p['joint_torque_limits_Nm'],scope='Native resultantGRF magnitude /BW; right-side net joint torque magnitudes /Nm. Push-off peak between the target bottom/prelean end and native takeoff. Not muscle tensions.')

def main():
    manifest=read(OUT/'source_manifest.json');chosen=[case for case in manifest['cases']if case['key']in ['initial','best_checked']]
    with ProcessPoolExecutor(max_workers=2)as pool:results=list(pool.map(one,chosen))
    assert results[0]['torque_limits_Nm']==results[1]['torque_limits_Nm'];atomic(OUT/'metrics.json',dict(cases={row['key']:row for row in results},force_unit='Body weight',torque_unit='Nm',individual_muscle_tensions_available=False,curves_phase_aligned=True,peak_comparison_phase='Push-off only'));print('DRAFT_PHYSICS_METRICS_READY',[(row['key'],row['peak_push'])for row in results],flush=True)

if __name__=='__main__':main()
