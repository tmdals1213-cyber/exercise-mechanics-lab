"""Four stages: replay exact native physics, keep signed torques and GRF."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import copy,numpy as np
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v2'

def one(c):
    from controlled_broad_jump import run
    source=Path(c['source']);r=read(source/'control_summary.json');p=copy.deepcopy(r['parameters']);p.update(record_energy_ledger=True,record_controller_components=False);result,z=run(p,OUT/'physics_metrics'/c['key'],dt=r['physics_dt_s']);original=np.load(source/'control_motion.npz');error=float(abs(z['qpos']-original['qpos']).max());assert error<1e-10
    names=read(source/'joint_names.json');ids=[names.index(n)for n in ['hip_flexion_r','knee_angle_r','ankle_angle_r']];ft=z['physics_times_s'];force=np.linalg.norm(z['physics_force_world_N'],axis=1)/(87*9.80665);tau=z['physics_joint_net_torque_Nm'][:,ids];bottom=p['ready_s']+p['crouch_s']+p.get('prepush_lean_s',0.);push=(ft>=bottom)&(ft<r['takeoff_s']);signals=dict(ground=force,hip=abs(tau[:,0]),knee=abs(tau[:,1]),ankle=abs(tau[:,2]));caps=np.array([p['joint_torque_limits_Nm'][names[j]]for j in ids]);assert abs(tau).max(axis=0).max()>0 and np.max(abs(tau)/caps)<=1+1e-7
    np.savez_compressed(OUT/f'{c["key"]}_metrics.npz',times=ft,signed_torque_Nm=tau,**signals)
    return dict(key=c['key'],source=str(source),dt_s=r['physics_dt_s'],source_replay_qpos_error=error,peak_push={k:float(v[push].max())for k,v in signals.items()},full_cycle_peak={k:float(v.max())for k,v in signals.items()},phase_times=[0.,p['ready_s'],bottom,r['takeoff_s'],r['touchdown_s'],float(z['times'][-1])],checked_distance_m=c['checked_distance_m'],joint_limits_Nm=caps.tolist(),torque_limits_all_Nm=p['joint_torque_limits_Nm'],max_torque_cap_ratio=float(np.max(abs(tau)/caps)),scope='Native resultant GRF / BW; right-side net torques / Nm. Not individual muscle forces.')

def main():
    m=read(OUT/'source_manifest.json');cases=[c for c in m['cases']if c['key']in m['final_four_keys']]
    with ProcessPoolExecutor(max_workers=2)as pool:rows=list(pool.map(one,cases))
    assert all(r['torque_limits_all_Nm']==rows[0]['torque_limits_all_Nm']for r in rows)
    atomic(OUT/'metrics.json',dict(cases={r['key']:r for r in rows},comparison_keys=m['final_four_keys'],individual_muscle_tensions_available=False,force_unit='BW',torque_unit='Nm',summary_scope='Push-off peaks; full-cycle native traces retained separately.'))
    print('V2_METRICS_READY',[(r['key'],r['peak_push'],r['full_cycle_peak']['ground'])for r in rows],flush=True)

if __name__=='__main__':main()
