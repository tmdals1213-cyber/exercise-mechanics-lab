"""Kinematic consistency diagnostics under the current, uncalibrated reference.

No forces are published, no hidden support is added, and no pose is corrected.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from retarget_measured_jump import MeasuredJumpRig
from mechanics_engine_adapter import sample
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=ROOT/a.folder
 z=np.load(folder/'motion.npz');r=json.loads((folder/'tracking_report.json').read_text());cfg=json.loads((ROOT/r['reference_configuration']).read_text());ts=z['source_times_s'];q=z['qpos'];rig=MeasuredJumpRig(87,1.78);m,d=rig.m,rig.d
 com=[];momentum=[]
 for i,pose in enumerate(q):
  left=max(0,i-1);right=min(len(q)-1,i+1);v=np.zeros(m.nv);mujoco.mj_differentiatePos(m,v,ts[right]-ts[left],q[left],q[right]);mech=sample(rig,pose,v,np.zeros(m.nv));c=m.body_mass@d.xipos/m.body_mass.sum();com.append(c.copy())
  H=np.einsum('bij,bj->bi',mech['inertia_world'],mech['omega'])+np.cross(d.xipos-c,m.body_mass[:,None]*mech['velocity']);momentum.append(H.sum(0))
 com=np.array(com);momentum=np.array(momentum);takeoff=cfg['support_boundaries_s'][1];contact=cfg['support_boundaries_s'][2]
 flight_start,flight_end=cfg.get('flight_COM_interval_s',[takeoff+.04,contact-.04]);flight=(ts>=flight_start)&(ts<=flight_end);tf=ts[flight]-takeoff;assert len(tf)>=4
 X=np.c_[np.ones(len(tf)),tf];g=np.asarray(m.opt.gravity);target=com[flight]-.5*tf[:,None]**2*g
 fit=X@np.linalg.lstsq(X,target,rcond=None)[0]+.5*tf[:,None]**2*g;free=np.linalg.lstsq(np.c_[X,.5*tf**2],com[flight],rcond=None)[0]
 residual=com[flight]-fit;H=momentum[flight];report=dict(status='uncalibrated_kinematic_flight_diagnostic',
  source_interval_s=[float(ts[flight][0]),float(ts[flight][-1])],samples=int(flight.sum()),
  assumed_native_body_mass_kg=87,assumed_native_height_m=1.78,physical_time_verified=False,metric_geometry_verified=False,
  constant_gravity_ballistic_fit_RMS_m=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
  freely_fitted_COM_acceleration_m_s2=free[2].tolist(),
  whole_body_angular_momentum_max_deviation_from_mean_kg_m2_s=float(np.linalg.norm(H-H.mean(0),axis=1).max()),
  source_motion_modified=False,force_analysis_allowed=False,
  interpretation='Diagnostics of the currently inferred 3D trajectory only. Camera geometry, playback rate, marker errors and anthropometry can affect these results. Numerical consistency is not an experimental force validation.')
 np.savez_compressed(folder/'kinematic_diagnostics.npz',source_times_s=ts,COM_m=com,angular_momentum_about_COM_kg_m2_s=momentum,flight_mask=flight)
 (folder/'flight_diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
