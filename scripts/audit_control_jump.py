"""Mechanical consistency of the control pilot; no human empirical validation."""
from pathlib import Path
import argparse,json,numpy as np,mujoco
from native_rigid_upper_landing import actual_contact_wrench

def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder');a=ap.parse_args();folder=Path(a.folder);z=np.load(folder/'control_motion.npz');rep=json.loads((folder/'control_summary.json').read_text());m=mujoco.MjModel.from_binary_path(str(folder/'control_model.mjb'));d=mujoco.MjData(m);mass=m.body_mass.sum();times=z['times'];to=rep['takeoff_s'];td=rep['touchdown_s'];mask=(times>to+.035)&(times<td-.035)&(z['force_world_N'][:,2]<1);ids=np.flatnonzero(mask);assert len(ids)>5
 t=times[ids]-times[ids[0]];C=z['com'][ids];V=z['velocity'][ids];g=np.array([0,0,-9.80665]);pred=z['com'][ids[0]]+t[:,None]*z['velocity'][ids[0]]+.5*t[:,None]**2*g;vpred=z['velocity'][ids[0]]+t[:,None]*g;H=z['angular_momentum'][ids];ballistic=float(np.linalg.norm(C-pred,axis=1).max());verr=float(np.linalg.norm(V-vpred,axis=1).max());herror=float(abs(H[:,0]-H[0,0]).max())
 root=np.array([m.jnt_dofadr[m.joint(n).id]for n in ['control_root_y','control_root_z','control_root_pitch']]);names=json.loads((folder/'joint_names.json').read_text());da=np.array([m.jnt_dofadr[m.joint(n).id]for n in names]);closure=[]
 # Reconstruct internal applied torque and native constraints at stored states.
 for i in np.linspace(0,len(times)-1,35).round().astype(int):
  d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.qfrc_applied[:]=0;d.qfrc_applied[da]=z['joint_torque_Nm'][i];mujoco.mj_forward(m,d);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);res=M@d.qacc+d.qfrc_bias-d.qfrc_passive-d.qfrc_applied-d.qfrc_constraint;closure.append(float(abs(res[root]).max()))
 late=times>times[-1]-.5;speed=np.linalg.norm(z['velocity'][late],axis=1);F=z['force_world_N'][:,2];ground=times<to;landing=(times>=td)&(times<td+.4);positive=np.maximum(z['joint_net_power_W']if'joint_net_power_W'in z else z['joint_power_W'],0).sum(1);work={name:float(np.trapezoid(positive[mm],times[mm]))for name,mm in [('ground',ground),('flight',mask),('landing_and_recovery',times>=td)]};stable=bool(not rep['fell'] and speed.max()<.3 and np.mean(F[late])>mass*9.80665*.8 and np.mean(F[late])<mass*9.80665*1.2)
 ft=z['physics_times_s']if'physics_times_s'in z else times;ff=z['physics_force_world_N'][:,2]if'physics_force_world_N'in z else F
 gates=dict(no_root_actuator=rep['maximum_root_applied_force_or_torque']==0,ballistic_COM=ballistic<.008,ballistic_velocity=verr<.03,flight_angular_momentum=herror<.05,native_equation_closure=max(closure)<1e-6,stable_half_second=stable)
 result=dict(status='passed'if all(gates.values())else'needs_work',gates=gates,ballistic_COM_max_error_m=ballistic,ballistic_velocity_max_error_m_s=verr,flight_Hx_max_drift_kg_m2_s=herror,root_equation_residual_max_SI=max(closure),last_half_second_max_COM_speed_m_s=float(speed.max()),last_half_second_mean_GRF_BW=float(np.mean(F[late])/(mass*9.80665)),propulsion_peak_GRF_BW=float(ff[ft<to].max()/(mass*9.80665)),landing_peak_GRF_BW=float(ff[(ft>=td)&(ft<td+.4)].max()/(mass*9.80665)),force_peak_sampling_hz=1/rep['physics_dt_s'],positive_internal_joint_work_J=work,scope='Sagittal mechanics/control pilot; no muscle force release or human performance validation.',lateral_motion_constrained=True,empirical_validation=False)
 result['work_accounting_basis']='net_virtual_joint_actuation'if'joint_net_power_W'in z else'legacy_explicit_applied_only';result['individual_muscle_work_inferred']=False
 (folder/'mechanics_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
