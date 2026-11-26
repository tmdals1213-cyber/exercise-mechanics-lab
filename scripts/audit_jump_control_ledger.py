"""Independent dynamic and work balance for explicit+implicit virtual actuation."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco

def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder');a=ap.parse_args();p=Path(a.folder);r=json.loads((p/'control_summary.json').read_text());z=np.load(p/'control_motion.npz');m=mujoco.MjModel.from_binary_path(str(p/'control_model.mjb'));d=mujoco.MjData(m);names=json.loads((p/'joint_names.json').read_text());da=np.array([m.jnt_dofadr[m.joint(n).id]for n in names]);M=np.zeros((m.nv,m.nv));residuals=[];feedback_errors=[]
 for i in np.linspace(0,len(z['times'])-1,65).round().astype(int):
  t=z['times'][i];d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];m.dof_damping[:]=0
  for j,n in enumerate(names):
   kd=35. if n.startswith(('hip','knee'))else 15. if n.startswith('ankle')else 70. if n=='flex_extension'else 10.
   if r['touchdown_s']is not None and t>=r['touchdown_s']:
    u=float(np.clip((t-r['touchdown_s']-.35)/1.5,0,1));rec=u**3*(10-15*u+6*u*u)
    if n.startswith(('hip','knee')):base=r['parameters'].get('landing_support_damping',10.);kd=base+(55-base)*rec
    if n.startswith('ankle'):kd=8+17*rec
   m.dof_damping[da[j]]=kd
  geoms=[j for j in range(m.ngeom)if m.geom(j).name.startswith('control_')];m.geom_solref[geoms,0]=r['parameters'].get('contact_time_constant_s',.008)if r['takeoff_s']is None or t<r['takeoff_s']else r['parameters'].get('landing_contact_time_constant_s',.02)
  d.qfrc_applied[:]=0;d.qfrc_applied[da]=z['joint_torque_Nm'][i];mujoco.mj_forward(m,d);mujoco.mj_fullM(m,d,M);required=M@d.qacc+d.qfrc_bias-d.qfrc_constraint;residuals.append(float(abs(required[da]-z['joint_net_torque_Nm'][i]).max()));feedback_errors.append(float(abs(d.qfrc_passive[da]-z['joint_velocity_feedback_Nm'][i]).max()))
 result=dict(status='passed'if max(residuals)<1e-6 and max(feedback_errors)<1e-8 else'needs_work',max_inverse_dynamic_joint_torque_residual_Nm=max(residuals),max_feedback_reconstruction_error_Nm=max(feedback_errors),definition='Net virtual joint actuation = explicit applied P/feedforward + implicit velocity feedback. All native constraints are retained in the independent inverse-dynamics check.',net_work_J=r['net_actuator_work_J'],individual_muscle_force_claim=False,source_documentation='https://mujoco.readthedocs.io/en/stable/computation/index.html')
 if'physics_energy_J'in z:
  t=z['physics_times_s'];E=z['physics_energy_J'].sum(1);input_work=np.trapezoid(z['physics_net_joint_power_W'],t);boundary_work=np.trapezoid(z['physics_constraint_power_W'],t);res=float(E[-1]-E[0]-input_work-boundary_work);scale=max(1.,float(np.trapezoid(z['physics_positive_net_joint_power_W'],t)));result['energy_balance']=dict(energy_change_J=float(E[-1]-E[0]),signed_net_actuator_work_J=float(input_work),signed_contact_and_limit_work_J=float(boundary_work),integration_residual_J=res,residual_over_positive_net_work=abs(res)/scale);result['status']='passed'if result['status']=='passed'and abs(res)/scale<.05 else'needs_work'
 (p/'actuator_ledger_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
