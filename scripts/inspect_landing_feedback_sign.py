"""Native contact-response sign around the actual nearly-quiet state."""
import json
import numpy as np,mujoco
from jump_monitor_data import ROOT,read,atomic
from native_rigid_upper_landing import actual_contact_wrench

def main():
    out=ROOT/'results/landing_controller_audit_v1';case=read(out/'selected_case.json');folder=out/case['id']/'baseline';z=np.load(folder/'control_motion.npz');m=mujoco.MjModel.from_binary_path(str(folder/'control_model.mjb'));d=mujoco.MjData(m);names=read(folder/'joint_names.json');da=np.array([m.jnt_dofadr[m.joint(name).id]for name in names]);i=int(np.argmin(abs(z['times']-case['quiet_time_s'])));d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];v=d.qvel[da];feedback=z['joint_velocity_feedback_Nm'][i];valid=abs(v)>1e-10;m.dof_damping[da[valid]]=-feedback[valid]/v[valid];base=z['joint_torque_Nm'][i].copy()
    def response(tau):
        d.qfrc_applied[:]=0.;d.qfrc_applied[da]=tau;d.qacc_warmstart[:]=0.;mujoco.mj_forward(m,d);F,M,COP=actual_contact_wrench(m,d);return np.r_[F/87,d.qacc[2]],F,COP
    first,F,COP=response(base);derivatives={}
    for stem in ['ankle_angle','hip_flexion','knee_angle','flex_extension']:
        direction=np.array([float(n.startswith(stem))for n in names]);a=response(base+2*direction)[0];b=response(base-2*direction)[0];derivatives[stem]=dict(COM_acceleration_y_per_paired_Nm=float((a[1]-b[1])/4),root_pitch_acceleration_per_paired_Nm=float((a[3]-b[3])/4))
    report=read(folder/'control_summary.json');p=report['parameters'];P=z['contact_positions_m'][i];C=z['com'][i];V=z['velocity'][i];desired_y=float(P[:,1].mean());balance=p['balance_gain']*(350*(C[1]-desired_y)+100*V[1]);derivative=derivatives['ankle_angle']['COM_acceleration_y_per_paired_Nm']
    result=dict(case=case,force_reconstruction_max_error_N=float(abs(F-z['force_world_N'][i]).max()),quiet_COM_y_m=float(C[1]),quiet_COM_vy_m_s=float(V[1]),sole_center_y_m=desired_y,actual_COP_y_m=float(COP[1]),balance_command_per_ankle_Nm=float(balance/2),native_response_derivatives=derivatives,ankle_velocity_feedback_acceleration_coefficient_per_s=float(derivative*p['balance_gain']*50),interpretation='Positive acceleration coefficient means the current +100*Vy ankle law reinforces COM velocity at this native contact state; negative means damping. This is a local contact-state result, not a global proof.')
    atomic(out/'feedback_sign.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
