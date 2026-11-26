"""Reproducible quiet-support check for the native landing contact bridge.

Passing checks numerical support, not realistic human landing or muscle forces.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from native_rigid_upper_landing import build,actual_contact_wrench

ROOT=Path(__file__).resolve().parents[1]


def main():
    r=MeasuredJumpRig(87,1.78)
    q,_=r.planted_pose(.1,.15,.07,arms=(np.pi/2,.12,.15),trunk=0.)
    r.b.pose(q)
    q[2]-=np.mean([r.d.site_xpos[r.m.site('subject_sole_reference_'+s).id,2] for s in ['r','l']])
    r.contact_stand_override=q.copy();points=contact_geometry(r)['points']
    m,d,a,red,mapping,check,_=build(r,q,np.zeros(r.m.nv),points,.0005,0,-.14)
    target=d.qpos[a.ids].copy();names=a.b.ind_names
    kp=np.array([2000. if n.startswith(('hip_flexion','knee_angle')) else 800. if n.startswith(('hip_adduction','hip_rotation','ankle_angle')) else 120. for n in names])
    kd=np.array([100. if n.startswith('hip_') else 70. if n.startswith('knee_') else 20. for n in names])
    m.dof_damping[a.dofs]=kd;C0=d.subtree_com[1].copy();rows=[]
    for i in range(4001):
        d.qfrc_applied[:]=0;d.qfrc_applied[a.dofs]=kp*(target-d.qpos[a.ids]);mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d)
        if i%16==0:
            F,M,cop=actual_contact_wrench(m,d)
            rows.append((i*.0005,d.subtree_com[1].copy(),d.subtree_linvel[1].copy(),F,cop))
        if i<4000:mujoco.mj_step(m,d)
    last=rows[-1];steady=[row for row in rows if row[0]>=1.]
    report=dict(status='quiet_support_check',duration_s=2.,integration_hz=2000,subject_mass_kg=87.,subject_height_m=1.78,
                passed=bool(abs(last[1][2]-C0[2])<.02 and np.linalg.norm(last[2])<.1),
                final_COM_velocity_m_s=last[2].tolist(),COM_height_change_m=float(last[1][2]-C0[2]),
                mean_GRF_last_second_N=np.mean([row[3] for row in steady],axis=0).tolist(),
                expected_weight_N=87*9.80665,root_actuator=False,strength_ceiling=False,
                transition_check=check,human_motion_validated=False,muscle_force_validated=False,
                limitation='Quiet stance only; this does not establish supported landing.')
    out=ROOT/'results/physical_jump_task/native_quiet_support';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'motion.npz',times=[x[0] for x in rows],com=[x[1] for x in rows],velocity=[x[2] for x in rows],force_world_N=[x[3] for x in rows],actual_COP_world_m=[x[4] for x in rows])
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
