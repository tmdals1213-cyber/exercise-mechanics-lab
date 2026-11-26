"""Shared model factory; articulated MTP joints for a heel-rise jump."""
import hashlib
from pathlib import Path
import numpy as np,mujoco
from scapulothoracic_rig import ScapulothoracicRig


class ArticulatedJumpRig(ScapulothoracicRig):
    def __init__(self,subject=None):
        super().__init__(subject=subject);m,b=self.m,self.b
        self.toe_joints=[b.joints['mtp_angle_'+side] for side in ['r','l']]
        self.toe_q=[int(m.jnt_qposadr[j]) for j in self.toe_joints]
        for k in range(m.neq):
            if int(m.eq_obj1id[k]) in self.toe_joints:m.eq_active0[k]=False;self.d.eq_active[k]=False
        self.eqs=[equation for equation in self.eqs if equation[0] not in self.toe_q]
        for j in self.toe_joints:
            if j not in b.independent:b.independent.append(j)
        b.ind_names=[m.joint(j).name for j in b.independent];b.coordinate_names=['root_x','root_y','root_z','root_rx','root_ry','root_rz']+b.ind_names
        self.ids=[int(m.jnt_qposadr[j]) for j in b.independent];self.dofs=[int(m.jnt_dofadr[j]) for j in b.independent];self.n=6+len(self.ids)
        self.B0=np.zeros((m.nv,self.n));self.B0[:6,:6]=np.eye(6);self.B0[self.dofs,np.arange(6,self.n)]=1.
        self.girdle_complete=b.complete;b.complete=self.complete_with_toes;b.pose=self.pose_with_toes
        self.profile['MTP']='bilateral independent anatomical hinges; native muscle paths retained; constant-zero couplers disabled'
        b.report['implementation_sha256']=hashlib.sha256((b.report['implementation_sha256']+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()

    def complete_with_toes(self,q):
        values=np.asarray(q)[self.toe_q].copy();result=self.girdle_complete(q);result[self.toe_q]=values;return result

    def pose_with_toes(self,q,velocity=None,acceleration=None):
        self.d.qpos[:]=self.complete_with_toes(q);self.d.qvel[:]=0 if velocity is None else velocity;mujoco.mj_forward(self.m,self.d);self.d.qacc[:]=0 if acceleration is None else acceleration


def create_rig(rig_type,setup=None):
    if rig_type=='measured_cmj_standard':
        from retarget_measured_jump import MeasuredJumpRig
        return MeasuredJumpRig(mass_kg=(setup or {}).get('mass_kg',88.),height_m=(setup or {}).get('height_m',1.8))
    if rig_type=='scapular_contact_articulated_toes':return ArticulatedJumpRig()
    if rig_type=='scapular_contact':return ScapulothoracicRig()
    from rigid_baseline_profile import RigidBaselineRig
    return RigidBaselineRig()
