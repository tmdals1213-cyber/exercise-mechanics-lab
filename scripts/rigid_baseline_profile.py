"""Explicit mechanics-only model profile, isolated from historical results.

Coordinate-helper inertias are numerical epsilons, not anatomical segments.
Non-helper mass fractions are preserved and renormalized to the subject mass.
No fitted human joint resistance or muscle-recruitment claim is introduced.
"""
import json,hashlib
from pathlib import Path
import numpy as np,mujoco
from jump_rig import JumpRig

HELPERS=[stem+'_'+side for side in ['r','l'] for stem in ['clavphant','scapphant','humphant','humphant1']]

def apply_rigid_profile(rig,helper_mass_kg=1e-6,helper_radius_m=.001,armature_kg_m2=1e-8):
    m=rig.m;total=float(m.body_mass.sum());ids=[m.body(n).id for n in HELPERS];physical=[i for i in range(1,m.nbody) if i not in ids and m.body_mass[i]>0];before=[]
    for i in ids:before.append({'body':m.body(i).name,'mass_kg':float(m.body_mass[i]),'I_kg_m2':m.body_inertia[i].tolist(),'radius_of_gyration_m':np.sqrt(m.body_inertia[i]/m.body_mass[i]).tolist()})
    ratio=(total-len(ids)*helper_mass_kg)/float(m.body_mass[physical].sum());m.body_mass[physical]*=ratio;m.body_inertia[physical]*=ratio
    for i in ids:m.body_mass[i]=helper_mass_kg;m.body_inertia[i]=.4*helper_mass_kg*helper_radius_m**2
    # These native control parameters are not calibrated anatomical tissues.
    m.dof_damping[:]=0;m.dof_frictionloss[:]=0;m.jnt_stiffness[:]=0;m.body_gravcomp[:]=0
    m.tendon_stiffness[:]=0;m.tendon_damping[:]=0;m.dof_armature[:]=armature_kg_m2;m.dof_armature[:6]=0
    mujoco.mj_setConst(m,rig.d);rig.b.pose(rig.b.q0)
    report={'profile':'rigid_baseline_v1','subject_mass_kg':float(m.body_mass.sum()),'helper_bodies':HELPERS,'helper_mass_kg':helper_mass_kg,'helper_radius_m':helper_radius_m,'helper_inertia_kg_m2':.4*helper_mass_kg*helper_radius_m**2,'numerical_armature_kg_m2':armature_kg_m2,'nonhelper_mass_inertia_rescale':ratio,'joint_damping_and_springs':'zero, uncalibrated passive joint resistance omitted','native_helper_inertias_before':before,'human_motion_validated':False,'muscle_forces_released':False}
    rig.b.report['rigid_baseline_profile']=report
    rig.b.report['implementation_sha256']=hashlib.sha256((rig.b.report['implementation_sha256']+json.dumps(report,sort_keys=True)+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()
    return report

class RigidBaselineRig(JumpRig):
    def __init__(self,subject=None,**settings):
        super().__init__(subject=subject);self.profile=apply_rigid_profile(self,**settings)
