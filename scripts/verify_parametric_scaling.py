"""Analytic scaling, independent shape settings, binary and policy checks."""
from pathlib import Path
import json
import numpy as np
import mujoco
from parametric_body import Subject,build_subject

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/parametric_body'


def gravity_and_unit_inertia(b):
    b.pose(b.q0);B=b.reduction(b.q0);g=np.zeros(b.m.nv);mujoco.mj_rne(b.m,b.d,1,g)
    u=np.zeros(B.shape[1]);u[b.coordinate_names.index('shoulder_elv_r')]=1.;b.d.qacc[:]=B@u
    total=np.zeros(b.m.nv);mujoco.mj_rne(b.m,b.d,1,total)
    b.d.qacc[:]=0
    return B.T@g,B.T@(total-g)


def main():
    base=build_subject(Subject());base.pose(base.q0);positions=base.d.xpos.copy();lengths=base.d.actuator_length.copy();g0,i0=gravity_and_unit_inertia(base);checks=[]
    for height in [1.65,1.95]:
        b=build_subject(Subject(height_m=height));scale=height/1.8;b.pose(b.q0)
        geometry=float(abs(b.d.xpos-positions*scale).max());path=float(abs(b.d.actuator_length-lengths*scale).max());inertia=float(abs(b.m.body_inertia-base.m.body_inertia*scale**2).max())
        g,i=gravity_and_unit_inertia(b);gf=np.r_[np.ones(3),np.full(len(g)-3,scale)];inf=np.r_[np.full(3,scale),np.full(len(i)-3,scale**2)]
        gravity_error=float(abs(g-g0*gf).max());inertial_error=float(abs(i-i0*inf).max())
        assert max(geometry,path,inertia)<1e-8 and max(gravity_error,inertial_error)<1e-5
        assert np.allclose(b.m.actuator_gainprm[:,2],base.m.actuator_gainprm[:,2])
        checks.append({'height_m':height,'position_similarity_error_m':geometry,'path_similarity_error_m':path,'inertia_similarity_error_kg_m2':inertia,'gravity_similarity_error':gravity_error,'dynamic_inertia_similarity_error':inertial_error})
    density=build_subject(Subject(segment_size={'upper_arm':1.1},mass_distribution='constant_density_then_normalize'))
    assert abs(density.m.body_mass.sum()-90)<1e-8
    assert density.m.body_mass[density.m.body('humerus_r').id]>base.m.body_mass[base.m.body('humerus_r').id]
    torso=build_subject(Subject(segment_size={'trunk':1.03,'pelvis':1.02}))
    assert abs(torso.report['stature_reference_m']-1.8)<1e-8
    rejected=0
    for subject in [Subject(height_m=-1),Subject(segment_size={'forearm':0}),Subject(segment_size={'imaginary':1.1})]:
        try:subject.validate()
        except ValueError:rejected+=1
    assert rejected==3
    # Native binary is self-contained and re-loadable without geometry paths.
    binary=np.empty(mujoco.mj_sizeModel(base.m),np.uint8);mujoco.mj_saveModel(base.m,buffer=binary)
    reloaded=mujoco.MjModel.from_binary_path('subject.mjb',assets={'subject.mjb':binary.tobytes()})
    assert reloaded.nu==base.m.nu and np.array_equal(reloaded.body_mass,base.m.body_mass)
    assert not base.graph()['major_group_anatomical_release_ready']
    report={'status':'passed','uniform_scaling':checks,'constant_density_mass_policy':'passed','trunk_and_pelvis_morphology':'compiled_and_stature_checked','invalid_parameters_rejected':rejected,'self_contained_binary_reload':'passed',
            'muscle_force_parameters_independent_of_stature':True,'native_joint_spring_count':int(np.count_nonzero(base.m.jnt_stiffness)),'scope':'Numerical implementation verification, not validation of individual anatomy or dynamic movements.'}
    OUT.mkdir(exist_ok=True);(OUT/'scaling_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
