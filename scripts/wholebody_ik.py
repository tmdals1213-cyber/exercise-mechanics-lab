"""Task-independent full-body point/frame IK with joint limits and couplers.

This fits configurations; it does not solve a continuous dynamic movement.
"""
from dataclasses import dataclass
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass
class FrameTarget:
    name:str
    body_id:int
    point_local:np.ndarray
    position_world:np.ndarray
    rotation_world:np.ndarray|None=None


def solve_wholebody_ik(body,seed,targets,support_bounds_world_xy=None,external_masses=(),max_evaluations=220):
    m,d=body.m,body.d
    joint_values=np.array([seed[m.jnt_qposadr[j]] for j in body.independent])
    x0=np.r_[seed[:3],np.zeros(3),joint_values]
    lower=np.r_[seed[:3]-.3,np.full(3,-.55),[m.jnt_range[j,0] if m.jnt_limited[j] else -np.pi for j in body.independent]]
    upper=np.r_[seed[:3]+.3,np.full(3,.55),[m.jnt_range[j,1] if m.jnt_limited[j] else np.pi for j in body.independent]]
    x0=np.maximum(np.minimum(x0,upper-1e-8),lower+1e-8)
    # The task adapter owns support assumptions. Hand-only or other tasks do
    # not acquire a hidden ground/support constraint merely by using this IK.
    bounds=None if support_bounds_world_xy is None else np.asarray(support_bounds_world_xy,float)
    if bounds is not None and (bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds[0]>=bounds[1])):raise ValueError('Invalid support bounds')
    mass_sum=0.;mass_position_sum=np.zeros(3)
    for mass,position_world in external_masses:
        position_world=np.asarray(position_world,float)
        if not np.isfinite(mass) or mass<0 or position_world.shape!=(3,) or not np.isfinite(position_world).all():raise ValueError('Invalid external point mass')
        mass_sum+=mass;mass_position_sum+=mass*position_world
    def position(x):
        q=seed.copy();q[:3]=x[:3];quat=q[3:7].copy();mujoco.mju_quatIntegrate(quat,x[3:6],1.);q[3:7]=quat
        for j,v in zip(body.independent,x[6:]):q[m.jnt_qposadr[j]]=v
        d.qpos[:]=body.complete(q);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        return d.qpos.copy()
    def residual(x):
        position(x);values=[]
        for target in targets:
            R=d.xmat[target.body_id].reshape(3,3);p=d.xpos[target.body_id]+R@target.point_local
            values.extend((p-target.position_world)*1000.)
            if target.rotation_world is not None:
                values.extend(Rotation.from_matrix(target.rotation_world.T@R).as_rotvec()*100.)
        # Whole-body posture regularization is deliberately soft: the trunk,
        # hips, knees, ankles, root and arms can compensate as geometry changes.
        values.extend((x-x0)*.2)
        com=np.average(d.xipos[1:],axis=0,weights=m.body_mass[1:])
        system=(m.body_mass.sum()*com+mass_position_sum)/(m.body_mass.sum()+mass_sum)
        if bounds is not None:
            values.extend(np.maximum(bounds[0]-system[:2],0)*1000.)
            values.extend(np.maximum(system[:2]-bounds[1],0)*1000.)
        return np.asarray(values)
    answer=least_squares(residual,x0,bounds=(lower,upper),ftol=1e-10,xtol=1e-10,gtol=1e-9,max_nfev=max_evaluations,x_scale='jac')
    q=position(answer.x);body.pose(q);position_errors={};rotation_errors={}
    for target in targets:
        R=d.xmat[target.body_id].reshape(3,3);p=d.xpos[target.body_id]+R@target.point_local
        position_errors[target.name]=float(np.linalg.norm(p-target.position_world))
        if target.rotation_world is not None:rotation_errors[target.name]=float(np.linalg.norm(Rotation.from_matrix(target.rotation_world.T@R).as_rotvec()))
    com=np.average(d.xipos[1:],axis=0,weights=m.body_mass[1:]);system=(m.body_mass.sum()*com+mass_position_sum)/(m.body_mass.sum()+mass_sum)
    margin=np.minimum(system[:2]-bounds[0],bounds[1]-system[:2]) if bounds is not None else None
    feasible=max(position_errors.values())<.001 and max(rotation_errors.values(),default=0)<.01 and (margin is None or margin.min()>-.001)
    report={'status':'fitted' if feasible else 'target_not_fitted','position_errors_m':position_errors,'orientation_errors_rad':rotation_errors,'system_COM_support_margin_xy_m':margin.tolist() if margin is not None else None,
            'solver_success':bool(answer.success),'function_evaluations':answer.nfev,'joint_angle_changes_rad':dict(zip(body.ind_names,(answer.x[6:]-joint_values).tolist())),
            'root_translation_change_m':(q[:3]-seed[:3]).tolist(),'root_rotation_increment_rad':answer.x[3:6].tolist(),'ground_support_constraint_applied':bounds is not None,'scope':'Full-body static configuration fit with arbitrary point/frame targets and optional task-supplied support bounds, not continuous motion or dynamic contact simulation.'}
    if not feasible:raise RuntimeError(str(report))
    return q,report
