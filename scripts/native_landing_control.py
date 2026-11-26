"""Task feedback through native contact dynamics; only internal leg torques.

The finite difference map uses the same compliant contact solver as integration.
It is a local control linearization, not measured-human validation.
"""
import numpy as np
import mujoco
import quadprog
from scipy.spatial.transform import Rotation


def pressure_feedback(model,data,adapter,controller,P,points,B,Jcom,target,kp,pitch,initial_yaw):
    """Pressure-centre feedback, realized by internal joint torques only.

    The simplex constraints restrict desired pressure to the sole polygon.
    Actual pressures and slip are still outputs of the native contact solver.
    """
    mass=model.body_mass.sum();C=data.subtree_com[1];V=Jcom@data.qvel
    centre=P[:,:2].mean(0);h=max(.3,C[2]-controller.height)
    fxy=mass*(-12*(C[:2]-centre)-7*V[:2])
    fz=max(0.,mass*(9.80665-14*V[2]))
    H=data.subtree_angmom[1];moment=-8*H
    cop=C[:2]-h*fxy/max(fz,1.)+np.array([-moment[1],moment[0]])/max(fz,1.)
    A=P[:,:2].T;N=len(P);Q=.02*np.eye(N)+10000*A.T@A;linear=10000*A.T@cop
    constraints=np.c_[np.ones(N),np.eye(N)];bounds=np.r_[1.,np.zeros(N)]
    weights=quadprog.solve_qp(Q,linear,constraints,bounds,meq=1)[0]
    support=np.zeros(model.nv)
    for weight,(bid,local,*_) in zip(weights,points):
        point=data.xpos[bid]+data.xmat[bid].reshape(3,3)@local;J=np.zeros((3,model.nv));jr=J.copy();mujoco.mj_jac(model,data,J,jr,point,bid)
        support+=J.T@np.array([0.,0.,fz*weight])
    feed=B.T@(data.qfrc_bias-support)
    gain=kp.copy()
    for i,name in enumerate(adapter.b.ind_names):
        if name.startswith(('ankle_angle','mtp_angle','subtalar')):gain[i]=40.
    controller.desired_cop=cop;controller.feasible_cop=A@weights
    result=feed[6:]+gain*(target-data.qpos[adapter.ids])
    # A hip strategy complements ankle/COP feedback. These are paired
    # internal torques; total-body angular momentum changes only by contact.
    R=Rotation.from_quat(np.roll(data.qpos[3:7],-1));desired=Rotation.from_rotvec([pitch,0.,initial_yaw]);error=(R.inv()*desired).as_rotvec()
    moment=500*error-100*data.qvel[3:6]
    hip=[(i,adapter.b.joints[name]) for i,name in enumerate(adapter.b.ind_names) if name.startswith(('hip_flexion','hip_adduction','hip_rotation'))]
    axes=np.column_stack([-R.as_matrix().T@data.xaxis[j] for i,j in hip])
    extra=axes.T@np.linalg.solve(axes@axes.T+1e-6*np.eye(3),moment)
    for (i,j),value in zip(hip,extra):result[i]+=value
    for side in ['r','l']:
        bid=model.body('calcn_'+side).id
        up=data.xmat[bid].reshape(3,3)@controller.foot_up_local[side]
        jr=np.zeros((3,model.nv));jp=jr.copy();mujoco.mj_jacBody(model,data,jp,jr,bid)
        omega=jr@data.qvel
        foot_moment=200*np.cross(up,[0.,0.,1.])-5*np.r_[omega[:2],0.]
        for name in ['ankle_angle_'+side,'subtalar_angle_'+side]:
            i=adapter.b.ind_names.index(name);j=adapter.b.joints[name];result[i]+=data.xaxis[j]@foot_moment
    return result


def torque(model,data,adapter,controller,target,pitch,B,Jcom,initial_yaw):
    ids=adapter.dofs;n=len(ids)
    C=data.subtree_com[1].copy();V=Jcom@data.qvel
    if not hasattr(controller,'native_impact'):
        controller.native_impact=(C.copy(),V.copy())
        controller.capture_omega=np.sqrt(9.80665/max(.3,C[2]-controller.height))
        controller.capture_goal_xy=C[:2]+V[:2]/controller.capture_omega
        from continue_jump_landing import Polynomial
        controller.absorb=Polynomial(C[2],V[2],-9.80665,max(controller.height+.58,C[2]-.20),0,0,.22)
        controller.recover=Polynomial(max(controller.height+.58,C[2]-.20),0,0,controller.stand_COM_z,0,0,1.)
    t=controller.elapsed;poly=controller.absorb if t<.22 else controller.recover;s=t if t<.22 else min(t-.22,1.)
    z,vz,az=[float(poly.value(s,k)[0]) for k in range(3)]
    om=controller.capture_omega
    acc=np.r_[om**2*(controller.capture_goal_xy-C[:2])-2*om*V[:2],az+100*(z-C[2])+20*(vz-V[2])]
    R=Rotation.from_quat(np.roll(data.qpos[3:7],-1));desired=Rotation.from_rotvec([pitch,0,initial_yaw]);err=(R.inv()*desired).as_rotvec()
    angular=80*err-18*data.qvel[3:6]
    joint=100*(target-data.qpos[adapter.ids])-20*data.qvel[ids]
    # Evaluate the actual acceleration map, including equality and contact response.
    jp=np.zeros((3,model.nv));jr=jp.copy()
    # COM convective acceleration from mass-weighted body COM Jacobian derivatives.
    conv=np.zeros(3);total=model.body_mass.sum()
    for bid in range(1,model.nbody):
        mujoco.mj_jacDot(model,data,jp,jr,data.xipos[bid],bid)
        conv+=model.body_mass[bid]*(jp@data.qvel)/total
    weights=np.r_[np.full(3,10.),np.full(3,3.),np.full(n,.2)]
    goal=weights*np.r_[acc,angular,joint]
    def response(x):
        data.qfrc_applied[:]=0.;data.qfrc_applied[ids]=x;mujoco.mj_forward(model,data)
        return weights*np.r_[Jcom@data.qacc+conv,data.qacc[3:6],data.qacc[ids]]
    x=getattr(controller,'native_previous_torque',np.zeros(n)).copy()
    for iteration in range(2):
        base=response(x);A=np.column_stack([response(x+np.eye(n)[j])-base for j in range(n)])
        # Penalize torque magnitude smoothly; no force ceiling or root actuator.
        delta=np.linalg.lstsq(np.vstack([A,.5*np.eye(n)]),np.r_[goal-base,-.5*x],rcond=1e-8)[0]
        x+=delta
    controller.native_previous_torque=x.copy()
    return x
