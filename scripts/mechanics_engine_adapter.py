"""Kinematics adapter for independent Newton-Euler verification.

Kinematics and inertial parameters come from MuJoCo. Body force/moment assembly
is independent. This cross-check does not independently validate anatomy.
"""
import numpy as np,mujoco
from newton_euler_reference import body_wrenches,external_wrenches,accumulate,at_point

def sample(rig,q,velocity,acceleration,contacts=()):
    m,d=rig.m,rig.d;rig.b.pose(q,velocity,acceleration);nb=m.nbody
    V=np.zeros((nb,3));W=V.copy();A=V.copy();Alpha=V.copy();I=np.zeros((nb,3,3));Js=[];Jrs=[]
    for bid in range(nb):
        J=np.zeros((3,m.nv));Jr=J.copy();Jdot=J.copy();Jrdot=J.copy();mujoco.mj_jacBodyCom(m,d,J,Jr,bid);mujoco.mj_jacDot(m,d,Jdot,Jrdot,d.xipos[bid],bid)
        V[bid]=J@velocity;W[bid]=Jr@velocity;A[bid]=J@acceleration+Jdot@velocity;Alpha[bid]=Jr@acceleration+Jrdot@velocity;rot=d.ximat[bid].reshape(3,3);I[bid]=rot@np.diag(m.body_inertia[bid])@rot.T;Js.append(J);Jrs.append(Jr)
    own=body_wrenches(m.body_mass,d.xipos,I,A,W,Alpha,m.opt.gravity);ext=external_wrenches(nb,contacts);net=own-ext;sub=accumulate(m.body_parentid,net);gen=np.zeros(m.nv)
    for i in range(nb):gen+=Js[i].T@net[i,:3]+Jrs[i].T@(net[i,3:]-np.cross(d.xipos[i],net[i,:3]))
    engine=np.zeros(m.nv);mujoco.mj_rne(m,d,1,engine)
    external_gen=np.zeros(m.nv)
    for bid,p,f,moment in contacts:
        J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,p,bid);external_gen+=J.T@f+Jr.T@moment
    body_gen=gen+external_gen;cuts={};cut_error=0.
    for jid in range(1,m.njnt):
        bid=int(m.jnt_bodyid[jid]);w=at_point(sub[bid],d.xanchor[jid]);adr=int(m.jnt_dofadr[jid]);kind=m.jnt_type[jid]
        if kind==mujoco.mjtJoint.mjJNT_HINGE:value=float(d.xaxis[jid]@w[3:])
        elif kind==mujoco.mjtJoint.mjJNT_SLIDE:value=float(d.xaxis[jid]@w[:3])
        else:continue
        cut_error=max(cut_error,abs(value-gen[adr]));cuts[m.joint(jid).name]={'proximal_force_world_N':w[:3].tolist(),'proximal_total_moment_world_Nm':w[3:].tolist(),'generalized_component':value}
    return {'body_wrenches':own,'net_wrenches':net,'subtree_wrenches':sub,'generalized':gen,'engine_body_generalized':engine,'crosscheck_max':float(abs(body_gen-engine).max()),'cut_vs_virtual_work_max':float(cut_error),'cuts':cuts,'com':d.xipos.copy(),'velocity':V,'omega':W,'acceleration':A,'alpha':Alpha,'inertia_world':I,'external_generalized':external_gen}

def path_speeds(rig):
    # Actuation-disabled mj_forward zeros actuator_velocity. Its geometric
    # derivative is still available from the native transmission Jacobian.
    return -rig.b.moments().T@rig.d.qvel
