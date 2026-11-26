"""Native MuJoCo contact cross-check with fixed relative upper-body joints.

The current pilot keeps the arms/torso joint angles at touchdown. Their masses,
geometry and inertia remain present. No joint is welded to the world and no
root force/torque is applied. Lower-body native polynomial couplers are kept.
Contact regularization is numerical and is not calibrated human foot tissue.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from continue_jump_landing import Landing,box_distance
from centroidal_jump import s5
ROOT=Path(__file__).resolve().parents[1]


def compile_contact_model(r,q,points,dt,height,front,radius=.01):
    source=r.b.spec.copy();m=r.m
    for body in source.bodies:
        bid=m.body(body.name).id
        if bid==0:continue
        body.explicitinertial=True;body.mass=float(m.body_mass[bid]);body.ipos=m.body_ipos[bid];body.iquat=m.body_iquat[bid];body.fullinertia=[np.nan,0,0,0,0,0];body.inertia=m.body_inertia[bid]
    for geom in source.geoms:geom.contype=geom.conaffinity=0
    # Explicit source pairs bypass contype/conaffinity filtering. They belong
    # to the source collision envelopes, not this foot-contact pilot.
    for pair in list(source.pairs):source.delete(pair)
    lower_bodies=set();pelvis=m.body('pelvis').id
    for bid in range(1,m.nbody):
        ancestor=bid
        while ancestor and ancestor!=pelvis:ancestor=int(m.body_parentid[ancestor])
        if ancestor==pelvis:lower_bodies.add(bid)
    lower_names={m.joint(j).name for j in range(1,m.njnt) if int(m.jnt_bodyid[j]) in lower_bodies}
    for i,eq in enumerate(source.equalities):
        if eq.name1 not in lower_names:eq.active=False
        elif eq.name1.startswith('mtp_angle_'):eq.active=False
        else:eq.active=bool(m.eq_active0[i])
    frozen=[]
    for j in range(1,m.njnt):
        name=m.joint(j).name
        if name in lower_names:continue
        value=float(q[m.jnt_qposadr[j]]-m.qpos0[m.jnt_qposadr[j]])
        source.add_equality(name='landing_hold_'+name,type=mujoco.mjtEq.mjEQ_JOINT,objtype=mujoco.mjtObj.mjOBJ_JOINT,name1=name,data=[value]+[0.]*11,solref=[.002,1.],solimp=[.99,.9999,.001,.5,2.])
        frozen.append(name)
    r.b.pose(r.contact_stand_override)
    for i,(bid,local,p0,fore) in enumerate(points):
        normal=r.d.xmat[bid].reshape(3,3).T@np.array([0.,0.,1.]);source.body(m.body(bid).name).add_geom(name='task_sole_'+str(i),type=mujoco.mjtGeom.mjGEOM_SPHERE,pos=local+radius*normal,size=[radius,0,0],contype=1,conaffinity=1,condim=3,friction=[.8,.005,.0001],solref=[.004,1.],solimp=[.95,.99,.002,.5,2.],density=0.)
    source.worldbody.add_geom(name='task_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[3.,3.,.1],contype=1,conaffinity=1,friction=[.8,.005,.0001],solref=[.004,1.],solimp=[.95,.99,.002,.5,2.])
    if height>0:source.worldbody.add_geom(name='task_box',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[-.025,front-.375,height/2],size=[.4,.375,height/2],contype=1,conaffinity=1,friction=[.8,.005,.0001],solref=[.004,1.],solimp=[.95,.99,.002,.5,2.])
    native=source.compile();native.opt.gravity[:]=m.opt.gravity;native.opt.disableflags=int(m.opt.disableflags)&~(int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT));native.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_ACTUATION);native.opt.timestep=dt;native.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;native.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON;native.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC;native.opt.iterations=100;native.opt.tolerance=1e-10
    native.dof_damping[:]=m.dof_damping;native.dof_frictionloss[:]=0;native.dof_armature[:]=m.dof_armature;native.jnt_stiffness[:]=0;native.tendon_stiffness[:]=0;native.tendon_damping[:]=0;native.body_gravcomp[:]=0
    # Preserve the already audited physical profile after compilation; near
    # degenerate source tensors can trigger compiler inertia balancing again.
    native.body_mass[:]=m.body_mass;native.body_inertia[:]=m.body_inertia;native.body_iquat[:]=m.body_iquat;native.body_ipos[:]=m.body_ipos
    mujoco.mj_setConst(native,mujoco.MjData(native))
    assert native.nq==m.nq and native.nv==m.nv
    assert np.max(abs(native.body_mass-m.body_mass))<1e-10
    assert np.max(abs(native.body_inertia-m.body_inertia))<1e-10
    return native,frozen,source


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('--output-dir',required=True);ap.add_argument('--height',type=float,default=0.);ap.add_argument('--front',type=float,default=-.14);ap.add_argument('--hz',type=int,default=2000);ap.add_argument('--seconds',type=float,default=2.);args=ap.parse_args();folder=ROOT/args.source;out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
    data=np.load(folder/'motion.npz');meta=json.loads((folder/'summary.json').read_text());r=MeasuredJumpRig(87,1.78);b,m,d=r.b,r.m,r.d;ids=np.array(r.ids);ts=data['times'];to=meta['phase_times_s']['end_ascent'];r.contact_stand_override=data['qpos'][0].copy();points=contact_geometry(r)['points']
    # Start before contact; native sphere geometry and the native solver choose
    # the actual touchdown. Do not translate the body onto a platform.
    start=None
    for i in np.flatnonzero(ts>to+.05):
        p=data['contact_positions_m'][i]
        gap=min(float(p[:,2].min()),min(box_distance(v,args.front,args.height)[0] for v in p) if args.height>0 else np.inf)
        if gap<.035 and data['velocity'][i,2]<0:start=max(to+.05,float(ts[i]-.008));break
    if start is None:raise RuntimeError('Available flight does not approach a surface')
    pos=CubicSpline(ts,data['qpos'][:,:3],axis=0);angles=CubicSpline(ts,data['qpos'][:,ids],axis=0);rot=RotationSpline(ts,Rotation.from_quat(np.roll(data['qpos'][:,3:7],-1,axis=1)));vel=CubicSpline(ts,data['qvel'],axis=0)
    q=data['qpos'][0].copy();q[:3]=pos(start);q[3:7]=np.roll(rot(start).as_quat(),1);q[ids]=angles(start);q=b.complete(q);v=vel(start)
    native,frozen,spec=compile_contact_model(r,q,points,1/args.hz,args.height,args.front);nd=mujoco.MjData(native);nd.qpos[:]=q;nd.qvel[:]=v;mujoco.mj_forward(native,nd)
    equality=np.flatnonzero(nd.efc_type[:nd.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY);initial_error=float(abs(nd.efc_pos[equality]).max()) if len(equality) else 0.
    if len(equality)==0:raise RuntimeError('Native relative joint constraints are disabled')
    if initial_error>1e-5:raise RuntimeError(('Native joint constraints do not match initial state',initial_error))
    (out/'native_contact_model.xml').write_text(spec.to_xml());low,_=r.planted_pose(1.5,1.3,.09,arms=(np.pi/2,.1,.2),trunk=0.);crouch=low[ids];stand=data['qpos'][0,ids];controller=Landing(r,points,args.front,args.height,1/args.hz);controller.stand_COM_z=args.height+float(data['com'][0,2]);controller.frozen_controller_joints=[n for n in b.ind_names if n in frozen];pitch_low=float(Rotation.from_quat(np.roll(low[3:7],-1)).as_rotvec()[0]);contact_start=None;fallbacks=0;max_root=0.;max_rne=0.;penetration=0.;arr={k:[] for k in ['times','qpos','qvel','qacc','body_xpos','body_xmat','com','velocity','angular_momentum','force_world_N','contact_positions_m','mode']};failure=None;max_hold=0.;forces=np.zeros(3)
    for step in range(round(args.seconds*args.hz)+1):
        t=step/args.hz;mujoco.mj_forward(native,nd);mujoco.mj_subtreeVel(native,nd);world_contacts=[];F=np.zeros(3)
        for ci in range(nd.ncon):
            c=nd.contact[ci];body1=int(native.geom_bodyid[c.geom1]);body2=int(native.geom_bodyid[c.geom2])
            if (body1==0)==(body2==0):continue
            bid=body2 if body1==0 else body1;frame=np.array(c.frame).reshape(3,3).copy()
            if body2==0:frame[0]*=-1;frame[2]*=-1
            local=nd.xmat[bid].reshape(3,3).T@(c.pos-nd.xpos[bid]);force=np.zeros(6);mujoco.mj_contactForce(native,nd,ci,force);F+=frame.T@force[:3];world_contacts.append((bid,local,c.pos.copy(),frame,float(c.dist)));penetration=max(penetration,max(0.,-float(c.dist)))
        if contact_start is None and np.linalg.norm(F)>1.:contact_start=t
        elapsed=max(0.,t-(contact_start if contact_start is not None else t));controller.elapsed=elapsed
        f=float(s5(elapsed/.20)) if elapsed<.20 else float(s5((elapsed-.20)/.9));target=(1-f)*q[ids]+f*crouch if elapsed<.20 else (1-f)*crouch+f*stand;pitch=pitch_low if elapsed<.20 else (1-f)*pitch_low
        ur=np.r_[nd.qvel[:6],nd.qvel[r.dofs]];B,curve=r.set_state(nd.qpos.copy(),ur);M=np.zeros((native.nv,native.nv));mujoco.mj_fullM(native,nd,M);Mr=B.T@M@B;bias=B.T@(nd.qfrc_bias+M@curve);Jcom=np.zeros((3,native.nv));mujoco.mj_jacSubtreeCom(native,nd,Jcom,1);P=np.array([nd.xpos[bid]+nd.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);a=100*(target-nd.qpos[ids])-20*ur[6:];tau=Mr@np.r_[np.zeros(6),a]+bias;tau[:6]=0.
        if world_contacts:
            operators=[];gaps=[];blocks=[];point_ids=[];local_points=[]
            for j,(bid,local,p,frame,gap) in enumerate(world_contacts):
                J=np.zeros((3,native.nv));Jr=J.copy();mujoco.mj_jac(native,nd,J,Jr,p,bid);order=frame[[1,2,0]];blocks.append((3*j,3));operators.extend(order@J@B);gaps.extend([0.,0.,gap]);point_ids.append((j,order,'native'));local_points.append((bid,local,p,True))
            original=controller.points;controller.points=local_points
            try:tau=controller.stance_torque(nd.qpos.copy(),ur,target,pitch,Mr,bias,B,curve,Jcom,P,operators,gaps,blocks,point_ids)
            except (ValueError,RuntimeError):fallbacks+=1
            controller.points=original
        nd.qfrc_applied[:]=0
        for i,n in enumerate(b.ind_names):
            if n not in frozen:nd.qfrc_applied[r.dofs[i]]=tau[6+i]
        max_root=max(max_root,float(abs(nd.qfrc_applied[:6]).max()));mujoco.mj_forward(native,nd);temp=np.zeros(native.nv);mujoco.mj_rne(native,nd,1,temp);max_rne=max(max_rne,float(abs(temp[:6]-nd.qfrc_constraint[:6]).max()))
        for name in frozen:
            adr=m.jnt_qposadr[b.joints[name]];max_hold=max(max_hold,abs(float(nd.qpos[adr]-q[adr])))
        if step%max(1,args.hz//125)==0:
            mujoco.mj_subtreeVel(native,nd);values=dict(times=start+t,qpos=nd.qpos.copy(),qvel=nd.qvel.copy(),qacc=nd.qacc.copy(),body_xpos=nd.xpos.copy(),body_xmat=nd.xmat.copy(),com=nd.subtree_com[1].copy(),velocity=nd.subtree_linvel[1].copy(),angular_momentum=nd.subtree_angmom[1].copy(),force_world_N=F.copy(),contact_positions_m=P.copy(),mode=3 if world_contacts else 2)
            for key,value in values.items():arr[key].append(value)
        if step%(args.hz//4)==0:print('NATIVE_LANDING',round(t,3),'COM',np.round(nd.subtree_com[1],3),'V',np.round(nd.subtree_linvel[1],2),'Fz',round(float(F[2])),flush=True)
        if nd.qpos[2]<args.height+.20:failure='Pelvis below viable posture; non-foot body contact is outside this pilot.';break
        mujoco.mj_step(native,nd)
    arrays={key:np.array(value) for key,value in arr.items()};np.savez_compressed(out/'landing_motion.npz',**arrays);mujoco.mj_forward(native,nd);mujoco.mj_subtreeVel(native,nd);P=np.array([nd.xpos[bid]+nd.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);feet=bool(np.max(abs(P[:,2]-args.height))<.025)
    if args.height>0:feet=feet and bool(P[:,1].max()<args.front+.005 and P[:,1].min()>args.front-.755)
    report=dict(status='native_contact_landing_candidate' if failure is None else 'failed_native_landing',failure=failure,native_mujoco_version=mujoco.__version__,contact_start_s=None if contact_start is None else start+contact_start,box_height_m=args.height,completed_supported_landing=bool(failure is None and feet and np.linalg.norm(nd.subtree_linvel[1])<.1),feet_on_support=feet,final_COM_velocity_m_s=nd.subtree_linvel[1].tolist(),initial_equality_error=initial_error,maximum_root_actuator_force_or_torque=max_root,root_inverse_dynamics_residual_max_SI=max_rne,maximum_contact_penetration_m=penetration,maximum_upper_joint_hold_error_rad=max_hold,controller_fallback_steps=fallbacks,upper_body='Relative joint angles held at pre-contact values; masses and inertias retained. No world weld.',contact_model='Native elliptic friction cone; 4 ms contact regularization, not a calibrated tissue model.',muscle_force_analysis_allowed=False,human_motion_validated=False)
    (out/'landing_status.json').write_text(json.dumps(report,indent=2));print('NATIVE_LANDING_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
