"""Native contact pilot: exact relative upper-body holds by removing their DOFs.

Fixed body transforms preserve each anatomical body's mass, COM and inertia.
This avoids soft constraints on near-massless shoulder coordinate helpers.
The source muscle/path model remains unchanged for later load reconstruction.
No body is fixed to world; only ideal lower-body joint torques are applied.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from continue_jump_landing import Landing,box_distance
from centroidal_jump import s5
from native_landing_control import torque as native_task_torque,pressure_feedback
ROOT=Path(__file__).resolve().parents[1]


def actual_contact_wrench(model,data):
    count=data.ncon
    if not count:return np.zeros(3),np.zeros(3),np.full(3,np.nan)
    contacts=data.contact;g1=contacts.geom1[:count];g2=contacts.geom2[:count];b1=model.geom_bodyid[g1];b2=model.geom_bodyid[g2];ids=np.flatnonzero((b1==0)!=(b2==0))
    if not len(ids):return np.zeros(3),np.zeros(3),np.full(3,np.nan)
    local=np.zeros((len(ids),6))
    for j,ci in enumerate(ids):mujoco.mj_contactForce(model,data,int(ci),local[j])
    frames=contacts.frame[:count].reshape(-1,3,3)[ids];positions=contacts.pos[:count][ids];sign=np.where(b1[ids]==0,1.,-1.)
    # contactForce is the force on geom2. Reverse every force/torque component
    # when the static world is geom2 (e.g. sphere against a box platform).
    forces=np.einsum('nji,nj->ni',frames,local[:,:3])*sign[:,None];couples=np.einsum('nji,nj->ni',frames,local[:,3:])*sign[:,None];force=forces.sum(0);moment=(np.cross(positions,forces)+couples).sum(0);pressure=(positions*np.maximum(forces[:,2],0)[:,None]).sum(0)
    return force,moment,pressure/force[2] if force[2]>1. else np.full(3,np.nan)


def build(r,q,v,points,dt,height,front):
    b,m,d=r.b,r.m,r.d;b.pose(q,v);X=d.xpos.copy();R=d.xmat.reshape(-1,3,3).copy();spec=b.spec.copy();spec.compiler.fusestatic=False
    pelvis=m.body('pelvis').id;lower=set()
    for bid in range(1,m.nbody):
        parent=bid
        while parent and parent!=pelvis:parent=int(m.body_parentid[parent])
        if parent==pelvis:lower.add(bid)
    lower_joints={m.joint(j).name for j in range(1,m.njnt) if int(m.jnt_bodyid[j]) in lower};upper=[j for j in range(1,m.njnt) if m.joint(j).name not in lower_joints]
    if max(abs(v[int(m.jnt_dofadr[j])]) for j in upper)>1e-4:raise RuntimeError('Upper body still has relative joint motion; cannot freeze without an impulse')
    for item in list(spec.actuators):spec.delete(item)
    for item in list(spec.sensors):spec.delete(item)
    for item in list(spec.tendons):spec.delete(item)
    for item in list(spec.keys):spec.delete(item)
    for item in list(spec.pairs):spec.delete(item)
    for item in list(spec.equalities):
        if item.name1 not in lower_joints or (item.name2 and item.name2 not in lower_joints) or item.name1.startswith('mtp_angle_'):spec.delete(item)
        else:item.solref=[.002,1.];item.solimp=[.99,.9999,.001,.5,2.]
    for body in spec.bodies:
        bid=m.body(body.name).id
        if bid==0:continue
        if bid not in lower and bid!=1:
            parent=int(m.body_parentid[bid]);owner=body.parent
            while not isinstance(owner,mujoco.MjsBody):owner=owner.parent
            identity=owner.add_frame(name='task_rigidframe_'+body.name,pos=[0.,0.,0.],quat=[1.,0.,0.,0.]);body.set_frame(identity)
            body.pos=R[parent].T@(X[bid]-X[parent]);body.quat=np.roll(Rotation.from_matrix(R[parent].T@R[bid]).as_quat(),1)
        body.explicitinertial=True;body.mass=float(m.body_mass[bid]);body.ipos=m.body_ipos[bid];body.iquat=m.body_iquat[bid];body.fullinertia=[np.nan,0,0,0,0,0];body.inertia=m.body_inertia[bid]
    for j in upper:spec.delete(spec.joint(m.joint(j).name))
    for geom in spec.geoms:geom.contype=geom.conaffinity=0
    b.pose(r.contact_stand_override)
    for i,(bid,local,p,fore) in enumerate(points):
        normal=d.xmat[bid].reshape(3,3).T@np.array([0.,0.,1.]);spec.body(m.body(bid).name).add_geom(name='task_sole_'+str(i),type=mujoco.mjtGeom.mjGEOM_SPHERE,pos=local+.01*normal,size=[.01,0,0],contype=1,conaffinity=1,condim=3,friction=[.8,.005,.0001],solref=[.004,1.],solimp=[.95,.99,.002,.5,2.],density=0.)
    spec.worldbody.add_geom(name='task_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[3.,3.,.1],contype=1,conaffinity=1,friction=[.8,.005,.0001],solref=[.004,1.])
    if height>0:spec.worldbody.add_geom(name='task_box',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[-.025,front-.375,height/2],size=[.4,.375,height/2],contype=1,conaffinity=1,friction=[.8,.005,.0001],solref=[.004,1.])
    native=spec.compile();assert [native.body(i).name for i in range(native.nbody)]==[m.body(i).name for i in range(m.nbody)]
    for attr in ['body_mass','body_inertia','body_iquat','body_ipos']:getattr(native,attr)[:]=getattr(m,attr)
    native.opt.gravity[:]=[0.,0.,-9.80665];native.opt.disableflags=0;native.opt.timestep=dt;native.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;native.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON;native.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC;native.opt.iterations=100;native.opt.tolerance=1e-10;native.dof_damping[:]=0;native.dof_frictionloss[:]=0;native.dof_armature[:]=1e-8;native.dof_armature[:6]=0;native.jnt_stiffness[:]=0;native.body_gravcomp[:]=0
    mujoco.mj_setConst(native,mujoco.MjData(native));nd=mujoco.MjData(native);nd.qpos[:7]=q[:7];nd.qvel[:6]=v[:6];mapping=[]
    for j in range(1,native.njnt):
        old=b.joints[native.joint(j).name];qa=int(native.jnt_qposadr[j]);da=int(native.jnt_dofadr[j]);oq=int(m.jnt_qposadr[old]);od=int(m.jnt_dofadr[old]);nd.qpos[qa]=q[oq];nd.qvel[da]=v[od];mapping.append((qa,da,oq,od))
    mujoco.mj_forward(native,nd);mujoco.mj_subtreeVel(native,nd);b.pose(q,v);mujoco.mj_subtreeVel(m,d)
    check=dict(body_position_max_difference_m=float(abs(nd.xpos-d.xpos).max()),body_rotation_max_difference=float(abs(nd.xmat-d.xmat).max()),COM_velocity_difference_m_s=float(np.linalg.norm(nd.subtree_linvel[1]-d.subtree_linvel[1])),H_difference_kg_m2_s=float(np.linalg.norm(nd.subtree_angmom[1]-d.subtree_angmom[1])),mass_difference_kg=float(abs(native.body_mass.sum()-m.body_mass.sum())))
    if max(check.values())>1e-6:raise RuntimeError(('Rigid upper conversion changed transition mechanics',check))
    dependent={int(native.eq_obj1id[i]) for i in range(native.neq)};ind=[j for j in range(1,native.njnt) if j not in dependent];names=[native.joint(j).name for j in ind];ids=np.array([native.jnt_qposadr[j] for j in ind]);dofs=np.array([native.jnt_dofadr[j] for j in ind]);B0=np.zeros((native.nv,6+len(ind)));B0[:6,:6]=np.eye(6);B0[dofs,6+np.arange(len(ind))]=1
    pending=list(range(native.neq));done=set(ind);eqs=[]
    while pending:
        available=[k for k in pending if int(native.eq_obj2id[k])<0 or int(native.eq_obj2id[k]) in done]
        if not available:raise RuntimeError('Cyclic native lower-body couplers')
        for k in available:eqs.append(k);done.add(int(native.eq_obj1id[k]));pending.remove(k)
    def reduction(q,u):
        B=B0.copy();curve=np.zeros(native.nv)
        for k in eqs:
            dep=int(native.eq_obj1id[k]);parent=int(native.eq_obj2id[k]);dd=int(native.jnt_dofadr[dep]);c=native.eq_data[k,:5]
            if parent<0:B[dd]=0;continue
            pd=int(native.jnt_dofadr[parent]);qa=int(native.jnt_qposadr[parent]);x=q[qa]-native.qpos0[qa];p1=c[1]+x*(2*c[2]+x*(3*c[3]+4*x*c[4]));p2=2*c[2]+x*(6*c[3]+12*x*c[4]);B[dd]=p1*B[pd];curve[dd]=p1*curve[pd]+p2*float(B[pd]@u)**2
        return B,curve
    def jacobian(bid,local):
        p=nd.xpos[bid]+nd.xmat[bid].reshape(3,3)@local;J=np.zeros((3,native.nv));Jr=J.copy();mujoco.mj_jac(native,nd,J,Jr,p,bid);return J,p
    adapter=SimpleNamespace(m=native,d=nd,n=6+len(ind),ids=ids,dofs=dofs,b=SimpleNamespace(ind_names=names,independent=ind,joints={native.joint(j).name:j for j in range(native.njnt)},jacobian=jacobian))
    return native,nd,adapter,reduction,mapping,check,spec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source');ap.add_argument('--output-dir',required=True)
    ap.add_argument('--height',type=float,default=0.);ap.add_argument('--front',type=float,default=-.14)
    ap.add_argument('--hz',type=int,default=2000);ap.add_argument('--seconds',type=float,default=2.)
    ap.add_argument('--controller',choices=['pressure','whole-body','native-response'],default='pressure')
    ap.add_argument('--hold-landing',action='store_true')
    ap.add_argument('--contact-time-constant',type=float,default=.02)
    args=ap.parse_args();folder=ROOT/args.source;out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
    source=np.load(folder/'motion.npz');meta=json.loads((folder/'summary.json').read_text());r=MeasuredJumpRig(87,1.78);b,m,d=r.b,r.m,r.d;r.contact_stand_override=source['qpos'][0].copy();points=contact_geometry(r)['points'];ts=source['times'];to=meta['phase_times_s']['end_ascent'];start=None
    for i in np.flatnonzero(ts>to+.05):
        P=source['contact_positions_m'][i];gap=min(float(P[:,2].min()),min(box_distance(p,args.front,args.height)[0] for p in P) if args.height>0 else np.inf)
        if gap<.035 and source['velocity'][i,2]<0:start=float(ts[i]-.008);break
    if start is None:raise RuntimeError('Flight does not reach a contact surface')
    pos=CubicSpline(ts,source['qpos'][:,:3],axis=0);angles=CubicSpline(ts,source['qpos'][:,r.ids],axis=0);rot=RotationSpline(ts,Rotation.from_quat(np.roll(source['qpos'][:,3:7],-1,axis=1)));vel=CubicSpline(ts,source['qvel'],axis=0);q=source['qpos'][0].copy();q[:3]=pos(start);q[3:7]=np.roll(rot(start).as_quat(),1);q[r.ids]=angles(start);q=b.complete(q);v=vel(start)
    native,nd,adapter,reduction,mapping,check,spec=build(r,q,v,points,1/args.hz,args.height,args.front);(out/'transition_check.json').write_text(json.dumps(check,indent=2));mujoco.mj_saveModel(native,str(out/'native_contact_model.mjb'));low,_=r.planted_pose(1.5,1.3,.09,arms=(np.pi/2,.1,.2),trunk=0.);names=adapter.b.ind_names;old_qids=[m.jnt_qposadr[b.joints[n]] for n in names];crouch=low[old_qids];stand=source['qpos'][0,old_qids];initial=q[old_qids];pitch_low=float(Rotation.from_quat(np.roll(low[3:7],-1)).as_rotvec()[0]);controller=Landing(adapter,points,args.front,args.height,1/args.hz);controller.stand_COM_z=args.height+float(source['com'][0,2]);contact_start=None;fallbacks=0;max_root=0.;max_balance=0.;penetration=0.;failure=None;front_collision=False;arr={k:[] for k in ['times','qpos','qvel','qacc','body_xpos','body_xmat','com','velocity','angular_momentum','force_world_N','contact_positions_m','mode']}
    controller.soft_contact_tracking=True;first_controller_failure=None
    use_impedance=args.controller=='pressure'
    controller.native_compliant_contact=True;controller.normal_tracking_weight=1e7;controller.tangent_tracking_weight=1e6
    kp=np.array([500. if n.startswith(('hip_flexion','knee_angle')) else 300. if n.startswith(('hip_adduction','hip_rotation','ankle_angle')) else 50. for n in names]);kd=np.array([30. if n.startswith('hip_') else 20. if n.startswith('knee_') else 2. for n in names])
    for gi in range(native.ngeom):
        if native.geom(gi).name.startswith('task_'):native.geom_solref[gi]=[args.contact_time_constant,1.]
    if use_impedance:native.dof_damping[adapter.dofs]=kd
    mujoco.mj_saveModel(native,str(out/'native_contact_model.mjb'))
    arr['actual_COP_world_m']=[];arr['desired_COP_world_m']=[];arr['feasible_COP_world_m']=[]
    max_applied_torque=0.;peak_actual_grf=0.
    initial_yaw=float(Rotation.from_quat(np.roll(q[3:7],-1)).as_rotvec()[2])
    controller.foot_up_local={s:source['body_xmat'][0,m.body('calcn_'+s).id].reshape(3,3).T@np.array([0.,0.,1.]) for s in ['r','l']}
    pitch_initial=float(Rotation.from_quat(np.roll(q[3:7],-1)).as_rotvec()[0])
    for step in range(round(args.seconds*args.hz)+1):
        t=step/args.hz;mujoco.mj_forward(native,nd);mujoco.mj_subtreeVel(native,nd);contacts=[];F=np.zeros(3)
        for ci in range(nd.ncon):
            c=nd.contact[ci];b1=int(native.geom_bodyid[c.geom1]);b2=int(native.geom_bodyid[c.geom2])
            if (b1==0)==(b2==0):continue
            bid=b2 if b1==0 else b1;frame=np.array(c.frame).reshape(3,3).copy()
            if b2==0:frame[0]*=-1;frame[2]*=-1
            local=nd.xmat[bid].reshape(3,3).T@(c.pos-nd.xpos[bid]);fc=np.zeros(6);mujoco.mj_contactForce(native,nd,ci,fc);F+=frame.T@fc[:3];contacts.append((bid,local,c.pos.copy(),frame,float(c.dist)));penetration=max(penetration,max(0.,-float(c.dist)))
            if args.height>0 and frame[0,2]<.9 and fc[0]>1.:front_collision=True
        if contact_start is None and np.linalg.norm(F)>1.:contact_start=t
        elapsed=max(0.,t-(contact_start if contact_start is not None else t));controller.elapsed=elapsed
        f=float(s5(elapsed/.20)) if elapsed<.20 else float(s5((elapsed-.20)/.9));target=(1-f)*initial+f*crouch if elapsed<.20 else (1-f)*crouch+f*stand;pitch=pitch_low if elapsed<.20 else (1-f)*pitch_low;u=np.r_[nd.qvel[:6],nd.qvel[adapter.dofs]];B,curve=reduction(nd.qpos,u);M=np.zeros((native.nv,native.nv));mujoco.mj_fullM(native,nd,M);Mr=B.T@M@B;bias=B.T@(nd.qfrc_bias+M@curve);Jcom=np.zeros((3,native.nv));mujoco.mj_jacSubtreeCom(native,nd,Jcom,1);P=np.array([nd.xpos[bid]+nd.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);a=100*(target-nd.qpos[adapter.ids])-20*u[6:];tau=Mr@np.r_[np.zeros(6),a]+bias;tau[:6]=0.
        pitch=(1-f)*pitch_initial+f*pitch_low if elapsed<.20 else (1-f)*pitch_low
        if args.hold_landing and elapsed>=.20:target=crouch.copy();pitch=pitch_low
        if contacts:
            operators=[];gaps=[];blocks=[];point_ids=[];local_points=[]
            for j,(bid,local,p,frame,gap) in enumerate(contacts):
                J=np.zeros((3,native.nv));Jr=J.copy();mujoco.mj_jac(native,nd,J,Jr,p,bid);order=frame[[1,2,0]];operators.extend(order@J@B);gaps.extend([0.,0.,gap]);blocks.append((3*j,3));point_ids.append((j,order,'native'));local_points.append((bid,local,p,True))
            previous=controller.points;controller.points=local_points
            try:
                if use_impedance:
                    # Damping is integrated implicitly by MuJoCo; explicit
                    # velocity feedback destabilizes the tiny toe inertia.
                    tau=np.r_[np.zeros(6),pressure_feedback(native,nd,adapter,controller,P,points,B,Jcom,target,kp,pitch,initial_yaw)]
                elif step%4==0 or not hasattr(controller,'native_previous_torque'):
                    if args.controller=='whole-body':tau=controller.stance_torque(nd.qpos.copy(),u,target,pitch,Mr,bias,B,curve,Jcom,P,operators,gaps,blocks,point_ids)
                    else:tau=np.r_[np.zeros(6),native_task_torque(native,nd,adapter,controller,target,pitch,B,Jcom,initial_yaw)]
                    controller.native_previous_torque=tau[6:].copy()
                else:tau=np.r_[np.zeros(6),controller.native_previous_torque]
            except (ValueError,RuntimeError) as e:
                fallbacks+=1
                if first_controller_failure is None:first_controller_failure=str(e)
            controller.points=previous
        nd.qfrc_applied[:]=0;nd.qfrc_applied[adapter.dofs]=tau[6:];max_root=max(max_root,float(abs(nd.qfrc_applied[:6]).max()));mujoco.mj_forward(native,nd);rn=np.zeros(native.nv);mujoco.mj_rne(native,nd,1,rn);max_balance=max(max_balance,float(abs(rn[:6]-nd.qfrc_constraint[:6]).max()))
        F,external_moment,actual_cop=actual_contact_wrench(native,nd)
        max_applied_torque=max(max_applied_torque,float(abs(nd.qfrc_applied[adapter.dofs]).max()));peak_actual_grf=max(peak_actual_grf,float(F[2]))
        if step%max(1,args.hz//125)==0:
            fullq=q.copy();fullv=np.zeros(m.nv);fulla=fullv.copy();fullq[:7]=nd.qpos[:7];fullv[:6]=nd.qvel[:6];fulla[:6]=nd.qacc[:6]
            for qa,da,oq,od in mapping:fullq[oq]=nd.qpos[qa];fullv[od]=nd.qvel[da];fulla[od]=nd.qacc[da]
            mujoco.mj_subtreeVel(native,nd);values=dict(times=start+t,qpos=fullq,qvel=fullv,qacc=fulla,body_xpos=nd.xpos.copy(),body_xmat=nd.xmat.copy(),com=nd.subtree_com[1].copy(),velocity=nd.subtree_linvel[1].copy(),angular_momentum=nd.subtree_angmom[1].copy(),force_world_N=F.copy(),contact_positions_m=P.copy(),mode=3 if contacts else 2)
            for key,value in values.items():arr[key].append(value)
            arr['actual_COP_world_m'].append(actual_cop)
            arr['desired_COP_world_m'].append(np.r_[getattr(controller,'desired_cop',[np.nan,np.nan]),args.height])
            arr['feasible_COP_world_m'].append(np.r_[getattr(controller,'feasible_cop',[np.nan,np.nan]),args.height])
        if step%(args.hz//4)==0:print('RIGID_UPPER_NATIVE',round(t,3),'COM',np.round(nd.subtree_com[1],3),'V',np.round(nd.subtree_linvel[1],2),'Fz',round(float(F[2])),flush=True)
        if nd.qpos[2]<args.height+.2:failure='Body fell outside the foot-contact-only model domain';break
        mujoco.mj_step(native,nd)
    result={k:np.array(v) for k,v in arr.items()};np.savez_compressed(out/'landing_motion.npz',**result);mujoco.mj_forward(native,nd);mujoco.mj_subtreeVel(native,nd);P=np.array([nd.xpos[bid]+nd.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);feet=bool(np.max(abs(P[:,2]-args.height))<.025)
    if args.height>0:feet=feet and bool(P[:,1].max()<args.front+.005 and P[:,1].min()>args.front-.755)
    report=dict(status='native_contact_candidate' if failure is None else 'failed_native_contact',failure=failure,front_collision=front_collision,box_height_m=args.height,contact_start_s=None if contact_start is None else start+contact_start,completed_supported_landing=bool(failure is None and not front_collision and feet and np.linalg.norm(nd.subtree_linvel[1])<.1),feet_on_support=feet,final_COM_velocity_m_s=nd.subtree_linvel[1].tolist(),maximum_root_applied_force_or_torque=max_root,root_inverse_dynamics_residual_max_SI=max_balance,maximum_contact_penetration_m=penetration,controller_fallback_steps=fallbacks,transition_check=check,relative_upper_body_hold='Exact fixed relative transforms; no world weld; zero initial relative upper joint speed required.',contact_model='Native MuJoCo elliptic friction, 4 ms uncalibrated contact regularization.',muscle_paths='Retained in the source analysis model; native contact pilot uses skeletal inertia and ideal joint torque controls.',muscle_force_analysis_allowed=False,human_motion_validated=False)
    report['first_controller_failure']=first_controller_failure
    report['peak_actual_GRF_N']=peak_actual_grf
    report['maximum_applied_internal_joint_torque_Nm']=max_applied_torque
    report['strength_ceiling']=False
    report['pressure_feedback']='Desired COP projected into the sole polygon; actual COP recorded separately from native contact forces.'
    report['contact_model']=f'Native MuJoCo elliptic friction, {args.contact_time_constant*1000:g} ms uncalibrated sole-contact regularization.'
    report['controller']=args.controller
    report['hold_crouched_landing']=args.hold_landing
    report['contact_acceleration_tracking']='Internal joint impedance and COP feedback.' if use_impedance else 'Whole-body inverse dynamics soft contact tracking.' if args.controller=='whole-body' else 'Native contact acceleration response linearization.'
    (out/'landing_status.json').write_text(json.dumps(report,indent=2));print('RIGID_UPPER_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
