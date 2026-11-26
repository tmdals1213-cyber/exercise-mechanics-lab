"""Contact-driven jump: solve support forces with floating-base acceleration.

Posture control defines an illustrative task, not observed human kinematics.
Support/COP are outputs. Flight uses joint-controlled foot clearance with
zero ground support. This reference stops before touchdown.
"""
import argparse,json,hashlib
import numpy as np,mujoco,quadprog
from scipy.linalg import qr
from scipy.spatial.transform import Rotation
from centroidal_jump import s5, Polynomial
from rigid_baseline_profile import RigidBaselineRig
from jump_rig import ROOT

def contact_geometry(rig):
    b,m,d=rig.b,rig.m,rig.d;scale=b.report['global_normalization_scale'];stand=getattr(rig,'contact_stand_override',None)
    if stand is None:stand=rig.planted_pose(.05,.12,.145,arms=(1.4,.25))[0]
    stand=stand.copy();b.pose(stand);feet={};points=[];ground=float(np.mean([d.site_xpos[m.site('subject_sole_reference_'+s).id,2] for s in ['r','l']]))
    for side in ['r','l']:
        heel=m.body('calcn_'+side).id;toe=m.body('toes_'+side).id;sid=m.site('subject_sole_reference_'+side).id;origin=d.site_xpos[sid].copy();ball=d.xanchor[b.joints['mtp_angle_'+side]].copy()
        feet[side]=dict(bid=heel,toe_bid=toe,local=m.site_pos[sid].copy(),p=origin.copy(),R=d.xmat[heel].reshape(3,3).copy())
        for bid,y,fore in [(heel,origin[1]+.010*scale,False),(toe,ball[1],True),(toe,origin[1]-.245*scale,True)]:
            for x in [-.037,.037]:
                p=np.array([origin[0]+x*scale,y,ground]);local=d.xmat[bid].reshape(3,3).T@(p-d.xpos[bid]);points.append((bid,local,p,fore))
    return dict(stand=stand,feet=feet,points=points,ground=ground)


class ContactJump:
    def __init__(self,rig,hz):
        self.rig=rig;self.b=rig.b;self.m=rig.m;self.d=rig.d;self.hz=hz;self.n=rig.n;self.W=float(self.m.body_mass.sum())*9.80665
        self.p=contact_geometry(rig);self.stand=self.p['stand'].copy();self.low=rig.planted_pose(.85,1.25,.145,arms=(1.4,.12),trunk=.10)[0]
        self.ext=self.stand.copy();rig.arm(self.ext,1.4,1.0,.25)
        for side in ['r','l']:
            for name,v in [('hip_flexion',.05),('knee_angle',.10),('ankle_angle',-.35),('mtp_angle',-.35)]:self.ext[self.m.jnt_qposadr[self.b.joints[name+'_'+side]]]=v
        self.air=self.stand.copy();rig.arm(self.air,1.4,1.0,.3)
        for side in ['r','l']:
            for name,v in [('hip_flexion',.15),('knee_angle',.50),('ankle_angle',-.05),('mtp_angle',-.05)]:self.air[self.m.jnt_qposadr[self.b.joints[name+'_'+side]]]=v
        self.bottom=.85;self.takeoff=1.10;self.toe=.94;self.landing=None;self.impacts=[];self.air_poly=None;self.actual_takeoff=None;self.flight_feet=None
        self.b.pose(self.stand);c0=self.d.subtree_com[1].copy();toey=np.mean([p[2][1] for p in self.p['points'] if p[3]])
        self.forward_T=self.toe-.25;self.forward=Polynomial(c0[1],0,0,toey+.005,0,0,self.forward_T);self.c0=c0
        self.trunk_bid=self.m.body('torso').id;self.Rstand=self.d.xmat[self.trunk_bid].reshape(3,3).copy();self.b.pose(self.b.q0);self.Rupright=self.d.xmat[self.trunk_bid].reshape(3,3).copy();self.theta_stand=float(Rotation.from_matrix(self.Rstand@self.Rupright.T).as_rotvec()[0]);self.b.pose(self.stand)

    def reference(self,t):
        if self.air_poly is not None:
            # Preserve the launch velocity/acceleration in centered derivative
            # probes; lower-clamping here creates a spurious v/epsilon spike.
            return self.air_poly.value(min(t-self.actual_takeoff,self.air_transition_T))
        if t<.25:return self.stand.copy()
        if t<self.bottom:return self.stand+s5((t-.25)/(self.bottom-.25))*(self.low-self.stand)
        if t<self.takeoff:return self.low+s5((t-self.bottom)/(self.takeoff-self.bottom))*(self.ext-self.low)
        if self.landing is None:
            return self.ext+s5((t-self.takeoff)/.24)*(self.air-self.ext)
        return self.air+s5((t-self.landing)/.9)*(self.stand-self.air)

    def solve(self,q,u,t,mode):
        r,m,d=self.rig,self.m,self.d;B,curve=r.set_state(q,u);n=r.n;M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M)
        released=getattr(self,'released_toe_points',set())
        active=[] if mode=='flight' else [i for i,p in enumerate(self.p['points']) if (mode!='toe' or p[3]) and i not in released]
        js=[];points=[]
        for bid,local,_,_ in self.p['points']:
            J,p=self.b.jacobian(bid,local);js.append(J);points.append(p)
        na=len(active);nv=n+3*na;rows=[];rhs=[];soft=[];root=np.zeros((6,nv));root[:,:n]=(M@B)[:6]
        for k,i in enumerate(active):root[:,n+3*k:n+3*k+3]=-self.W*js[i].T[:6]
        rows.append(root);rhs.append((d.qfrc_passive-d.qfrc_bias-M@curve)[:6])
        if active:
            # A declared balance-control strategy: choose contact moment to
            # keep sagittal whole-body angular momentum near zero. This is
            # delivered by the same solved ground forces, never a root motor.
            row=np.zeros((3,nv));com=d.subtree_com[1]
            for k,i in enumerate(active):
                x,y,z=points[i]-com;row[:,n+3*k:n+3*k+3]=self.W*np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            momentum_goal=self.ground_momentum_target(t) if hasattr(self,'ground_momentum_target') else -8*d.subtree_angmom[1]
            if getattr(self,'soft_ground_momentum_target',False):
                axis=np.sqrt(np.array(getattr(self,'ground_momentum_axis_weights',[1.,1.,1.])))
                soft.append((axis[:,None]*row,axis*momentum_goal,getattr(self,'ground_momentum_weight',100.)/self.W**2))
            else:rows.append(row);rhs.append(momentum_goal)
            equal=np.zeros((1,nv))
            for k,i in enumerate(active):equal[0,n+3*k+2]=1 if self.m.body(self.p['points'][i][0]).name.endswith('_r') else -1
            if getattr(self,'symmetric_vertical_reactions',True):rows.append(equal);rhs.append(np.zeros(1))
            contact_start=sum(len(v) for v in rhs)
            if getattr(self,'rigid_contact_frames',False):
                groups={}
                for i in active:groups.setdefault(self.p['points'][i][0],[]).append(i)
                if mode=='plant':
                    # A flat articulated foot has seven constrained DOFs:
                    # heel rigid pose plus its MTP angle. Cartesian constraints
                    # at every sole vertex are redundant off the exact manifold.
                    groups={m.body('calcn_'+s).id:[] for s in ['r','l']}
                for bid,indices in groups.items():
                    R0,p0=self.contact_frame_reference[bid]
                    if mode=='plant':local=np.zeros(3);target=p0;rot_rows=np.eye(3)
                    else:
                        local=np.mean([self.p['points'][i][1] for i in indices],axis=0);targets=np.array([self.p['points'][i][2] for i in indices]);target=targets.mean(0)
                        if len(indices)==1:rot_rows=np.empty((0,3))
                        else:
                            _,sv,Vt=np.linalg.svd(targets-target,full_matrices=True)
                            if len(sv)>1 and sv[1]>1e-6:rot_rows=np.eye(3)
                            else:rot_rows=Vt[1:]
                    J,p=self.b.jacobian(bid,local);Jr=J.copy();Jd=J.copy();Jrd=J.copy();mujoco.mj_jac(m,d,J,Jr,p,bid);mujoco.mj_jacDot(m,d,Jd,Jrd,p,bid)
                    row=np.zeros((3,nv));row[:,:n]=J@B;rows.append(row);rhs.append(-30*(J@d.qvel)+225*(target-p)-Jd@d.qvel-J@curve)
                    if len(rot_rows):
                        err=Rotation.from_matrix(R0@d.xmat[bid].reshape(3,3).T).as_rotvec();row=np.zeros((len(rot_rows),nv));row[:,:n]=rot_rows@Jr@B
                        rows.append(row);rhs.append(rot_rows@(225*err-30*(Jr@d.qvel)-Jrd@d.qvel-Jr@curve))
                    if mode=='plant':
                        side=m.body(bid).name[-1];i=self.b.ind_names.index('mtp_angle_'+side);col=6+i;row=np.zeros((1,nv));row[0,col]=1
                        rows.append(row);rhs.append(np.array([400*(self.stand[r.ids[i]]-q[r.ids[i]])-40*u[col]]))
            else:
                for i in active:
                    bid,local,target,_=self.p['points'][i];J,p=self.b.jacobian(bid,local);Jd=np.zeros_like(J);Jrd=np.zeros_like(J);mujoco.mj_jacDot(m,d,Jd,Jrd,p,bid)
                    row=np.zeros((3,nv));row[:,:n]=J@B;rows.append(row);rhs.append(-30*(J@d.qvel)+225*(target-p)-Jd@d.qvel-J@curve)
            contact_end=sum(len(v) for v in rhs)
        elif self.flight_feet is not None:
            ss=t-self.actual_takeoff;T=self.flight_T;ratio=np.clip(ss/T,0,1);height=self.clearance_height;bump=16*ratio*ratio*(1-ratio)**2;db=32*ratio*(1-ratio)*(1-2*ratio)/T;ddb=(32-192*ratio+192*ratio*ratio)/T**2
            for bid,local,poly,Rtoe in self.flight_feet:
                target=poly.value(ss);tv=poly.value(ss,1);ta=poly.value(ss,2);target[2]+=height*bump;tv[2]+=height*db;ta[2]+=height*ddb
                J,p=self.b.jacobian(bid,local);Jd=np.zeros_like(J);Jrd=np.zeros_like(J);mujoco.mj_jacDot(m,d,Jd,Jrd,p,bid);row=np.zeros((3,nv));row[:,:n]=J@B;soft.append((row,ta+30*(tv-J@d.qvel)+225*(target-p)-Jd@d.qvel-J@curve,getattr(self,'flight_point_weight',500.)))
                Jr=np.zeros_like(J);mujoco.mj_jac(m,d,J,Jr,d.xpos[bid],bid);mujoco.mj_jacDot(m,d,Jd,Jrd,d.xpos[bid],bid);err=Rotation.from_matrix(Rtoe@d.xmat[bid].reshape(3,3).T).as_rotvec();row=np.zeros((3,nv));row[:,:n]=Jr@B;soft.append((row,100*err-20*(Jr@d.qvel)-Jrd@d.qvel-Jr@curve,getattr(self,'flight_orientation_weight',50.)))
            if hasattr(self,'flight_knee_half_width_m'):
                for side in ['r','l']:
                    jid=self.b.joints['knee_angle_'+side];bid=m.jnt_bodyid[jid];point=d.xanchor[jid]
                    J=np.zeros((3,m.nv));Jr=J.copy();Jd=J.copy();Jrd=J.copy();mujoco.mj_jac(m,d,J,Jr,point,bid);mujoco.mj_jacDot(m,d,Jd,Jrd,point,bid)
                    target=self.c0[0]+(-1 if side=='r' else 1)*self.flight_knee_half_width_m
                    row=np.zeros((1,nv));row[0,:n]=(J@B)[0];value=80*(target-point[0])-18*(J@d.qvel)[0]-(Jd@d.qvel+J@curve)[0]
                    soft.append((row,np.array([value]),1000.))
            for bid,local,poly in getattr(self,'flight_knees',[]):
                ss=t-self.actual_takeoff;tt=min(ss,self.flight_T*.75)
                target=self.d.subtree_com[1]+poly.value(tt)
                tv=self.d.subtree_linvel[1]+(poly.value(tt,1) if ss<self.flight_T*.75 else 0)
                ta=np.array([0.,0.,-9.80665])+(poly.value(tt,2) if ss<self.flight_T*.75 else 0)
                J,p=self.b.jacobian(bid,local);Jd=np.zeros_like(J);Jrd=np.zeros_like(J);mujoco.mj_jacDot(m,d,Jd,Jrd,p,bid)
                row=np.zeros((3,nv));row[:,:n]=J@B
                soft.append((row,ta+30*(tv-J@d.qvel)+225*(target-p)-Jd@d.qvel-J@curve,getattr(self,'flight_knee_position_weight',10000.)))
        # Optional prescribed internal joint coordination (e.g. no arm swing).
        # These rows constrain joint acceleration, never apply a root support.
        for name,angle in getattr(self,'locked_joint_targets',{}).items():
            i=self.b.ind_names.index(name);col=6+i;row=np.zeros((1,nv));row[0,col]=1
            rows.append(row);rhs.append(np.array([400*(angle-q[self.rig.ids[i]])-40*u[col]]))
        if hasattr(self,'prescribed_joint_motion'):
            for name,(angle,rate,accel) in self.prescribed_joint_motion(t).items():
                if name=='mean_hip_flexion':
                    group=[self.b.ind_names.index('hip_flexion_'+s) for s in ['r','l']];row=np.zeros((1,nv));row[0,6+np.array(group)]=.5
                    rows.append(row);rhs.append(np.array([accel+400*(angle-float(np.mean(q[np.array(r.ids)[group]])))+40*(rate-float(np.mean(u[6+np.array(group)])))]));continue
                i=self.b.ind_names.index(name);col=6+i;row=np.zeros((1,nv));row[0,col]=1
                rows.append(row);rhs.append(np.array([accel+400*(angle-q[r.ids[i]])+40*(rate-u[col])]))
        A=np.vstack(rows);target=np.concatenate(rhs);scale=np.maximum(np.linalg.norm(A,axis=1),.02);A/=scale[:,None];target/=scale
        eps=1e-4;des=self.reference(t);before=self.reference(t-eps);after=self.reference(t+eps);qd=(after-before)/(2*eps);qdd=(after-2*des+before)/eps**2
        if not np.array_equal(d.qpos,q):raise RuntimeError('Reference queries changed the mechanical state')
        posture=np.zeros(n);R=Rotation.from_quat(np.r_[q[4:7],q[3]]);Rd=Rotation.from_quat(np.r_[des[4:7],des[3]])
        posture[3:6]=64*(R.inv()*Rd).as_rotvec()-16*u[3:6]
        posture[6:]=getattr(self,'reference_acceleration_scale',1.)*qdd[r.ids]+100*(des[r.ids]-q[r.ids])-20*(u[6:]-qd[r.ids])
        weights=np.r_[np.full(3,.001),np.full(3,.5),np.ones(n-6),np.full(3*na,.02)]
        for i,name in enumerate(self.b.ind_names):
            if name.startswith(('elv_angle','shoulder_','elbow_')):weights[6+i]=4.
            if name in getattr(self,'posture_weight_by_joint',{}):weights[6+i]=self.posture_weight_by_joint[name]
            if mode=='flight' and name in getattr(self,'flight_posture_weight_by_joint',{}):weights[6+i]=self.flight_posture_weight_by_joint[name]
        P=np.diag(weights);linear=np.r_[weights[:n]*posture,np.zeros(3*na)]
        for Cs,vs,strength in soft:P+=strength*Cs.T@Cs;linear+=strength*Cs.T@vs
        if mode=='flight' and hasattr(self,'flight_pelvis_orientation_weight'):
            # Control pelvis attitude through internal joints while the six
            # floating-root equations still conserve external momentum.
            Cp=np.zeros((3,nv));Cp[:,3:6]=np.eye(3)
            P+=self.flight_pelvis_orientation_weight*Cp.T@Cp
            linear+=self.flight_pelvis_orientation_weight*Cp.T@posture[3:6]
        # Track actual thorax orientation, rather than just pelvis quaternion.
        # No external torque is introduced: these are joint-motion objectives.
        if t<.25:theta,omega,alpha=self.theta_stand,0.,0.
        elif t<self.bottom:
            poly=Polynomial(self.theta_stand,0,0,getattr(self,'bottom_trunk_lean_rad',.28),0,0,self.bottom-.25);ss=t-.25;theta,omega,alpha=[float(poly.value(ss,k)[0]) for k in range(3)]
        elif t<self.takeoff:
            poly=Polynomial(getattr(self,'bottom_trunk_lean_rad',.28),0,0,0.,0,0,self.takeoff-self.bottom);ss=t-self.bottom;theta,omega,alpha=[float(poly.value(ss,k)[0]) for k in range(3)]
        else:theta,omega,alpha=0.,0.,0.
        if mode=='flight':theta,omega,alpha=0.,0.,0.
        Rt=Rotation.from_rotvec([theta,0,0]).as_matrix()@self.Rupright
        if hasattr(self,'trunk_target_rotation'):
            Rt=self.trunk_target_rotation(t);omega=alpha=0.
        J=np.zeros((3,m.nv));Jr=np.zeros_like(J);Jd=np.zeros_like(J);Jrd=np.zeros_like(J);mujoco.mj_jac(m,d,J,Jr,d.xpos[self.trunk_bid],self.trunk_bid);mujoco.mj_jacDot(m,d,Jd,Jrd,d.xpos[self.trunk_bid],self.trunk_bid)
        err=Rotation.from_matrix(Rt@d.xmat[self.trunk_bid].reshape(3,3).T).as_rotvec();Ct=np.zeros((3,nv));Ct[:,:n]=Jr@B;vt=np.array([alpha,0,0])+100*err+20*(np.array([omega,0,0])-Jr@d.qvel)-Jrd@d.qvel-Jr@curve
        gain=getattr(self,'trunk_tracking_gain',100.);vt=np.array([alpha,0,0])+gain*err+2*np.sqrt(gain)*(np.array([omega,0,0])-Jr@d.qvel)-Jrd@d.qvel-Jr@curve
        if hasattr(self,'trunk_target_kinematics'):
            Rt,wr,ar=self.trunk_target_kinematics(t);err=Rotation.from_matrix(Rt@d.xmat[self.trunk_bid].reshape(3,3).T).as_rotvec()
            vt=ar+100*err+20*(wr-Jr@d.qvel)-Jrd@d.qvel-Jr@curve
        trunk_weight=getattr(self,'flight_trunk_weight',2000.) if mode=='flight' else getattr(self,'ground_trunk_weight',2000.)
        P+=trunk_weight*Ct.T@Ct;linear+=trunk_weight*Ct.T@vt
        if active:
            J=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,J,1)
            # COM task is a controller objective. GRF is still solved from EOM.
            ss=np.clip(t-.25,0,self.forward_T);cy=self.forward.value(ss)[0];vy=self.forward.value(ss,1)[0];ay=self.forward.value(ss,2)[0]
            if hasattr(self,'horizontal_target'):cy,vy,ay=self.horizontal_target(t)
            value=np.array([60*(self.c0[0]-d.subtree_com[1,0])-16*d.subtree_linvel[1,0],ay+60*(cy-d.subtree_com[1,1])+16*(vy-d.subtree_linvel[1,1])])
            if t<.25:value[1]=60*(self.c0[1]-d.subtree_com[1,1])-16*d.subtree_linvel[1,1]
            zmin=self.c0[2]-getattr(self,'crouch_drop_m',.15)
            if t<.25:cz,vz,az=self.c0[2],0.,0.
            elif t<self.bottom:
                poly=Polynomial(self.c0[2],0,0,zmin,0,0,self.bottom-.25);ss=t-.25;cz,vz,az=[float(poly.value(ss,k)[0]) for k in range(3)]
            else:
                T=self.takeoff-self.bottom;ss=t-self.bottom;u0=np.clip(ss/T,0,1);G=9.80665;C=np.pi/2*(getattr(self,'requested_launch_vz_m_s',1.6)/(G*T)+.5)
                cz=zmin+G*T*T*(-u0**3/6+C*(u0/np.pi-np.sin(np.pi*u0)/np.pi**2));vz=G*T*(-u0*u0/2+C*(1-np.cos(np.pi*u0))/np.pi);az=G*(-u0+C*np.sin(np.pi*u0))
                if ss>T:extra=ss-T;cz+=vz*extra-.5*G*extra*extra;vz-=G*extra;az=-G
            value=np.r_[value,az+100*(cz-d.subtree_com[1,2])+20*(vz-d.subtree_linvel[1,2])]
            if hasattr(self,'com_acceleration_target'):
                value=self.com_acceleration_target(t)
            elif t>=.25 and hasattr(self,'balanced_horizontal_acceleration'):
                value[1]=self.balanced_horizontal_acceleration(t,float(value[2]))
            Ccom=np.zeros((3,nv));Ccom[:,:n]=J@B
            # Obtain COM convective acceleration by mass-weighted body Jdot.
            conv=np.zeros(3)
            for bid in range(1,m.nbody):
                if m.body_mass[bid]<=0:continue
                Jp=np.zeros((3,m.nv));Jr=np.zeros_like(Jp);Jd=np.zeros_like(Jp);Jrd=np.zeros_like(Jp);mujoco.mj_jacBodyCom(m,d,Jp,Jr,bid);mujoco.mj_jacDot(m,d,Jd,Jrd,d.xipos[bid],bid)
                conv+=m.body_mass[bid]*(Jd@d.qvel+Jp@curve)
            value-=conv/m.body_mass.sum();com_weight=getattr(self,'com_tracking_weight',5000.);P+=com_weight*Ccom.T@Ccom;linear+=com_weight*Ccom.T@value
            if hasattr(self,'hard_com_tracking_axes'):
                selected=self.hard_com_tracking_axes
                if hasattr(self,'hard_com_axes_after_start') and t>=self.hard_com_start_s:selected=self.hard_com_axes_after_start
                axes=np.array(selected,dtype=int);sc=np.maximum(np.linalg.norm(Ccom[axes],axis=1),.02)
                A=np.vstack([A,Ccom[axes]/sc[:,None]]);target=np.r_[target,value[axes]/sc]
            if hasattr(self,'release_acceleration_target') and t>=self.release_start_s:
                release=self.release_acceleration_target(t)
                axes=np.array(getattr(self,'release_tracking_axes',[0,1,2]),dtype=int)
                row_scale=np.maximum(np.linalg.norm(Ccom[axes],axis=1),.02)
                A=np.vstack([A,Ccom[axes]/row_scale[:,None]]);target=np.r_[target,(release-conv/m.body_mass.sum())[axes]/row_scale]
        _,tri,piv=qr(A.T,pivoting=True,mode='economic');rank=int(np.sum(abs(np.diag(tri))>1e-8));keep=piv[:rank]
        inequalities=[];limits=[]
        if mode in ['toe','flight']:
            # Joint-controlled clearance, NOT support: root equations still
            # contain zero external ground force in flight.
            for idx,((bid,local,_,_),J,p) in enumerate(zip(self.p['points'],js,points)):
                if idx in active:continue
                Jd=np.zeros_like(J);Jrd=np.zeros_like(J);mujoco.mj_jacDot(m,d,Jd,Jrd,p,bid);row=np.zeros(nv)
                if mode=='flight' and hasattr(self,'clearance_profile'):
                    height,hy,hyy=self.clearance_profile(p[1]);normal=np.array([0.,-hy,1.]);vel=J@d.qvel
                    row[:n]=normal@J@B;gap=p[2]-height;bound=-float(normal@(Jd@d.qvel+J@curve))+hyy*vel[1]**2-80*float(normal@vel)-1600*gap
                else:
                    row[:n]=(J@B)[2];gap=p[2]-self.p['ground'];bound=-float((Jd@d.qvel+J@curve)[2])-80*float((J@d.qvel)[2])-1600*gap
                inequalities.append(row);limits.append(bound)
        for k in range(na):
            for x,y,z in [(0,0,1)]+[(float(np.cos(phi)),float(np.sin(phi)),float(.8*np.cos(np.pi/16))) for phi in np.linspace(0,2*np.pi,16,endpoint=False)]:
                row=np.zeros(nv);row[n+3*k:n+3*k+3]=[x,y,z];inequalities.append(row);limits.append(0.)
        for i,jid in enumerate(self.b.independent):
            if m.jnt_limited[jid]:
                lo,hi=m.jnt_range[jid];x=q[r.ids[i]];v=u[6+i];k=getattr(self,'joint_range_barrier_rate',40.);row=np.zeros(nv);row[6+i]=1
                inequalities.extend([row,-row]);limits.extend([-2*k*v+k*k*(lo+1e-5-x),2*k*v-k*k*(hi-1e-5-x)])
        C=np.vstack([A[keep],inequalities]);bounds=np.r_[target[keep],limits]
        try:
            if getattr(self,'reduce_equalities',False):
                from scipy.linalg import null_space
                E=A[keep];e=target[keep];base=np.linalg.lstsq(E,e,rcond=1e-12)[0];N=null_space(E,rcond=1e-10)
                D=C[rank:];lower=bounds[rank:]-D@base
                if N.shape[1]:
                    G=N.T@P@N;g=N.T@(linear-P@base);G=(G+G.T)/2
                    DN=D@N;row_norm=np.linalg.norm(DN,axis=1);valid=row_norm>1e-10
                    if np.any(lower[~valid]>1e-7):raise ValueError('Infeasible fixed inequality')
                    import osqp,scipy.sparse as sparse
                    objective_scale=max(1.,float(np.diag(G).max()))
                    qp=osqp.OSQP();qp.setup(P=sparse.csc_matrix(np.triu(G/objective_scale)),q=-g/objective_scale,
                        A=sparse.csc_matrix(DN[valid]/row_norm[valid,None]),l=lower[valid]/row_norm[valid],
                        u=np.full(int(valid.sum()),np.inf),verbose=False,eps_abs=1e-8,eps_rel=1e-8,
                        max_iter=30000,polishing=True)
                    solution=qp.solve()
                    if solution.x is None or solution.info.status_val not in [1,2]:raise ValueError('Reduced QP '+solution.info.status)
                    sol=base+N@solution.x
                else:
                    if np.max(lower)>1e-6:raise ValueError('No free direction for inequalities')
                    sol=base
                if np.min(C[rank:]@sol-bounds[rank:])<-1e-6:raise ValueError('Reduced QP inequality residual')
            else:
                objective_scale=max(1.,float(np.diag(P).max())) if getattr(self,'polish_equalities',False) else 1.
                sol=quadprog.solve_qp(P/objective_scale,linear/objective_scale,C.T,bounds,meq=rank)[0]
                if getattr(self,'polish_equalities',False):
                    for _ in range(3):sol+=np.linalg.lstsq(A[keep],target[keep]-A[keep]@sol,rcond=1e-12)[0]
                    if np.min(C[rank:]@sol-bounds[rank:])<-1e-6:raise ValueError('Polished QP inequality residual')
        except ValueError as e:raise RuntimeError(('Contact QP',t,mode,str(e))) from e
        residual=abs(A@sol-target)
        contact_rows=np.arange(contact_start,contact_end) if active else np.array([],dtype=int)
        required=np.ones(len(residual),bool);required[contact_rows]=False
        contact_error=float(residual[contact_rows].max()) if len(contact_rows) else 0.
        self.maximum_contact_stabilization_residual=max(getattr(self,'maximum_contact_stabilization_residual',0.),contact_error)
        if residual[required].max()>1e-5 or contact_error>getattr(self,'contact_stabilization_tolerance',1e-5):
            if hasattr(self,'debug_folder'):np.savez_compressed(self.debug_folder/'constraint_failure.npz',A=A,target=target,sol=sol,keep=keep,q=q,u=u,time=t,C=C,bounds=bounds,rank=rank)
            raise RuntimeError(('Equality residual',t,float(residual.max()),int(np.argmax(residual))))
        forces=np.zeros((len(js),3))
        for k,i in enumerate(active):forces[i]=sol[n+3*k:n+3*k+3]*self.W
        if mode=='toe' and getattr(self,'adaptive_toe_release',False):
            zero=[i for i in active if forces[i,2]<1e-4]
            if zero:
                self.released_toe_points.update(zero)
                self.toe_release_events.append(dict(time_s=float(t),point_indices=zero))
                return self.solve(q,u,t,mode)
        return sol[:n],B,curve,np.array(points),forces

    def run(self):
        dt=1/self.hz;q=self.stand.copy();u=np.zeros(self.n);arr={k:[] for k in ['times','qpos','qvel','qacc','u','udot','com','velocity','angular_momentum','force_world_N','COP_y_m','mode','body_xpos','body_xmat','contact_forces_N','contact_positions_m']}
        self.trace=arr
        events=[.25,self.bottom,self.toe,self.takeoff]
        if hasattr(self,'planned_takeoff_s'):events.append(self.planned_takeoff_s)
        if hasattr(self,'release_start_s'):events.append(self.release_start_s)
        times=np.unique(np.round(np.r_[np.arange(0,2.5+dt/2,dt),events],10));steps=[]
        if hasattr(self,'resume_data'):
            previous=self.resume_data;k=np.flatnonzero(previous['mode']==2)[0]
            q=previous['qpos'][k].copy();u=previous['u'][k].copy()
            for key in arr:arr[key]=list(previous[key][:k])
            steps=list(zip(previous['integration_dt_s'][:k],previous['integration_force_world_N'][:k],previous['integration_power_W'][:k]))
            times=times[times>=self.actual_takeoff-1e-9]
        for i,t in enumerate(times):
            # Lift-off can precede the nominal heel-rise target. Once released,
            # never reattach merely because t is still below that target.
            mode='flight' if self.actual_takeoff is not None else 'plant' if t<self.toe else 'toe'
            commanded=False
            if mode=='toe' and t>=getattr(self,'planned_takeoff_s',float('inf'))-1e-9:
                B,curve=self.rig.set_state(q,u);self.actual_takeoff=float(t)
                qd=np.zeros_like(q);qd[self.rig.ids]=(B@u)[self.rig.dofs]
                self.configure_flight(q,u,qd,np.zeros_like(q));mode='flight';commanded=True
            a,B,curve,points,forces=self.solve(q,u,t,mode)
            if commanded:
                velocity=[];acceleration=[];qacc=B@a+curve
                for bid,local,_,_ in self.p['points']:
                    J,p=self.b.jacobian(bid,local);Jd=np.zeros_like(J);Jrd=np.zeros_like(J);mujoco.mj_jacDot(self.m,self.d,Jd,Jrd,p,bid)
                    if p[2]-self.p['ground']<.001:
                        velocity.append(float((J@self.d.qvel)[2]));acceleration.append(float((J@qacc+Jd@self.d.qvel)[2]))
                self.flight_release_diagnostic=dict(commanded_time_s=float(t),minimum_near_ground_normal_velocity_m_s=min(velocity,default=0),minimum_near_ground_normal_acceleration_m_s2=min(acceleration,default=0),velocity_reset_applied=False)
                if min(velocity,default=0)<-.005 or min(acceleration,default=0)<-.5:raise RuntimeError(('Commanded release does not separate',self.flight_release_diagnostic))
                print('COMMANDED_SEPARATING_LIFTOFF',self.flight_release_diagnostic,flush=True)
            if mode in ['plant','toe'] and forces[:,2].sum()<1. and self.d.subtree_linvel[1,2]>.2:
                self.actual_takeoff=float(t);qd=np.zeros_like(q);qdd=np.zeros_like(q);qd[self.rig.ids]=(B@u)[self.rig.dofs];qdd[self.rig.ids]=(B@a+curve)[self.rig.dofs]
                vz=self.d.subtree_linvel[1,2];self.flight_T=float((vz+np.sqrt(vz*vz+2*9.80665*.08))/9.80665);self.air_transition_T=max(.2,self.flight_T);endpoint=self.air.copy();endpoint[:7]=q[:7];self.air_poly=Polynomial(q,qd,qdd,endpoint,0,0,self.air_transition_T)
                self.clearance_height=vz*vz/(2*9.80665)+.04;self.flight_feet=[]
                for side,foot in self.p['feet'].items():
                    bid=foot['toe_bid'];local=np.mean([p[1] for p in self.p['points'] if p[0]==bid],axis=0);J,p=self.b.jacobian(bid,local);end=p.copy();end[:2]+=self.d.subtree_linvel[1,:2]*self.flight_T;self.flight_feet.append((bid,local,Polynomial(p,J@self.d.qvel,0,end,0,0,self.flight_T),self.d.xmat[bid].reshape(3,3).copy()))
                if hasattr(self,'configure_flight'):self.configure_flight(q,u,qd,qdd)
                mode='flight';a,B,curve,points,forces=self.solve(q,u,t,mode)
                normal_velocity=[];normal_acceleration=[];qacc=B@a+curve
                for bid,local,_,_ in self.p['points']:
                    J,p=self.b.jacobian(bid,local);Jd=np.zeros_like(J);Jrd=np.zeros_like(J)
                    mujoco.mj_jacDot(self.m,self.d,Jd,Jrd,p,bid)
                    if p[2]-self.p['ground']<.001:
                        normal_velocity.append(float((J@self.d.qvel)[2]))
                        normal_acceleration.append(float((J@qacc+Jd@self.d.qvel)[2]))
                self.flight_release_diagnostic=dict(mode='vanishing normal force; any support phase',
                    time_s=float(t),minimum_near_ground_normal_velocity_m_s=min(normal_velocity,default=0),
                    minimum_near_ground_normal_acceleration_m_s2=min(normal_acceleration,default=0),velocity_reset_applied=False)
                if min(normal_velocity,default=0)<-.005 or min(normal_acceleration,default=0)<-.5:
                    raise RuntimeError(('Passive release does not separate',self.flight_release_diagnostic))
                print('LIFTOFF',t,self.d.subtree_linvel[1],flush=True)
            if mode=='flight' and t>=self.actual_takeoff+self.flight_T:break
            # End before the first touchdown. Impact/contact transition is a
            # separate solve; never differentiate a velocity reset as muscle force.
            if mode=='flight' and t>self.actual_takeoff+.04 and points[:,2].min()<self.p['ground']-.00001:
                print('TOUCHDOWN_BOUNDARY',float(t),flush=True);break
            F=forces.sum(0);cop=float((points[:,1]*forces[:,2]).sum()/F[2]) if F[2]>1e-8 else 0.
            self.previous_force_world_N=F.copy()
            vals=dict(times=t,qpos=q.copy(),qvel=B@u,qacc=B@a+curve,u=u.copy(),udot=a,com=self.d.subtree_com[1].copy(),velocity=self.d.subtree_linvel[1].copy(),angular_momentum=self.d.subtree_angmom[1].copy(),force_world_N=F,COP_y_m=cop,mode=['plant','toe','flight','land'].index(mode),body_xpos=self.d.xpos.copy(),body_xmat=self.d.xmat.copy(),contact_forces_N=forces,contact_positions_m=points)
            for k,v in vals.items():arr[k].append(v)
            if i%max(1,self.hz//2)==0:print('CONTACT_JUMP',round(float(t),3),mode,'COM',vals['com'].round(3),'F',F.round(1),flush=True)
            if i+1==len(times):break
            step=times[i+1]-t;qm=q.copy();mujoco.mj_integratePos(self.m,qm,B@u,step/2);qm=self.b.complete(qm);um=u+step*a/2
            am,Bm,cm,_,fm=self.solve(qm,um,t+step/2,mode);self.d.qacc[:]=Bm@am+cm;required=np.zeros(self.m.nv);mujoco.mj_rne(self.m,self.d,1,required);steps.append((step,fm.sum(0),float(required@self.d.qvel)))
            qn=q.copy();mujoco.mj_integratePos(self.m,qn,Bm@um,step);q=self.b.complete(qn);u+=step*am
        # The last integrated interval can terminate at the touchdown boundary,
        # outside the saved stance/flight samples. Restrict quadrature to data.
        steps=steps[:len(arr['times'])-1]
        if self.actual_takeoff is None:
            np.savez_compressed(ROOT/'results/contact_jump_scapula/no_liftoff_debug.npz',**{k:np.asarray(v) for k,v in arr.items()})
            raise RuntimeError('No passive contact release: the reference did not jump')
        return {**{k:np.asarray(v) for k,v in arr.items()},'integration_dt_s':np.array([s[0] for s in steps]),'integration_force_world_N':np.array([s[1] for s in steps]),'integration_power_W':np.array([s[2] for s in steps])}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hz',type=int,default=600);ap.add_argument('--scapula-contact',action='store_true');a=ap.parse_args()
    if a.scapula_contact:
        from jump_model import ArticulatedJumpRig
        rig=ArticulatedJumpRig()
    else:rig=RigidBaselineRig()
    motion=ContactJump(rig,a.hz);dest=ROOT/f'results/{"contact_jump_scapula" if a.scapula_contact else "contact_jump"}/{a.hz}hz';dest.mkdir(exist_ok=True,parents=True)
    data=motion.run();np.savez_compressed(dest/'motion.npz',**data)
    deps=['build_contact_jump.py','rigid_baseline_profile.py','parametric_body.py','jump_rig.py']+(['scapulothoracic_rig.py','jump_model.py'] if a.scapula_contact else [])
    (dest/'setup.json').write_text(json.dumps(dict(profile=rig.profile,rig_type='scapular_contact_articulated_toes' if a.scapula_contact else 'rigid_baseline',physics_hz=a.hz,coordinate_names=rig.b.coordinate_names,body_names=[rig.m.body(i).name for i in range(rig.m.nbody)],takeoff_s=motion.actual_takeoff,countermovement_end_s=motion.bottom,heel_rise_start_s=motion.toe,scope='stance, countermovement, push-off and flight; stops before touchdown',ground_reaction_and_COP_are_inputs=False,posture_goals_are_inputs=True,human_motion_validated=False,friction_model='16-face inner approximation to circular Coulomb cone, mu=0.8',source_sha256={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in deps}),indent=2))
    (dest/'contact_definition.json').write_text(json.dumps(dict(points=[dict(body_id=bid,body=rig.m.body(bid).name,local_m=local.tolist(),initial_world_m=p.tolist(),forefoot=fore) for bid,local,p,fore in motion.p['points']],mass_kg=float(rig.m.body_mass.sum()),gravity_m_s2=9.80665,ground_z_m=motion.p['ground']),indent=2))
    print('CONTACT_JUMP_COMPLETE',dest,flush=True)

if __name__=='__main__':main()
