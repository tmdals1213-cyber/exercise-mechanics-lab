"""Full-body geometry and exact coordinate reduction for jump references."""
import json,hashlib,inspect
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from parametric_body import Subject,build_subject
from wholebody_ik import FrameTarget,solve_wholebody_ik
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/jump_validation'
PHYSICS=json.loads((ROOT/'scenarios/jump_physics.json').read_text(encoding='utf-8'))

class JumpRig:
    def __init__(self,subject=None):
        self.b=build_subject(Subject() if subject is None else subject);self.m=self.b.m;self.d=self.b.d;m=self.m
        m.opt.gravity[:]=PHYSICS['gravity_world_m_s2']
        m.opt.density=PHYSICS['fluid_density_kg_m3'];m.opt.viscosity=PHYSICS['fluid_viscosity_Pa_s']
        m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        m.dof_damping[:6]=0
        self.ids=[int(m.jnt_qposadr[j]) for j in self.b.independent]
        self.dofs=[int(m.jnt_dofadr[j]) for j in self.b.independent]
        self.n=6+len(self.ids);self.M=np.zeros((m.nv,m.nv))
        # Topologically order polynomial couplers once.
        pending=list(range(m.neq));done=set(self.b.independent);self.eqs=[]
        while pending:
            available=[k for k in pending if int(m.eq_obj2id[k])<0 or int(m.eq_obj2id[k]) in done]
            if not available:raise ValueError('Cyclic coordinate equality')
            for k in available:
                dep=int(m.eq_obj1id[k]);ind=int(m.eq_obj2id[k]);c=m.eq_data[k,:5].copy()
                self.eqs.append((int(m.jnt_qposadr[dep]),int(m.jnt_dofadr[dep]),int(m.jnt_qposadr[ind]) if ind>=0 else -1,int(m.jnt_dofadr[ind]) if ind>=0 else -1,c))
                done.add(dep);pending.remove(k)
        self.B0=np.zeros((m.nv,self.n));self.B0[:6,:6]=np.eye(6)
        self.B0[self.dofs,np.arange(6,self.n)]=1.
        self.foot=[];self.radius=.018;self.k=1.0e6;self.alpha=1.5;self.kt=4.0e4;self.ct=180.;self.mu=.85
        self.b.pose(self.b.q0)
        for side in ['r','l']:
            bid=m.body('calcn_'+side).id;sole=self.d.site_xpos[m.site('subject_sole_reference_'+side).id].copy()
            for x,y in [(-.037,-.16),(.037,-.16),(-.037,.095),(.037,.095)]:
                p=sole+np.array([x,y-.085,self.radius])*self.b.report['global_normalization_scale']
                self.foot.append((bid,self.d.xmat[bid].reshape(3,3).T@(p-self.d.xpos[bid])))
        self.anchors=np.zeros((len(self.foot),2));self.contact_active=np.zeros(len(self.foot),bool)
        self.last_pose=None


    def coordinates(self,q,u):
        B=self.B0.copy();curve=np.zeros(self.m.nv)
        for adr,dof,ia,idof,c in self.eqs:
            x=q[ia] if ia>=0 else 0.;p1=c[1]+x*(2*c[2]+x*(3*c[3]+4*x*c[4]));p2=2*c[2]+x*(6*c[3]+12*x*c[4])
            if idof>=0:
                B[dof]=p1*B[idof];v=float(B[idof]@u);curve[dof]=p1*curve[idof]+p2*v*v
        return B,curve


    def set_state(self,q,u):
        B,curve=self.coordinates(q,u);self.d.qpos[:]=q;self.d.qvel[:]=B@u
        mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d);mujoco.mj_crb(self.m,self.d);mujoco.mj_comVel(self.m,self.d)
        mujoco.mj_rne(self.m,self.d,0,self.d.qfrc_bias);mujoco.mj_passive(self.m,self.d);mujoco.mj_subtreeVel(self.m,self.d)
        return B,curve


    def arm(self,q,plane,elevation,elbow=.12):
        for side in ['r','l']:
            for name,value in [('elv_angle',plane),('shoulder_elv',elevation),('shoulder_rot',0.),('elbow_flexion',elbow),('pro_sup',0.)]:
                q[self.m.jnt_qposadr[self.b.joints[name+'_'+side]]]=value
        return q


    def planted_pose(self,hip,knee,com_forward=.025,arms=(1.4,.2),trunk=.05):
        signature=hashlib.sha256((inspect.getsource(type(self).planted_pose)+self.b.report['implementation_sha256']+str((hip,knee,com_forward,arms,trunk))).encode()).hexdigest()
        cache=OUT/'pose_cache'/f'{signature}.json'
        if cache.exists():
            record=json.loads(cache.read_text('utf-8'));q=np.array(record['qpos']);self.b.pose(q);return q,record['info']
        b=self.b;m=self.m;q=b.q0.copy();self.arm(q,*arms)
        for side in ['r','l']:
            q[m.jnt_qposadr[b.joints['hip_adduction_'+side]]]=0.
            q[m.jnt_qposadr[b.joints['hip_flexion_'+side]]]=hip;q[m.jnt_qposadr[b.joints['knee_angle_'+side]]]=knee
        q[m.jnt_qposadr[b.joints['flex_extension']]]=trunk
        b.pose(b.q0);base_sole=np.mean([self.d.site_xpos[m.site('subject_sole_reference_'+s).id] for s in ['r','l']],axis=0)
        target_y=base_sole[1]-com_forward
        def pose(v):
            qq=q.copy();qq[3:7]=[np.cos(v[0]/2),np.sin(v[0]/2),0,0]
            for side in ['r','l']:qq[m.jnt_qposadr[b.joints['ankle_angle_'+side]]]=v[1]
            b.pose(qq);sole=np.mean([self.d.site_xpos[m.site('subject_sole_reference_'+s).id] for s in ['r','l']],axis=0)
            qq[:3]+=base_sole-sole;b.pose(qq)
            return qq
        b.pose(b.q0);R0=self.d.xmat[m.body('calcn_r').id].reshape(3,3).copy()
        def residual(v):
            pose(v);R=self.d.xmat[m.body('calcn_r').id].reshape(3,3);com=self.d.subtree_com[1]
            return np.r_[Rotation.from_matrix(R0.T@R).as_rotvec()*10,(com[1]-target_y)*10]
        opt=least_squares(residual,[.15,.2],bounds=([-.9,-.65],[.9,.55]),ftol=1e-11,xtol=1e-11,gtol=1e-11)
        result=pose(opt.x)
        b.pose(b.q0);targets=[]
        for side in ['r','l']:
            bid=m.body('calcn_'+side).id;sid=m.site('subject_sole_reference_'+side).id
            targets.append(FrameTarget('foot_'+side,bid,m.site_pos[sid].copy(),self.d.site_xpos[sid].copy(),self.d.xmat[bid].reshape(3,3).copy()))
        # Native knees/ankles have oblique axes. Fit the full 3D chain rather
        # than treating them as ideal planar hinges or widening their ROM.
        xcentre=float(self.d.subtree_com[1,0]);bounds=np.array([[xcentre-.015,target_y-.008],[xcentre+.015,target_y+.008]])
        result,fit=solve_wholebody_ik(b,result,targets,support_bounds_world_xy=bounds)
        info={'fit':fit,'COM':self.d.subtree_com[1].tolist()};cache.parent.mkdir(parents=True,exist_ok=True)
        cache.write_text(json.dumps({'qpos':result.tolist(),'info':info}),encoding='utf-8')
        return b.complete(result),info

