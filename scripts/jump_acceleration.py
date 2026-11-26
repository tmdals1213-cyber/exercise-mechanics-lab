"""Development acceleration-level reference solver with an unactuated root.

Contact force profiles remain prescribed inputs. Root Newton-Euler balance is
an exact QP equality; no residual/root actuator is applied to the body.
"""
import argparse,json,hashlib,importlib.metadata
import numpy as np
import mujoco
import quadprog
from scipy.linalg import qr
from scipy.spatial.transform import Rotation
from centroidal_jump import ReferenceMotion,force_plan
from jump_rig import JumpRig,OUT,ROOT,PHYSICS
from verify_jump_reference import normal_weights


class AccelerationMotion(ReferenceMotion):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        path=ROOT/'assets/motion_guides/long_jump_geometry_only.npz'
        self.guide=dict(np.load(path)) if self.kind=='long' and path.exists() else None

    def acceleration(self,q,u,t,mode):
        m,d=self.m,self.d;B,curve=self.rig.set_state(q,u)
        M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M)
        _,_,force,_,cop=self.state_target(t);external=np.zeros(m.nv)
        points=[];jac=[]
        for bid,local,_,fore in self.p['points']:
            J,p=self.b.jacobian(bid,local);points.append(p);jac.append(J)
        points=np.array(points);active=[i for i,p in enumerate(self.p['points']) if mode!='toe' or p[3]]
        if force[2]>1e-8:
            weights=normal_weights(points[active],d.subtree_com[1,0],cop)
            for i,w in zip(active,weights):external+=jac[i].T@(w*force)
        rows=[(M@B)[:6]];rhs=[(external+d.qfrc_passive-d.qfrc_bias-M@curve)[:6]]
        soft=[]
        for side,foot in self.p['feet'].items():
            bid=foot['bid']
            if mode=='toe':
                targets=[(p[1],p[2]) for p in self.p['points'] if p[0]==bid and p[3]]
            else:targets=[(foot['local'],self.foot_target(side,t,mode)[0])]
            for local,target in targets:
                J,p=self.b.jacobian(bid,local);Jdot=np.zeros_like(J);Jrdot=np.zeros_like(J)
                mujoco.mj_jacDot(m,d,Jdot,Jrdot,p,bid)
                targetv=np.zeros(3);targeta=np.zeros(3)
                if mode=='flight':
                    eps=1e-5;targetv=self.foot_target(side,t,mode)[1]
                    targeta=(self.foot_target(side,t+eps,mode)[1]-self.foot_target(side,t-eps,mode)[1])/(2*eps)
                rows.append(J@B);rhs.append(targeta+40*(targetv-J@d.qvel)+400*(target-p)-Jdot@d.qvel-J@curve)
            if mode!='toe':
                target,targetv,Rtarget,omega=self.foot_target(side,t,mode)
                J=np.zeros((3,m.nv));Jr=np.zeros_like(J);Jdot=np.zeros_like(J);Jrdot=np.zeros_like(J)
                p=d.xpos[bid];mujoco.mj_jac(m,d,J,Jr,p,bid);mujoco.mj_jacDot(m,d,Jdot,Jrdot,p,bid)
                R=d.xmat[bid].reshape(3,3);err=Rotation.from_matrix(Rtarget@R.T).as_rotvec()
                angular_a=np.zeros(3)
                if mode=='flight':
                    eps=1e-5;angular_a=(self.foot_target(side,t+eps,mode)[3]-self.foot_target(side,t-eps,mode)[3])/(2*eps)
                target=angular_a+40*(omega-Jr@d.qvel)+400*err-Jrdot@d.qvel-Jr@curve
                rows.append(Jr@B);rhs.append(target)
        A=np.vstack(rows);target=np.concatenate(rhs);scale=np.maximum(np.linalg.norm(A,axis=1),.02);A/=scale[:,None];target/=scale
        desired=self.reference(t);posture=np.zeros(self.rig.n);guide_u=np.zeros(self.rig.n)
        if self.guide is not None:
            desired=np.array([np.interp(t,self.guide['times'],self.guide['qpos'][:,i]) for i in range(m.nq)])
            desired[3:7]/=np.linalg.norm(desired[3:7])
            guide_u=np.array([np.interp(t,self.guide['times'],self.guide['u'][:,i]) for i in range(self.rig.n)])
            trunk_joint=self.b.joints['flex_extension'];trunk_q=m.jnt_qposadr[trunk_joint]
            if desired[trunk_q]>.25:
                desired[trunk_q]=.25;guide_u[6+self.b.independent.index(trunk_joint)]=0.
        R=Rotation.from_quat(np.r_[q[4:7],q[3]]);Rd=Rotation.from_quat(np.r_[desired[4:7],desired[3]])
        posture[3:6]=64*(R.inv()*Rd).as_rotvec()-16*(u[3:6]-guide_u[3:6])
        posture[6:]=225*(desired[self.rig.ids]-q[self.rig.ids])-30*(u[6:]-guide_u[6:])
        weights=np.r_[np.full(3,.001),np.full(3,.2),np.ones(self.rig.n-6)]
        P=np.diag(weights);objective=-weights*posture
        for C,value,strength in soft:P+=strength*C.T@C;objective-=strength*C.T@value
        lower=np.full(self.rig.n,-np.inf);upper=np.full(self.rig.n,np.inf)
        for i,jid in enumerate(self.b.independent):
            if m.jnt_limited[jid]:
                lo,hi=m.jnt_range[jid];x=q[self.rig.ids[i]];speed=u[6+i];k=150.
                lower[6+i]=-2*k*speed+k*k*(lo+1e-5-x)
                upper[6+i]=-2*k*speed+k*k*(hi-1e-5-x)
        _,Rqr,piv=qr(A.T,pivoting=True,mode='economic');rank=int(np.sum(abs(np.diag(Rqr))>1e-8));keep=piv[:rank]
        I=np.eye(self.rig.n);il=np.where(np.isfinite(lower))[0];iu=np.where(np.isfinite(upper))[0]
        C=np.vstack([A[keep],I[il],-I[iu]]);limit=np.r_[target[keep],lower[il],-upper[iu]]
        try:solution=quadprog.solve_qp(P,-objective,C.T,limit,meq=rank)[0]
        except ValueError as error:
            (OUT/'debug').mkdir(exist_ok=True,parents=True);np.savez(OUT/'debug/acceleration_failure.npz',q=q,u=u,t=t,A=A,target=target,lower=lower,upper=upper)
            raise RuntimeError(('Acceleration QP',t,mode,str(error))) from error
        violation=max(float(abs(A@solution-target).max()),float(np.maximum(lower-solution,solution-upper).max()))
        if violation>1e-4:raise RuntimeError(('Acceleration QP residual',t,mode,violation))
        return solution,B,curve

    def run(self):
        dt=1/self.hz;events=[.25,.6,.85,self.p['toe_start'],self.p['takeoff'],self.t_land,self.t_absorb,self.t_recover,self.duration]
        times=np.unique(np.round(np.r_[np.arange(0,self.duration,dt),events],10));q=self.p['stand'].copy();u=np.zeros(self.rig.n);previous='plant'
        arrays={k:[] for k in ['times','qpos','qvel','qacc','u','udot','com','velocity','angular_momentum','force_world_N','COP_y_m','mode','body_xpos','body_xmat']};self.trace=arrays
        for i,t in enumerate(times):
            mode=self.mode(t);step=times[i+1]-t if i+1<len(times) else dt
            if mode=='flight' and previous!='flight':self.capture_flight(q,u)
            a,B,curve=self.acceleration(q,u,t,mode);self.rig.set_state(q,u)
            _,_,F,_,cop=self.state_target(t)
            values={'times':t,'qpos':q.copy(),'qvel':B@u,'qacc':B@a+curve,'u':u.copy(),'udot':a,'com':self.d.subtree_com[1].copy(),'velocity':self.d.subtree_linvel[1].copy(),'angular_momentum':self.d.subtree_angmom[1].copy(),'force_world_N':F,'COP_y_m':cop,'mode':['plant','toe','flight','land'].index(mode),'body_xpos':self.d.xpos.copy(),'body_xmat':self.d.xmat.copy()}
            for k,v in values.items():arrays[k].append(v)
            if i%max(1,self.hz//2)==0:print('ACCEL',self.kind,round(t,3),mode,values['com'].round(3),flush=True)
            if i+1==len(times):break
            qmid=q.copy();mujoco.mj_integratePos(self.m,qmid,B@u,step/2);qmid=self.b.complete(qmid);umid=u+step*a/2
            amid,Bmid,_=self.acceleration(qmid,umid,t+step/2,mode)
            newq=q.copy();mujoco.mj_integratePos(self.m,newq,Bmid@umid,step);q=self.b.complete(newq);u+=step*amid;previous=mode
        return {k:np.asarray(v) for k,v in arrays.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['vertical','long'],default='vertical');ap.add_argument('--hz',type=int,default=600);a=ap.parse_args()
    rig=JumpRig();plan=force_plan(rig,a.kind,a.hz);motion=AccelerationMotion(rig,plan,a.kind,a.hz);dest=OUT/a.kind/f'acceleration_{a.hz}hz';dest.mkdir(exist_ok=True,parents=True);error=None
    try:data=motion.run()
    except RuntimeError as failure:
        error=str(failure);data={k:np.asarray(v) for k,v in motion.trace.items()}
    np.savez_compressed(dest/('partial_motion.npz' if error else 'motion.npz'),**data)
    (dest/'setup.json').write_text(json.dumps({'kind':a.kind,'physics_hz':a.hz,'body_names':[rig.m.body(i).name for i in range(rig.m.nbody)],'coordinate_names':rig.b.coordinate_names,'takeoff_s':plan['takeoff'],'landing_s':motion.t_land,'duration_s':motion.duration,'absorption_end_s':motion.t_absorb,'recovery_end_s':motion.t_recover,'solver':'acceleration_QP_unactuated_root','force_profiles_are_prescribed_inputs':True,'individual_muscle_forces_computed':False,'physics':PHYSICS,'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','mujoco','myo-sim','quadprog']},'source_sha256':{name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ['jump_acceleration.py','centroidal_jump.py','jump_rig.py','parametric_body.py']}},indent=2))
    definition={'points':[{'body_id':bid,'body':rig.m.body(bid).name,'local_m':local.tolist(),'initial_world_m':p.tolist(),'forefoot':fore} for bid,local,p,fore in plan['points']],'mass_kg':float(rig.m.body_mass.sum()),'gravity_m_s2':9.80665,'ground_z_m':plan['ground']}
    (dest/'contact_definition.json').write_text(json.dumps(definition,indent=2));rig.b.pose(rig.b.q0);np.savez_compressed(dest/'reference_pose.npz',body_xpos=rig.d.xpos,body_xmat=rig.d.xmat)
    if error:
        (dest/'failure.json').write_text(json.dumps({'status':'failed','reason':error,'last_valid_time_s':float(data['times'][-1]),'completed':False},indent=2));raise RuntimeError(error)
    print('ACCELERATION_REFERENCE_COMPLETE',dest,flush=True)


if __name__=='__main__':main()


