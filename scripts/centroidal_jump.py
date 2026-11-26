"""Prescribed-force, full-body momentum-consistent jump reference motions.

Ground force profiles are INPUTS, not predictions of personal performance or
landing material stiffness. Actual full-body COM and angular momentum are
matched, contacts are constrained, and net inverse-dynamics loads are audited.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
import osqp,mujoco
from jump_rig import JumpRig,ROOT,OUT,PHYSICS

G=-PHYSICS['gravity_world_m_s2'][2]


def s5(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)


class Polynomial:
    def __init__(self,p0,v0,a0,p1,v1,a1,T):
        p0=np.atleast_1d(p0);v0=np.broadcast_to(v0,p0.shape);a0=np.broadcast_to(a0,p0.shape);p1=np.broadcast_to(p1,p0.shape);v1=np.broadcast_to(v1,p0.shape);a1=np.broadcast_to(a1,p0.shape)
        self.c=np.zeros((6,len(p0)));self.c[:3]=[p0,v0,a0/2]
        A=np.array([[T**3,T**4,T**5],[3*T**2,4*T**3,5*T**4],[6*T,12*T**2,20*T**3]])
        self.c[3:]=np.linalg.solve(A,np.array([p1-p0-v0*T-a0*T*T/2,v1-v0-a0*T,a1-a0]))
    def value(self,t,derivative=0):
        c=self.c.copy()
        for _ in range(derivative):c=c[1:]*np.arange(1,len(c))[:,None]
        return np.array([t**k for k in range(len(c))])@c


def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def rotvec_rate_world(R0,r,v):
    a=np.linalg.norm(r);K=skew(r)
    J=np.eye(3)+(.5 if a<1e-6 else (1-np.cos(a))/a**2)*K+(1/6 if a<1e-6 else (a-np.sin(a))/a**3)*(K@K)
    return R0@J@v


def integrate(t,a,p0,v0=0):
    v=v0+cumulative_trapezoid(a,t,axis=0,initial=0)
    p=p0+cumulative_trapezoid(v,t,axis=0,initial=0)
    return p,v


def cop_plan(t,com,F,lo,hi,H0=0):
    # Sparse H / COP optimization. Bound COP to actual support, penalize large
    # whole-body angular momentum and prefer a quiet final momentum.
    count=len(t);n=2*count;dt=np.diff(t)
    P=sp.diags(np.r_[np.full(count,.03),np.ones(count)],format='csc').tolil();P[-1,-1]=1000.;P=P.tocsc()
    q=np.r_[-.03*(lo+hi)/2,np.zeros(count)]
    rows=[];cols=[];values=[];rhs=[]
    for k in range(count-1):
        for col,value in [(count+k+1,1),(count+k,-1),(k,-dt[k]*F[k,2]/2),(k+1,-dt[k]*F[k+1,2]/2)]:rows.append(k);cols.append(col);values.append(value)
        offset=-com[:,1]*F[:,2]+com[:,2]*F[:,1]
        rhs.append(dt[k]*(offset[k]+offset[k+1])/2)
    D=sp.csc_matrix((values,(rows,cols)),shape=(count-1,n));initial=sp.csc_matrix(([1.],([0],[count])),shape=(1,n))
    A=sp.vstack([D,initial,sp.eye(n,format='csc')],format='csc');lower=np.r_[rhs,H0,lo,np.full(count,-np.inf)];upper=np.r_[rhs,H0,hi,np.full(count,np.inf)]
    solver=osqp.OSQP();solver.setup(P=P,q=q,A=A,l=lower,u=upper,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=200000,polishing=True)
    result=solver.solve(raise_error=False)
    if not result.info.status.startswith('solved'):raise RuntimeError('COP plan '+result.info.status)
    return result.x[:count],result.x[count:]


def force_plan(rig,kind,hz):
    b=rig.b;m=rig.m;mass=m.body_mass.sum();ground=.025*b.report['global_normalization_scale']
    stand,info=rig.planted_pose(.05,.12,.025)
    b.pose(stand);c0=rig.d.subtree_com[1].copy()
    feet={};points=[]
    for side in ['r','l']:
        bid=m.body('calcn_'+side).id;sid=m.site('subject_sole_reference_'+side).id
        feet[side]={'bid':bid,'local':m.site_pos[sid].copy(),'p':rig.d.site_xpos[sid].copy(),'R':rig.d.xmat[bid].reshape(3,3).copy()}
        origin=feet[side]['p']
        for x,y in [(-.037,-.245),(.037,-.245),(-.037,.010),(.037,.010)]:
            p=origin+np.array([x,y,.025])*b.report['global_normalization_scale'];local=feet[side]['R'].T@(p-rig.d.xpos[bid]);points.append((bid,local,p.copy(),y<0))
    propT=.14 if kind=='vertical' else .20;t_to=.85+propT;toe_start=t_to-.02;t=np.arange(round(t_to*hz)+1)/hz;az=np.zeros(len(t));vz_to=2.5 if kind=='vertical' else 2.4
    mask=(t>=.25)&(t<.6);u=(t[mask]-.25)/.35;az[mask]=-4.5*np.sin(np.pi*u)**2
    mask=(t>=.6)&(t<.85);u=(t[mask]-.6)/.25;az[mask]=6.3*np.sin(np.pi*u)**2
    mask=t>=.85;u=(t[mask]-.85)/propT;P=np.pi/2*(vz_to/propT+G/2);az[mask]=-G*u+P*np.sin(np.pi*u)
    z,vz=integrate(t,az,c0[2]);F=np.zeros((len(t),3));F[:,2]=mass*(az+G);F[-1]=0
    if kind=='vertical':
        y=np.full(len(t),c0[1]);vy=np.zeros(len(t));ay=np.zeros(len(t));F[:,1]=0
    else:
        y_pre=.12;poly=Polynomial(c0[1],0,0,y_pre,0,0,.6)
        y=np.array([c0[1] if a<.25 else poly.value(min(a-.25,.6))[0] for a in t]);vy=np.array([0 if a<.25 else poly.value(min(a-.25,.6),1)[0] for a in t]);ay=np.array([0 if a<.25 else poly.value(min(a-.25,.6),2)[0] for a in t]);
        start=round(.85*hz);rho=-1.2/(vz_to+G*propT);ay[start:]=rho*F[start:,2]/mass;y2,v2=integrate(t[start:],ay[start:],y[start],vy[start]);y[start:]=y2;vy[start:]=v2;F[:,1]=mass*ay
    com=np.c_[np.full(len(t),c0[0]),y,z];vel=np.c_[np.zeros(len(t)),vy,vz]
    sample=np.unique(np.r_[np.arange(0,len(t),max(1,hz//60)),len(t)-1]);toe_y=float(np.mean([p[2][1] for p in points if p[3]]));min_y=min(p[2][1] for p in points);max_y=max(p[2][1] for p in points)
    lo=np.full(len(sample),min_y+.001);hi=np.full(len(sample),max_y-.001);lo[t[sample]>=toe_start]=toe_y;hi[t[sample]>=toe_start]=toe_y
    cop_sample,_=cop_plan(t[sample],com[sample]-[0,0,ground],F[sample],lo,hi)
    cop=np.interp(t,t[sample],cop_sample);cop[t>=toe_start]=toe_y
    moment=(cop-com[:,1])*F[:,2]+(com[:,2]-ground)*F[:,1];H=cumulative_trapezoid(moment,t,initial=0)
    # Ballistic flight endpoint is a chosen landing geometry height, not a
    # measured athlete result. All coordinates are actual body COM coordinates.
    z_land=.90
    Tflight=(vel[-1,2]+np.sqrt(vel[-1,2]**2+2*G*(com[-1,2]-z_land)))/G
    Tv=.11;Th=.34 if kind=='long' else Tv
    a_t=np.arange(round(max(Tv,Th)*hz)+1)/hz
    normal=np.full(len(a_t),mass*G);mask=a_t<=Tv;u=a_t[mask]/Tv
    v0z=vel[-1,2]-G*Tflight;A=1-2*v0z/(G*Tv);normal[mask]=mass*G*(u+A*np.sin(np.pi*u)**2)
    ay_land=np.zeros(len(a_t));weight=np.sin(np.pi*np.minimum(a_t/Th,1))**2
    if kind=='long':
        beta=-vel[-1,1]/np.trapezoid(weight*normal/mass,a_t);ay_land=beta*weight*normal/mass
    y_land0=com[-1,1]+vel[-1,1]*Tflight
    yz,yv=integrate(a_t,np.c_[ay_land,normal/mass-G],[y_land0,z_land],[vel[-1,1],v0z])
    landing_com=np.c_[np.full(len(a_t),c0[0]),yz];landing_vel=np.c_[np.zeros(len(a_t)),yv];landing_F=np.c_[np.zeros(len(a_t)),mass*ay_land,normal]
    foot_shift=landing_com[-1,1]-c0[1]+.04 # Landing COM remains inside the sole polygon.
    land_feet={s:{**p,'p':p['p']+np.array([0,foot_shift,0])} for s,p in feet.items()}
    landing_ylo=min_y+foot_shift+.001;landing_yhi=max_y+foot_shift-.001
    sample2=np.unique(np.r_[np.arange(0,len(a_t),max(1,hz//60)),len(a_t)-1]);cop2,_=cop_plan(a_t[sample2],landing_com[sample2]-[0,0,ground],landing_F[sample2],np.full(len(sample2),landing_ylo),np.full(len(sample2),landing_yhi),H[-1])
    cop_l=np.interp(a_t,a_t[sample2],cop2);moment2=(cop_l-landing_com[:,1])*landing_F[:,2]+(landing_com[:,2]-ground)*landing_F[:,1];H_l=H[-1]+cumulative_trapezoid(moment2,a_t,initial=0)
    ratio=np.max(abs(np.r_[F[:,1],landing_F[:,1]])/np.maximum(np.r_[F[:,2],landing_F[:,2]],1e-8))
    print('FORCE_PLAN',kind,'TO',com[-1],vel[-1],'H',H[-1],'landing',landing_com[0],landing_com[-1],'Hend',H_l[-1],'friction',ratio,flush=True)
    return {'stand':stand,'feet':feet,'points':points,'ground':ground,'t_ground':t,'com_ground':com,'vel_ground':vel,'F_ground':F,'cop_ground':cop,'H_ground':H,'takeoff':t_to,'propT':propT,'toe_start':toe_start,'flight_T':Tflight,'t_land':a_t,'com_land':landing_com,'vel_land':landing_vel,'F_land':landing_F,'cop_land':cop_l,'H_land':H_l,'land_feet':land_feet,'max_friction_ratio':float(ratio)}


class ReferenceMotion:
    def __init__(self,rig,plan,kind,hz):
        self.rig=rig;self.b=rig.b;self.m=rig.m;self.d=rig.d;self.p=plan;self.kind=kind;self.hz=hz
        self.t_land=plan['takeoff']+plan['flight_T'];self.t_absorb=self.t_land+plan['t_land'][-1];self.t_recover=self.t_absorb+.85;self.duration=self.t_recover+.25
        self.flight_paths=None;self.maximum_residual=0.
        self.crouch=rig.planted_pose(1.2,1.65,.035,arms=(1.4,.15),trunk=.05)[0];self.reach=plan['stand'].copy();self.land=plan['stand'].copy();self.extend=plan['stand'].copy()
        for side in ['r','l']:
            for q,h,k,a in [(self.crouch,1.2,1.65,.45),(self.land,1.5,1.95,.45),(self.extend,.03,.05,-.35),(self.reach,.5,.9,.15)]:
                for name,v in [('hip_flexion',h),('knee_angle',k),('ankle_angle',a)]:q[self.m.jnt_qposadr[self.b.joints[name+'_'+side]]]=v
        rig.arm(self.crouch,1.4,.15);rig.arm(self.extend,1.4,2.4);rig.arm(self.reach,1.4,2.9 if kind=='vertical' else 2.0);rig.arm(self.land,1.4,1.0)

    def state_target(self,t):
        p=self.p
        def at(times,values,s):return np.array([np.interp(s,times,values[:,i]) for i in range(values.shape[1])]) if values.ndim==2 else float(np.interp(s,times,values))
        if t<=p['takeoff']+1e-10:
            return at(p['t_ground'],p['com_ground'],t),at(p['t_ground'],p['vel_ground'],t),at(p['t_ground'],p['F_ground'],t),np.array([at(p['t_ground'],p['H_ground'],t),0,0]),at(p['t_ground'],p['cop_ground'],t)
        if t<self.t_land:
            s=t-p['takeoff'];c=p['com_ground'][-1]+s*p['vel_ground'][-1]+[0,0,-.5*G*s*s];v=p['vel_ground'][-1]+[0,0,-G*s]
            return c,v,np.zeros(3),np.array([p['H_ground'][-1],0,0]),0.
        if t<=self.t_absorb:
            s=t-self.t_land;return at(p['t_land'],p['com_land'],s),at(p['t_land'],p['vel_land'],s),at(p['t_land'],p['F_land'],s),np.array([at(p['t_land'],p['H_land'],s),0,0]),at(p['t_land'],p['cop_land'],s)
        s=min(t-self.t_absorb,.85);T=.85;c=p['com_land'][-1].copy();poly=Polynomial(c[2],0,0,p['com_ground'][0,2],0,0,T);c[2]=poly.value(s)[0];v=np.array([0,0,poly.value(s,1)[0]]);F=np.array([0,0,self.m.body_mass.sum()*(G+poly.value(s,2)[0])])
        u=s/T;H=p['H_land'][-1]*(1-s5(u));dH=-p['H_land'][-1]*30*u*u*(1-u)**2/T
        cop=c[1]+dH/max(F[2],1e-8)
        return c,v,F,np.array([H,0,0]),cop

    def mode(self,t):
        if t<self.p['toe_start']-1e-10:return 'plant'
        if t<self.p['takeoff']-1e-10:return 'toe'
        if t<self.t_land-1e-10:return 'flight'
        return 'land'

    def reference(self,t):
        p=self.p
        if t<.25:return p['stand']
        if t<.85:return p['stand']+s5((t-.25)/.6)*(self.crouch-p['stand'])
        if t<p['takeoff']:return self.crouch+s5((t-.85)/p['propT'])*(self.extend-self.crouch)
        if t<self.t_land:
            s=t-p['takeoff'];ratio=s/p['flight_T']
            if ratio<.5:return self.extend+s5(ratio/.5)*(self.reach-self.extend)
            return self.reach+s5((ratio-.5)/.5)*(self.land-self.reach)
        if t<self.t_absorb:return self.land
        return self.land+s5((t-self.t_absorb)/.85)*(p['stand']-self.land)

    def capture_flight(self,q,u):
        B,_=self.rig.set_state(q,u);paths={};T=self.p['flight_T'];self.flight_paths=paths
        for side,foot in self.p['feet'].items():
            bid=foot['bid'];local=foot['local'];J,p=self.b.jacobian(bid,local);Jp=np.zeros((3,self.m.nv));Jr=np.zeros_like(Jp);mujoco.mj_jac(self.m,self.d,Jp,Jr,p,bid)
            R0=self.d.xmat[bid].reshape(3,3).copy();rfinal=Rotation.from_matrix(R0.T@foot['R']).as_rotvec();omega=Jr@(B@u)
            target=self.p['land_feet'][side]['p'];position=Polynomial(p,J@(B@u),np.zeros(3),target,np.zeros(3),np.zeros(3),T)
            orientation=Polynomial(np.zeros(3),R0.T@omega,np.zeros(3),rfinal,np.zeros(3),np.zeros(3),T)
            paths[side]={'position':position,'orientation':orientation,'R0':R0,'bump':.60 if self.kind=='vertical' else .55}

    def foot_target(self,side,t,mode):
        if mode=='flight':
            s=t-self.p['takeoff'];u=np.clip(s/self.p['flight_T'],0,1);path=self.flight_paths[side];poly=path['position'];p=poly.value(s);v=poly.value(s,1)
            bump=16*u*u*(1-u)**2;dbump=32*u*(1-u)*(1-2*u)/self.p['flight_T']
            p[2]+=path['bump']*bump;v[2]+=path['bump']*dbump
            r=path['orientation'].value(s);rdot=path['orientation'].value(s,1);R=path['R0']@Rotation.from_rotvec(r).as_matrix();omega=rotvec_rate_world(path['R0'],r,rdot)
            angle=0.;rate=0.
            omega+=R@np.array([rate,0,0]);R=R@Rotation.from_rotvec([angle,0,0]).as_matrix()
            return p,v,R,omega
        foot=(self.p['land_feet'] if mode=='land' else self.p['feet'])[side]
        return foot['p'],np.zeros(3),foot['R'],np.zeros(3)

    def velocity(self,q,w,t,dt,mode_override=None):
        m,d=self.m,self.d;B,_=self.rig.set_state(q,np.zeros(self.rig.n));mode=mode_override or self.mode(t);com,v,F,H,cop=self.state_target(t)
        J=np.zeros((3,m.nv));A_H=np.zeros_like(J);mujoco.mj_jacSubtreeCom(m,d,J,1);mujoco.mj_angmomMat(m,d,A_H,1)
        rows=[J@B,A_H@B];rhs=[v+50*(com-d.subtree_com[1]),H];soft=[]
        for side,foot in self.p['feet'].items():
            bid=foot['bid']
            if mode=='toe':
                group=[p for p in self.p['points'] if p[0]==bid and p[3]]
                for _,local,target,_ in group:
                    Jp,p=self.b.jacobian(bid,local);rows.append(Jp@B);rhs.append(50*(target-p))
            else:
                target,targetv,Rtarget,omega=self.foot_target(side,t,mode);Jp,p=self.b.jacobian(bid,foot['local']);Jr=np.zeros_like(Jp);mujoco.mj_jac(m,d,Jp,Jr,p,bid)
                R=d.xmat[bid].reshape(3,3);err=Rotation.from_matrix(Rtarget@R.T).as_rotvec()
                rows.append(Jp@B);rhs.append(targetv+50*(target-p))
                if mode=='flight':
                    # Airborne feet need not remain flat. Gradually align them
                    # before touchdown rather than forcing the ankle to absorb
                    # an arbitrary world-fixed orientation throughout flight.
                    remaining=max(self.t_land-t,.001)
                    strength=min(1e6,.02+(.05/remaining)**6)
                    soft.append((Jr@B,omega+max(35.,4/remaining)*err,strength))
                else:rows.append(Jr@B);rhs.append(omega+35*err)
        A=np.vstack(rows);target=np.concatenate(rhs);scale=np.maximum(np.linalg.norm(A,axis=1),.02);A=A/scale[:,None];target=target/scale
        weights=np.r_[np.full(3,.05),np.full(3,2.),np.ones(self.rig.n-6)];denseP=np.diag(weights);objective=-weights*w
        for C,value,strength in soft:denseP+=strength*C.T@C;objective-=strength*C.T@value
        P=sp.csc_matrix(denseP)
        lower=np.full(self.rig.n,-np.inf);upper=np.full(self.rig.n,np.inf)
        for i,jid in enumerate(self.b.independent):
            if m.jnt_limited[jid]:
                lo,hi=m.jnt_range[jid];x=q[self.rig.ids[i]]
                # Continuous velocity damper acts before a joint reaches its
                # anatomical limit; next-step clipping alone creates abrupt
                # velocity redistribution and unreliable differentiated loads.
                lower[6+i]=max((lo+1e-6-x)/max(dt,1e-6),-40*(x-lo-1e-6))
                upper[6+i]=min((hi-1e-6-x)/max(dt,1e-6),40*(hi-1e-6-x))
        constraints=sp.vstack([sp.csc_matrix(A),sp.eye(self.rig.n,format='csc')],format='csc')
        qp=osqp.OSQP();qp.setup(P=P,q=objective,A=constraints,l=np.r_[target,lower],u=np.r_[target,upper],verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=100000,polishing=True)
        result=qp.solve(raise_error=False)
        feasible_candidate=result.x is not None and np.isfinite(result.x).all() and max(float(abs(A@result.x-target).max()),float(np.maximum(lower-result.x,result.x-upper).max()))<1e-4
        if not result.info.status.startswith('solved') and not feasible_candidate:
            (OUT/'debug').mkdir(exist_ok=True,parents=True);np.savez(OUT/'debug/failure.npz',q=q,w=w,A=A,target=target,lower=lower,upper=upper)
            raise RuntimeError(('Kinematics infeasible',t,mode,result.info.status,result.info.prim_res,result.info.dual_res))
        u=result.x;error=float(abs(A@u-target).max());self.maximum_residual=max(self.maximum_residual,error)
        desired=self.reference(t);u_des=np.zeros(self.rig.n);u_des[:3]=u[:3]
        R=Rotation.from_quat(np.r_[q[4:7],q[3]]);Rd=Rotation.from_quat(np.r_[desired[4:7],desired[3]])
        u_des[3:6]=8*(R.inv()*Rd).as_rotvec();u_des[6:]=20*(desired[self.rig.ids]-q[self.rig.ids]);wdot=35*(u_des-w)
        return u,wdot,B

    def run(self):
        dt=1/self.hz;events=[.25,.6,.85,self.p['toe_start'],self.p['takeoff'],self.t_land,self.t_absorb,self.t_recover,self.duration]
        times=np.unique(np.round(np.r_[np.arange(0,self.duration,dt),events],10));times.sort();q=self.p['stand'].copy();w=np.zeros(self.rig.n);previous_mode='plant';previous_u=np.zeros_like(w)
        arrays={k:[] for k in ['times','qpos','qvel','u','com','velocity','angular_momentum','force_world_N','COP_y_m','mode','body_xpos','body_xmat']}
        for i,t in enumerate(times):
            mode=self.mode(t);step=times[i+1]-t if i+1<len(times) else dt
            if mode!=previous_mode:
                old=previous_mode
                if mode=='flight':
                    u_old,_,_=self.velocity(q,w,t,step,mode_override='toe');self.capture_flight(q,u_old);w=u_old.copy()
                else:w=previous_u.copy()
            u,wdot,B=self.velocity(q,w,t,step);self.rig.set_state(q,u)
            com,v,F,H,cop=self.state_target(t)
            values={'times':t,'qpos':q.copy(),'qvel':B@u,'u':u.copy(),'com':self.d.subtree_com[1].copy(),'velocity':self.d.subtree_linvel[1].copy(),'angular_momentum':self.d.subtree_angmom[1].copy(),'force_world_N':F,'COP_y_m':cop,'mode':['plant','toe','flight','land'].index(mode),'body_xpos':self.d.xpos.copy(),'body_xmat':self.d.xmat.copy()}
            for k,value in values.items():arrays[k].append(value)
            if i%max(1,self.hz//2)==0:print('CENTROIDAL',self.kind,round(t,3),mode,'com',values['com'].round(3),'H',values['angular_momentum'].round(2),flush=True)
            if i+1==len(times):break
            qmid=q.copy();mujoco.mj_integratePos(self.m,qmid,B@u,step/2);qmid=self.b.complete(qmid);wmid=w+step*wdot/2
            umid,wmid_dot,Bmid=self.velocity(qmid,wmid,t+step/2,step,mode_override=mode)
            newq=q.copy();mujoco.mj_integratePos(self.m,newq,Bmid@umid,step);q=self.b.complete(newq);w+=step*wmid_dot;previous_mode=mode;previous_u=u
        return {k:np.asarray(v) for k,v in arrays.items()}


def build(kind,hz):
    rig=JumpRig();plan=force_plan(rig,kind,hz);motion=ReferenceMotion(rig,plan,kind,hz);data=motion.run();dest=OUT/kind/f'reference_{hz}hz';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'motion.npz',**data)
    definition={'points':[{'body_id':bid,'body':rig.m.body(bid).name,'local_m':local.tolist(),'initial_world_m':p.tolist(),'forefoot':fore} for bid,local,p,fore in plan['points']],
                'landing_foot_shift_m':(plan['land_feet']['r']['p']-plan['feet']['r']['p']).tolist(),'mass_kg':float(rig.m.body_mass.sum()),'gravity_m_s2':G,'ground_z_m':plan['ground']}
    (dest/'contact_definition.json').write_text(json.dumps(definition,indent=2),encoding='utf-8')
    b=rig.b;b.pose(b.q0)
    np.savez_compressed(dest/'reference_pose.npz',qpos=b.q0,body_xpos=b.d.xpos.copy(),body_xmat=b.d.xmat.copy())
    (dest/'setup.json').write_text(json.dumps({'kind':kind,'physics_hz':hz,'takeoff_s':plan['takeoff'],'landing_s':motion.t_land,'absorption_end_s':motion.t_absorb,'recovery_end_s':motion.t_recover,'duration_s':motion.duration,'friction_required_max':plan['max_friction_ratio'],'ground_z_m':plan['ground'],'body_names':[rig.m.body(i).name for i in range(rig.m.nbody)],'coordinate_names':rig.b.coordinate_names,'max_kinematic_velocity_constraint_residual':motion.maximum_residual,'force_profiles_are_prescribed_inputs':True,'individual_muscle_forces_computed':False},indent=2),encoding='utf-8')
    print('CENTROIDAL_REFERENCE_COMPLETE',kind,dest,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['vertical','long'],default='vertical');ap.add_argument('--hz',type=int,default=600);ap.add_argument('--plan-only',action='store_true');args=ap.parse_args()
    if args.plan_only:force_plan(JumpRig(),args.kind,args.hz)
    else:build(args.kind,args.hz)




