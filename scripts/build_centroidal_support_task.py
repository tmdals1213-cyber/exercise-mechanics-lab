"""Task-specified COM motion, feasible COP, and full-body contact coordination.

COM displacement/cadence are scenario inputs, not athlete-performance outputs.
GRF follows m(a-g); COP is confined to the active support patch. Whole-body
momentum and contact kinematics constrain the joint coordination simultaneously.
No floating-root actuator, force cap or muscle strength ceiling is introduced.
"""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco,quadprog
from scipy.linalg import qr
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from retarget_measured_jump import MeasuredJumpRig
from centroidal_jump import Polynomial,cop_plan,s5
from build_contact_jump import contact_geometry

ROOT=Path(__file__).resolve().parents[1];G=9.80665


def bounded_cop(t,C,F,lo,hi,mass):
    n=len(t);K=np.zeros((n,n));offset=np.zeros(n)
    for k,dt in enumerate(np.diff(t)):
        K[k+1]=K[k];K[k+1,k]+=dt*F[k,2]/2;K[k+1,k+1]+=dt*F[k+1,2]/2
        offset[k+1]=offset[k]+dt*((-C[k,1]*F[k,2]+C[k,2]*F[k,1])+(-C[k+1,1]*F[k+1,2]+C[k+1,2]*F[k+1,1]))/2
    K/=mass*G;offset/=mass*G;weight=np.ones(n);weight[-1]=1000.;mid=(lo+hi)/2
    def objective(c):
        h=K@c+offset;return .5*np.sum(weight*h*h)+.015*np.sum((c-mid)**2),K.T@(weight*h)+.03*(c-mid)
    result=minimize(objective,mid,jac=True,bounds=list(zip(lo,hi)),method='L-BFGS-B',options=dict(ftol=1e-13,gtol=1e-10,maxiter=2000))
    if not result.success:raise RuntimeError(('COP optimization',result.message))
    return result.x


class Task:
    def __init__(self,r,hz,vz=2.5,vy=-.45,drop_m=.22,plan_momentum=True,push_duration=.28,takeoff_rise=.06):
        self.r,self.b,self.m,self.d=r,r.b,r.m,r.d
        b,m,d=self.b,self.m,self.d;self.ids=np.array(r.ids);self.names=b.ind_names
        self.start,_=r.planted_pose(.1,.15,.07,arms=(np.pi/2,.12,.15),trunk=0.)
        b.pose(self.start);self.start[2]-=np.mean([d.site_xpos[m.site('subject_sole_reference_'+s).id,2] for s in ['r','l']]);b.pose(self.start)
        self.C0=d.subtree_com[1].copy();self.R0=Rotation.from_quat(np.roll(self.start[3:7],-1));r.contact_stand_override=self.start;self.geom=contact_geometry(r);self.points=self.geom['points']
        b.pose(self.start)
        self.foot={}
        for side,group in [('r',[4,5]),('l',[10,11])]:
            self.foot[side]={}
            for stem in ['calcn','toes']:
                bid=m.body(stem+'_'+side).id;self.foot[side][stem]=(bid,d.xpos[bid].copy(),d.xmat[bid].reshape(3,3).copy())
            pivot=np.mean([self.points[i][2] for i in group],axis=0);bid,p,R=self.foot[side]['toes'];self.foot[side]['pivot']=(bid,R.T@(pivot-p),pivot)
        self.bottom=.85;self.takeoff=self.bottom+push_duration;self.heel=self.bottom+.5*push_duration;self.tip=self.bottom+(25/28)*push_duration
        self.hz=hz;self.grid=np.unique(np.round(np.r_[np.arange(0,self.takeoff+1/hz/2,1/hz),.2,self.bottom,self.heel,self.tip,self.takeoff],10));self.grid=self.grid[self.grid<=self.takeoff]
        self.zprep=Polynomial(self.C0[2],0,0,self.C0[2]-drop_m,0,0,self.bottom-.2)
        self.zpush=Polynomial(self.C0[2]-drop_m,0,0,self.C0[2]+takeoff_rise,vz,-G,self.takeoff-self.bottom)
        self.yprep=Polynomial(self.C0[1],0,0,self.C0[1]-.06,0,0,self.bottom-.2)
        self.ypush=Polynomial(self.C0[1]-.06,0,0,self.C0[1]-.13,vy,0,self.takeoff-self.bottom)
        centers=np.array([self.com(t) for t in self.grid]);forces=np.array([m.body_mass.sum()*(self.com(t,2)+[0,0,G]) for t in self.grid]);y=np.array([p[2][1] for p in self.points]);fore=np.array([p[3] for p in self.points]);tips=np.array([i in [4,5,10,11] for i in range(12)])
        if forces[:,2].min()<-.001:raise RuntimeError('Scenario needs tensile ground support')
        lo=np.full(len(self.grid),y.min()+.0001);hi=np.full(len(self.grid),y.max()-.0001)
        hi[self.grid>=self.heel]=y[fore].max()-.0001;lo[self.grid>=self.tip]=hi[self.grid>=self.tip]=y[tips].mean()
        lo[self.grid<=.2]=hi[self.grid<=.2]=self.C0[1]
        if not plan_momentum:
            self.force=forces
            self.lo=m.jnt_range[b.independent,0].copy();self.hi=m.jnt_range[b.independent,1].copy()
            return
        cop=[];h=0.;previous_moment=0.
        for i,(C,F) in enumerate(zip(centers,forces)):
            # A causal balance policy, not a future video/pose fit. Support
            # limits constrain COP; resulting angular momentum is retained.
            desired=C[1]-(C[2]*F[1]+8*h)/F[2] if F[2]>1e-5 else (lo[i]+hi[i])/2
            ycop=min(max(desired,lo[i]),hi[i]);moment=(ycop-C[1])*F[2]+C[2]*F[1]
            if i:h+=(moment+previous_moment)*(self.grid[i]-self.grid[i-1])/2
            cop.append(ycop);previous_moment=moment
        cop=np.array(cop)
        moment=(cop-centers[:,1])*forces[:,2]+centers[:,2]*forces[:,1]
        H=cumulative_trapezoid(moment,self.grid,initial=0)
        self.cop=CubicSpline(self.grid,cop);self.H=CubicSpline(self.grid,H)
        self.force=forces
        self.lo=m.jnt_range[b.independent,0].copy();self.hi=m.jnt_range[b.independent,1].copy()
        for i,n in enumerate(self.names):
            if n.startswith('shoulder_elv_'):self.hi[i]=min(self.hi[i],1.8)
            if n.startswith('hip_rotation_'):
                self.lo[i]=max(self.lo[i],self.start[self.ids[i]]-.22);self.hi[i]=min(self.hi[i],self.start[self.ids[i]]+.22)
        movable=('hip_','knee_','ankle_','subtalar_','mtp_angle_','shoulder_elv_','elbow_flexion_')
        self.frozen=[i for i,n in enumerate(self.names) if not n.startswith(movable)]
        print('CENTROIDAL_TASK_PLAN',json.dumps(dict(peak_GRF_N=float(forces[:,2].max()),takeoff_v=self.com(self.takeoff,1).tolist(),takeoff_H=float(self.H(self.takeoff)),COM_start=self.C0.tolist(),COM_end=self.com(self.takeoff).tolist())),flush=True)

    def com(self,t,k=0):
        if t<=.2:return self.C0.copy() if k==0 else np.zeros(3)
        prep=t<=self.bottom;s=t-.2 if prep else t-self.bottom
        return np.array([self.C0[0] if k==0 else 0.,(self.yprep if prep else self.ypush).value(s,k)[0],(self.zprep if prep else self.zpush).value(s,k)[0]])

    def mode(self,t):return 'plant' if t<self.heel-1e-9 else 'fore' if t<self.tip-1e-9 else 'tip'

    def reference(self,t):
        a=self.start[self.ids].copy();low=a.copy();end=a.copy()
        for i,n in enumerate(self.names):
            for stem,value in [('hip_flexion',1.3),('knee_angle',1.5),('shoulder_elv',.08),('elbow_flexion',.2)]:
                if n.startswith(stem+'_'):low[i]=value
            for stem,value in [('hip_flexion',.2),('knee_angle',.30),('ankle_angle',-.3),('mtp_angle',-.25),('shoulder_elv',1.3),('elbow_flexion',.35)]:
                if n.startswith(stem+'_'):end[i]=value
        if t<=.2:return a,0.
        if t<=self.bottom:u=s5((t-.2)/(self.bottom-.2));return a+u*(low-a),.3*u
        u=s5((t-self.bottom)/(self.takeoff-self.bottom));return low+u*(end-low),.3*(1-u)

    def velocity(self,q,t,dt,mode=None,preferred=None,blend=None):
        r,b,m,d=self.r,self.b,self.m,self.d;B,_=r.coordinates(q,np.zeros(r.n));b.pose(q);J=np.zeros((3,m.nv));AH=J.copy();mujoco.mj_jacSubtreeCom(m,d,J,1);mujoco.mj_angmomMat(m,d,AH,1)
        rows=[J@B,AH@B];rhs=[self.com(t,1)+40*(self.com(t)-d.subtree_com[1]),np.array([self.H(t),0.,0.])]
        mode=mode or self.mode(t)
        for side,foot in self.foot.items():
            if mode in ['plant','fore']:
                bid,p,R=foot['calcn' if mode=='plant' else 'toes'];Jp=J.copy();Jr=J.copy();mujoco.mj_jacBody(m,d,Jp,Jr,bid);err=Rotation.from_matrix(R@d.xmat[bid].reshape(3,3).T).as_rotvec()
                rows.extend([Jp@B,Jr@B]);rhs.extend([50*(p-d.xpos[bid]),50*err])
            else:
                bid,local,p0=foot['pivot'];Jp,p=b.jacobian(bid,local);Jr=J.copy();mujoco.mj_jac(m,d,Jp,Jr,p,bid)
                R0=foot['toes'][2];err=Rotation.from_matrix(R0@d.xmat[bid].reshape(3,3).T).as_rotvec()
                rows.extend([Jp@B,(Jr@B)[1:]]);rhs.extend([50*(p0-p),(50*err)[1:]])
        frozen=self.frozen+([self.names.index('mtp_angle_'+s) for s in ['r','l']] if mode=='plant' else [])
        for i in frozen:
            row=np.zeros((1,r.n));row[0,6+i]=1.;rows.append(row);rhs.append(np.array([50*(self.start[self.ids[i]]-q[self.ids[i]])]))
        E=np.vstack(rows);v=np.concatenate(rhs);scale=np.maximum(np.linalg.norm(E,axis=1),1e-6);E/=scale[:,None];v/=scale
        _,tri,piv=qr(E.T,mode='economic',pivoting=True);rank=int((abs(np.diag(tri))>1e-9).sum());keep=piv[:rank]
        ref,pitch=self.reference(t);R=Rotation.from_quat(np.roll(q[3:7],-1));goal=np.r_[np.zeros(3),8*(R.inv()*self.R0*Rotation.from_rotvec([pitch,0,0])).as_rotvec(),20*(ref-q[self.ids])]
        if preferred is not None:goal=preferred+(min(dt/.06,1.) if blend is None else blend)*(goal-preferred)
        weights=np.r_[np.full(3,.01),np.full(3,4.),np.ones(r.n-6)]
        for i,n in enumerate(self.names):
            if n.startswith(('hip_rotation_','hip_adduction_','subtalar_')):weights[6+i]=20.
            if n.startswith(('shoulder_elv_','elbow_flexion_')):weights[6+i]=4.
        C=np.c_[np.zeros((r.n-6,6)),np.eye(r.n-6)]
        lower=35*(self.lo-q[self.ids]);upper=35*(self.hi-q[self.ids])
        try:u=quadprog.solve_qp(np.diag(weights),weights*goal,np.vstack([E[keep],C,-C]).T,np.r_[v[keep],lower,-upper],meq=rank)[0]
        except ValueError:
            raw=quadprog.solve_qp(np.diag(weights),weights*goal,E[keep].T,v[keep],meq=rank)[0]
            bad=np.maximum(lower-raw[6:],raw[6:]-upper);worst=np.argsort(bad)[-5:]
            raise RuntimeError(('Pose/velocity infeasible',t,'H',float(self.H(t)),[(self.names[i],float(q[self.ids[i]]),float(raw[6+i]),float(lower[i]),float(upper[i])) for i in worst]))
        if abs(E@u-v).max()>1e-5:raise RuntimeError(('Velocity constraint residual',t,float(abs(E@u-v).max())))
        return u,B

    def run(self,folder):
        q=self.start.copy();states=[];speeds=[];centers=[];momenta=[];previous=np.zeros(self.r.n);old_mode='plant'
        for i,t in enumerate(self.grid):
            dt=float(self.grid[i+1]-t) if i+1<len(self.grid) else 1/self.hz
            try:u,B=self.velocity(q,float(t),dt,preferred=previous,blend=0. if self.mode(t)!=old_mode else None)
            except Exception as e:
                np.savez_compressed(folder/'partial_support.npz',times=self.grid[:len(states)],qpos=states,u=speeds);raise RuntimeError(('Support solve failed',float(t),str(e))) from e
            self.r.set_state(q,u);mujoco.mj_subtreeVel(self.m,self.d);states.append(q.copy());speeds.append(u.copy());centers.append(self.d.subtree_com[1].copy());momenta.append(self.d.subtree_angmom[1].copy())
            if i%100==0:print('CENTROIDAL_SUPPORT',i,len(self.grid),round(t,3),self.mode(t),np.round(centers[-1],3),flush=True)
            if i+1==len(self.grid):break
            mid=q.copy();mujoco.mj_integratePos(self.m,mid,B@u,dt/2);mid=self.b.complete(mid);um,Bm=self.velocity(mid,float(t+dt/2),dt/2,mode=self.mode(t),preferred=u);new=q.copy();mujoco.mj_integratePos(self.m,new,Bm@um,dt);q=self.b.complete(new);previous=um;old_mode=self.mode(t)
        speeds=np.array(speeds);states=np.array(states);ud=CubicSpline(self.grid,speeds,axis=0)(self.grid,1)
        np.savez_compressed(folder/'dynamic_ground_integrated.npz',times=self.grid,qpos=states,u=speeds,udot=ud,force_world_N=self.force,COM_target=np.array([self.com(t) for t in self.grid]),com=centers,H=momenta,COP_y_m=self.cop(self.grid))
        report=dict(status='centroidal_support_generated_pending_inverse_dynamics',COM_motion_is_task_input=True,GRF_from_COM_acceleration=True,COP_is_support_constrained=True,root_actuator=False,force_clipping=False,strength_ceiling=False,takeoff_velocity_m_s=self.com(self.takeoff,1).tolist(),takeoff_H=float(self.H(self.takeoff)),peak_GRF_N=float(self.force[:,2].max()),maximum_COM_error_m=float(np.linalg.norm(np.array(centers)-np.array([self.com(t) for t in self.grid]),axis=1).max()),integration_hz=self.hz,force_analysis_allowed=False)
        (folder/'support_summary.json').write_text(json.dumps(report,indent=2));print('CENTROIDAL_SUPPORT_COMPLETE',json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='centroidal_support_v1');ap.add_argument('--hz',type=int,default=500);ap.add_argument('--vz',type=float,default=2.5);ap.add_argument('--vy',type=float,default=-.45);args=ap.parse_args();folder=ROOT/'results/physical_jump_task'/args.label;folder.mkdir(parents=True,exist_ok=True)
    Task(MeasuredJumpRig(87,1.78),args.hz,args.vz,args.vy).run(folder)


if __name__=='__main__':main()
