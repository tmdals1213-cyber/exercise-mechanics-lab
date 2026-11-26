"""Whole-path support feasibility and a hard terminal takeoff velocity.

Build a full-3D contact manifold for a bilateral jump, then solve the squared
path speed and support forces together. Forces are not capped. The objective
only regularizes timing; it is not a physiological effort or fatigue model.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, RotationSpline
import mujoco
import osqp
import quadprog
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from centroidal_jump import Polynomial, rotvec_rate_world
from mechanics_engine_adapter import sample
from build_loaded_squat_power import skew

ROOT = Path(__file__).resolve().parents[1]
G = 9.80665


class SupportPath:
    def __init__(self, rig, vz=2.5, vy=-.45, nodes=121):
        self.r, self.b, self.m, self.d = rig, rig.b, rig.m, rig.d
        r,b,m,d = rig,rig.b,rig.m,rig.d
        self.names=b.ind_names; self.ids=np.array(r.ids)
        self.start,_=r.planted_pose(.10,.15,.07,arms=(np.pi/2,.12,.15),trunk=0.)
        b.pose(self.start)
        self.start[2]-=np.mean([d.site_xpos[m.site('subject_sole_reference_'+s).id,2] for s in ['r','l']])
        b.pose(self.start); self.C0=d.subtree_com[1].copy()
        r.contact_stand_override=self.start
        self.geometry=contact_geometry(r);self.points=self.geometry['points']
        b.pose(self.start)
        self.toes=[(m.body('toes_'+s).id,d.xpos[m.body('toes_'+s).id].copy(),d.xmat[m.body('toes_'+s).id].reshape(3,3).copy()) for s in ['r','l']]
        self.pivots=[np.mean([self.points[i][2] for i in group],axis=0) for group in ([4,5],[10,11])]
        self.pivot_local=[R.T@(p-p0) for (_,p0,R),p in zip(self.toes,self.pivots)]
        self.adjust_names=[stem+'_'+s for s in ['r','l'] for stem in ['hip_adduction','hip_rotation','ankle_angle','subtalar_angle']]
        self.adjust_indices=np.array([self.names.index(n) for n in self.adjust_names]);self.adjust_q=self.ids[self.adjust_indices]
        self.adjust_dof=np.array(r.dofs)[self.adjust_indices]
        self.R0=Rotation.from_quat(np.roll(self.start[3:7],-1))
        self.last=np.r_[self.start[:3],np.zeros(3),self.start[self.adjust_q]]
        ranges=m.jnt_range[b.independent]
        self.lo=np.r_[[-.3,-.8,.25],[-1.,-.15,-.15],ranges[self.adjust_indices,0]]
        self.hi=np.r_[[.3,.4,1.5],[1.,.15,.15],ranges[self.adjust_indices,1]]
        # Native oblique axes require small frontal/transverse adjustments.
        # Do not give the solver large axial spins to satisfy a sagittal task.
        for i,n in enumerate(self.adjust_names):
            if n.startswith('hip_rotation_'):
                self.lo[6+i]=max(self.lo[6+i],self.last[6+i]-.20)
                self.hi[6+i]=min(self.hi[6+i],self.last[6+i]+.20)
        self.design=[stem+'_'+s for s in ['r','l'] for stem in ['hip_flexion','knee_angle','mtp_angle','shoulder_elv','elbow_flexion']]
        self.design_q=np.array([m.jnt_qposadr[b.joints[n]] for n in self.design])
        start=self.start[self.design_q].copy();low=start.copy();end=start.copy()
        for i,n in enumerate(self.design):
            stem=n[:-2]
            low[i]=dict(hip_flexion=1.3,knee_angle=1.5,mtp_angle=0.,shoulder_elv=.08,elbow_flexion=.2)[stem]
            end[i]=dict(hip_flexion=.35,knee_angle=.55,mtp_angle=-.30,shoulder_elv=1.25,elbow_flexion=.35)[stem]
        for values in (start,low,end):
            for n,value in zip(self.design,values):
                bounds=m.jnt_range[b.joints[n]]
                if value<bounds[0]-1e-6 or value>bounds[1]+1e-6:raise ValueError(('Design outside native ROM',n,float(value),bounds.tolist()))
        self.end=self.fit(end,self.C0[:2]+[0,-.065],toe_pitch=.25)
        self.terminal_scale=2.
        uend=self.terminal_velocity(self.end,vz,vy)/self.terminal_scale
        aend=self.terminal_acceleration(self.end,uend*self.terminal_scale)/self.terminal_scale**2
        self.terminal_toe_pitch_rates/=self.terminal_scale
        self.terminal_toe_pitch_accelerations/=self.terminal_scale**2
        self.terminal_u=uend.copy()
        self.bottom=.75;self.to=1.08
        end_rates=np.array([uend[6+self.names.index(n)] for n in self.design])
        prep=Polynomial(start,0,0,low,0,0,self.bottom-.2)
        end_acc=np.array([aend[6+self.names.index(n)] for n in self.design])
        push=Polynomial(low,0,0,end,end_rates,end_acc,self.to-self.bottom)
        self.heel_start=self.bottom+.17
        roll_start=self.bottom+.26
        toe_roll=Polynomial([0.,0.],0,0,[.25,.25],self.terminal_toe_pitch_rates,self.terminal_toe_pitch_accelerations,self.to-roll_start)
        mtps={i:Polynomial(0,0,0,end[i],end_rates[i],end_acc[i],self.to-self.heel_start) for i,n in enumerate(self.design) if n.startswith('mtp_angle_')}
        yprep=Polynomial(self.C0[1],0,0,self.C0[1]+.06,0,0,self.bottom-.2)
        yp=Polynomial(self.C0[1]+.06,0,0,self.C0[1]-.065,vy/self.terminal_scale,0,self.to-self.bottom)
        self.last=np.r_[self.start[:3],np.zeros(3),self.start[self.adjust_q]]
        grid=np.unique(np.r_[np.linspace(0,self.to,nodes),.2,self.bottom,self.to])
        configurations=[];error=0.
        for k,s in enumerate(grid):
            if s<=.2:j=start;cy=self.C0[1];phi=np.zeros(2)
            elif s<=self.bottom:j=prep.value(s-.2);cy=yprep.value(s-.2)[0];phi=np.zeros(2)
            else:
                j=push.value(s-self.bottom);cy=yp.value(s-self.bottom)[0];phi=np.zeros(2) if s<=roll_start else toe_roll.value(s-roll_start)
                for i,p in mtps.items():j[i]=0. if s<=self.heel_start else p.value(s-self.heel_start)[0]
            q=self.fit(j,[self.C0[0],cy],toe_pitch=phi);configurations.append(q);error=max(error,self.last_error)
            if k%40==0:print('SUPPORT_MANIFOLD',k,len(grid),'error',self.last_error,flush=True)
        qs=np.array(configurations)
        # Match the hard terminal velocity in the continuous geometric path.
        bc=lambda v,a: ([(1,np.zeros_like(v)),(2,np.zeros_like(a))],[(1,v),(2,a)])
        self.pos=make_interp_spline(grid,qs[:,:3],k=5,axis=0,bc_type=bc(uend[:3],aend[:3]))
        self.angles=make_interp_spline(grid,qs[:,self.ids],k=5,axis=0,bc_type=bc(uend[6:],aend[6:]))
        rho=(self.R0.inv()*Rotation.from_quat(np.roll(qs[:,3:7],-1,axis=1))).as_rotvec()
        J=self.right_jacobian(rho[-1]);rd=np.linalg.solve(J,uend[3:6]);h=1e-5
        Jdot=(self.right_jacobian(rho[-1]+h*rd)-self.right_jacobian(rho[-1]-h*rd))/(2*h)
        rdd=np.linalg.solve(J,aend[3:6]-Jdot@rd)
        self.rotation_vector=make_interp_spline(grid,rho,k=5,axis=0,bc_type=bc(rd,rdd))
        self.grid=grid;self.max_IK_error=error

    def fit(self,design,com_xy,toe_pitch=0.):
        b,m,d=self.b,self.m,self.d;qbase=self.start.copy();qbase[self.design_q]=design
        pitch=np.broadcast_to(toe_pitch,(2,))
        targets=[]
        for (bid,p,R),pivot,phi in zip(self.toes,self.pivots,pitch):
            tilt=Rotation.from_rotvec([phi,0,0]).as_matrix();targets.append((bid,pivot+tilt@(p-pivot),tilt@R))
        def pose(x):
            q=qbase.copy();q[:3]=x[:3];q[3:7]=np.roll((self.R0*Rotation.from_rotvec(x[3:6])).as_quat(),1);q[self.adjust_q]=x[6:];b.pose(q);return d.qpos.copy()
        def fun(x):
            pose(x);out=[]
            for bid,p,R in targets:
                out.extend(d.xpos[bid]-p);out.extend(Rotation.from_matrix(R.T@d.xmat[bid].reshape(3,3)).as_rotvec())
            out.extend(d.subtree_com[1,:2]-com_xy)
            return np.array(out)
        def jac(x):
            # Use the exact body Jacobians; only root rotvec coordinates need
            # a nontrivial local angular-velocity basis.
            pose(x);V=np.zeros((m.nv,len(x)));V[:3,:3]=np.eye(3)
            R=Rotation.from_rotvec(x[3:6]).as_matrix()
            for i in range(3):V[3:6,3+i]=R.T@rotvec_rate_world(np.eye(3),x[3:6],np.eye(3)[i])
            V[self.adjust_dof,6+np.arange(len(self.adjust_dof))]=1
            rows=[]
            for bid,p,Rt in targets:
                J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jacBody(m,d,J,Jr,bid)
                # Right inverse SO(3) Jacobian in the orientation residual.
                rv=Rotation.from_matrix(Rt.T@d.xmat[bid].reshape(3,3)).as_rotvec();a=np.linalg.norm(rv);K=skew(rv)
                Ji=np.eye(3)-.5*K+(1/12 if a<1e-5 else 1/a**2-(1+np.cos(a))/(2*a*np.sin(a)))*(K@K)
                rows.extend([J@V,Ji@Rt.T@Jr@V])
            J=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,J,1);rows.append((J@V)[:2])
            return np.vstack(rows)
        fit=least_squares(fun,np.clip(self.last,self.lo+1e-8,self.hi-1e-8),jac=jac,bounds=(self.lo,self.hi),max_nfev=70,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        self.last=fit.x.copy();self.last_error=float(abs(fun(fit.x)).max())
        if self.last_error>2e-5:raise RuntimeError(('Contact manifold fit',self.last_error,design.tolist(),fit.x.tolist(),fit.active_mask.tolist(),fit.message,fun(fit.x).tolist()))
        return pose(fit.x)

    def terminal_velocity(self,q,vz,vy):
        r,b,m,d=self.r,self.b,self.m,self.d;B,_=r.set_state(q,np.zeros(r.n));J=np.zeros((3,m.nv));A=J.copy();mujoco.mj_jacSubtreeCom(m,d,J,1);mujoco.mj_angmomMat(m,d,A,1)
        rows=[J@B,A@B];rhs=[np.array([0,vy,vz]),np.zeros(3)]
        toe_jac=[]
        for (bid,p,R),local in zip(self.toes,self.pivot_local):
            Jp,pivot=b.jacobian(bid,local);Jr=J.copy();mujoco.mj_jac(m,d,Jp,Jr,pivot,bid)
            rows.extend([Jp@B,(Jr@B)[1:]]);rhs.extend([np.zeros(3),np.zeros(2)]);toe_jac.append(Jr@B)
        movable=set(self.design+self.adjust_names)
        for i,n in enumerate(self.names):
            if n not in movable:
                row=np.zeros((1,r.n));row[0,6+i]=1;rows.append(row);rhs.append(np.zeros(1))
        E=np.vstack(rows);rhs=np.concatenate(rhs)
        from scipy.linalg import qr
        _,tri,piv=qr(E.T,mode='economic',pivoting=True);keep=piv[:int((abs(np.diag(tri))>1e-9).sum())]
        desired=np.zeros(r.n);weights=np.ones(r.n)
        for i,n in enumerate(self.names):
            if n.startswith('hip_flexion_'):desired[6+i]=-1.
            elif n.startswith('knee_angle_'):desired[6+i]=-2.
            elif n.startswith('mtp_angle_'):desired[6+i]=-5.
            elif n.startswith('shoulder_elv_'):desired[6+i]=1.;weights[6+i]=8
            elif n.startswith('elbow_flexion_'):weights[6+i]=8
        cost=np.diag(weights);objective=weights*desired
        # Native ankle/knee axes are oblique. Small coordinate rates are not
        # equivalent to a quiet transverse bone rotation. Regularize actual
        # bone angular velocity instead of falsely locking compensating DOFs.
        for stem in ['femur','tibia','calcn']:
            for side in ['r','l']:
                Jp=np.zeros((3,m.nv));Jr=Jp.copy();mujoco.mj_jacBody(m,d,Jp,Jr,m.body(stem+'_'+side).id)
                D=(Jr@B)[1:];cost+=50*D.T@D
        C=[E[keep]];v=[rhs[keep]]
        for i,n in enumerate(self.names):
            limit=3. if n.startswith(('shoulder_elv_','elbow_flexion_')) else None
            if limit:
                row=np.zeros(r.n);row[6+i]=1;C.extend([row[None],-row[None]]);v.extend([np.array([-limit]),np.array([-limit])])
        try:u=quadprog.solve_qp(cost,objective,np.vstack(C).T,np.concatenate(v),meq=len(keep))[0]
        except ValueError:
            raw=quadprog.solve_qp(np.diag(weights),weights*desired,E[keep].T,rhs[keep],meq=len(keep))[0]
            print('TERMINAL_UNBOUNDED_DIAGNOSTIC',json.dumps({n:float(raw[6+i]) for i,n in enumerate(self.names) if n in movable}),flush=True)
            raise
        if abs(E@u-rhs).max()>1e-7:raise RuntimeError('Terminal velocity equalities')
        self.terminal_toe_pitch_rates=np.array([J[0]@u for J in toe_jac])
        print('TERMINAL_VELOCITY',json.dumps({n:round(float(u[6+i]),4) for i,n in enumerate(self.names) if n in movable}),flush=True)
        return u

    def state(self,s):
        rho=self.rotation_vector(s);rd=self.rotation_vector(s,1);rdd=self.rotation_vector(s,2);J=self.right_jacobian(rho);h=1e-5
        Jdot=(self.right_jacobian(rho+h*rd)-self.right_jacobian(rho-h*rd))/(2*h)
        q=self.start.copy();q[:3]=self.pos(s);q[3:7]=np.roll((self.R0*Rotation.from_rotvec(rho)).as_quat(),1);q[self.ids]=self.angles(s);q=self.b.complete(q)
        u=np.r_[self.pos(s,1),J@rd,self.angles(s,1)]
        ud=np.r_[self.pos(s,2),J@rdd+Jdot@rd,self.angles(s,2)]
        B,curve=self.r.coordinates(q,u)
        return q,u,ud,B,curve

    @staticmethod
    def right_jacobian(rho):
        a=np.linalg.norm(rho);K=skew(rho)
        return np.eye(3)-(.5 if a<1e-6 else (1-np.cos(a))/a**2)*K+(1/6 if a<1e-6 else (a-np.sin(a))/a**3)*(K@K)

    def terminal_acceleration(self,q,u):
        r,b,m,d=self.r,self.b,self.m,self.d;B,curve=r.set_state(q,u)
        M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M)
        rows=[(M@B)[:6]];rhs=[-(d.qfrc_bias+M@curve)[:6]]
        toe_ops=[]
        for (bid,p,R),local in zip(self.toes,self.pivot_local):
            J,pivot=b.jacobian(bid,local);Jr=J.copy();Jd=J.copy();Jrd=J.copy();mujoco.mj_jac(m,d,J,Jr,pivot,bid);mujoco.mj_jacDot(m,d,Jd,Jrd,pivot,bid)
            rows.extend([J@B,(Jr@B)[1:]]);rhs.extend([-J@curve-Jd@d.qvel,(-Jr@curve-Jrd@d.qvel)[1:]]);toe_ops.append((Jr,Jrd))
        movable=set(self.design+self.adjust_names)
        for i,n in enumerate(self.names):
            if n not in movable:
                row=np.zeros((1,r.n));row[0,6+i]=1;rows.append(row);rhs.append(np.zeros(1))
        E=np.vstack(rows);v=np.concatenate(rhs);scale=np.maximum(np.linalg.norm(E,axis=1),1e-6);E/=scale[:,None];v/=scale
        from scipy.linalg import qr
        _,tri,piv=qr(E.T,mode='economic',pivoting=True);keep=piv[:int((abs(np.diag(tri))>1e-9).sum())]
        goal=-u/.04;goal[:6]=0
        a=quadprog.solve_qp(np.eye(r.n),goal,E[keep].T,v[keep],meq=len(keep))[0]
        if abs(E@a-v).max()>1e-6:raise RuntimeError('Terminal acceleration equality mismatch')
        self.terminal_toe_pitch_accelerations=np.array([(Jr@(B@a+curve)+Jrd@d.qvel)[0] for Jr,Jrd in toe_ops])
        print('TERMINAL_RELEASE_ACCEL',max(abs(a)),flush=True)
        return a


def timing(path,folder,cells=80):
    r,b,m,d=path.r,path.b,path.m,path.d;W=m.body_mass.sum()*G
    ss=np.unique(np.r_[np.linspace(0,path.to,cells+1),.2,path.bottom]);N=len(ss);K=N-1
    # Midpoint collocation. Do not impose three independent equalities on two
    # speed variables when the final support is a line and fixes the COP.
    # A dense independent replay is still required before accepting the path.
    node_data=[];offset=N
    for k in range(K):
        ds=ss[k+1]-ss[k]
        for alpha in (.5,):
            s=ss[k]+alpha*ds;q,u,ud,B,curve=path.state(float(s))
            zero=np.zeros(m.nv);w0=sample(r,q,zero,zero)['body_wrenches'].sum(0)
            w1=sample(r,q,zero,B@u)['body_wrenches'].sum(0)-w0
            w2=sample(r,q,B@u,B@ud+curve)['body_wrenches'].sum(0)-w0
            b.pose(q);P=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,*_ in path.points])
            if P[:,2].min()<-.0001:raise RuntimeError(('Ground penetration in geometric path',s,float(P[:,2].min())))
            active=np.flatnonzero(P[:,2]<1e-5)
            node_data.append((k,alpha,ds,s,w0,w1,w2,P,active,offset));offset+=3*len(active)
        if k%25==0:print('WRENCH_BASIS',k,K,flush=True)
    rows=[];cols=[];vals=[];lower=[];upper=[]
    def row(entries,lo,hi):
        i=len(lower)
        for j,v in entries:
            if v:rows.append(i);cols.append(j);vals.append(float(v))
        lower.append(lo);upper.append(hi)
    for k,alpha,ds,s,w0,w1,w2,P,active,off in node_data:
        L=w2*(1-alpha)-w1/(2*ds);R=w2*alpha+w1/(2*ds)
        A=np.hstack([np.vstack([np.eye(3),skew(P[i])]) for i in active])
        for j in range(6):row([(k,-L[j]/W),(k+1,-R[j]/W)]+[(off+i,A[j,i]) for i in range(A.shape[1])],w0[j]/W,w0[j]/W)
        for j in range(len(active)):
            row([(off+3*j+2,1)],0,np.inf)
            for phi in np.linspace(0,2*np.pi,8,endpoint=False):
                row([(off+3*j,np.cos(phi)),(off+3*j+1,np.sin(phi)),(off+3*j+2,-.8*np.cos(np.pi/8))],-np.inf,0)
    for i in range(N):
        fixed=1. if i==0 else path.terminal_scale**2 if i==N-1 else None
        row([(i,1)],fixed if fixed is not None else .015,fixed if fixed is not None else np.inf)
    A=sp.csc_matrix((vals,(rows,cols)),shape=(len(lower),offset))
    D=sp.diags([np.ones(N-1),-np.ones(N-1)],[0,1],shape=(N-1,N),format='csc')
    D2=D[:-1]@D.T if False else sp.diags([np.ones(N-2),-2*np.ones(N-2),np.ones(N-2)],[0,1,2],shape=(N-2,N),format='csc')
    P=sp.block_diag([sp.eye(N)*.1+D.T@D*1.+D2.T@D2*100.,sp.eye(offset-N)*.0001],format='csc');linear=np.r_[np.full(N,-.1),np.zeros(offset-N)]
    qp=osqp.OSQP();qp.setup(P=P,q=linear,A=A,l=np.array(lower),u=np.array(upper),verbose=False,eps_abs=2e-7,eps_rel=2e-7,max_iter=100000,polishing=True)
    sp.save_npz(folder/'timing_constraint_matrix.npz',A)
    np.savez_compressed(folder/'timing_constraints.npz',lower=lower,upper=upper)
    np.savez_compressed(folder/'timing_node_metadata.npz',s=np.array([n[3] for n in node_data]))
    result=qp.solve(raise_error=False)
    report=dict(status=result.info.status,primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res),hard_terminal_velocity=True,strength_ceiling=False,force_clipping=False)
    if not result.info.status.startswith('solved'):
        from scipy.optimize import linprog
        lo=np.array(lower);hi=np.array(upper);eq=np.isfinite(lo)&(lo==hi);iu=np.isfinite(hi)&~eq;il=np.isfinite(lo)&~eq
        feasibility=linprog(np.zeros(offset),A_ub=sp.vstack([A[iu],-A[il]]),b_ub=np.r_[hi[iu],-lo[il]],A_eq=A[eq],b_eq=lo[eq],bounds=[(None,None)]*offset,method='highs')
        report.update(independent_LP_feasible=bool(feasibility.success),independent_LP_message=feasibility.message)
        ne=int(eq.sum())-2;E=A[eq];slack=sp.vstack([sp.eye(ne),sp.csc_matrix((2,ne))],format='csc')
        relaxed=linprog(np.r_[np.zeros(offset),np.ones(2*ne)],A_ub=sp.hstack([sp.vstack([A[iu],-A[il]]),sp.csc_matrix((int(iu.sum()+il.sum()),2*ne))]),b_ub=np.r_[hi[iu],-lo[il]],A_eq=sp.hstack([E,slack,-slack]),b_eq=lo[eq],bounds=[(None,None)]*offset+[(0,None)]*(2*ne),method='highs')
        if relaxed.success:
            residual=(relaxed.x[offset:offset+ne]-relaxed.x[offset+ne:]).reshape(-1,6);largest=np.argsort(np.linalg.norm(residual,axis=1))[-8:]
            report['diagnostic_only_relaxed_wrench_nodes']=[dict(s=float(node_data[i][3]),residual_bodyweight_units=residual[i].tolist()) for i in largest]
        (folder/'timing_failure.json').write_text(json.dumps(report,indent=2));raise RuntimeError(report)
    x=result.x[:N];dt=2*np.diff(ss)/(np.sqrt(x[:-1])+np.sqrt(x[1:]));times=np.r_[0,np.cumsum(dt)]
    np.savez_compressed(folder/'path_timing.npz',s=ss,speed_squared=x,times=times)
    report.update(duration_s=float(times[-1]),speed_squared_min=float(x.min()),speed_squared_max=float(x.max()),max_constraint_violation=float(max(np.max(np.maximum(np.array(lower)-A@result.x,0)),np.max(np.maximum(A@result.x-np.array(upper),0)))))
    (folder/'timing_summary.json').write_text(json.dumps(report,indent=2));return ss,x,times


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='support_path_v1');ap.add_argument('--vz',type=float,default=2.5);ap.add_argument('--vy',type=float,default=-.45);ap.add_argument('--nodes',type=int,default=121);ap.add_argument('--cells',type=int,default=80);args=ap.parse_args()
    folder=ROOT/'results/physical_jump_task'/args.label;folder.mkdir(parents=True,exist_ok=True)
    r=MeasuredJumpRig(87,1.78);path=SupportPath(r,args.vz,args.vy,args.nodes)
    q,us,uds,B,c=path.state(path.to);r.set_state(q,us*path.terminal_scale);mujoco.mj_subtreeVel(r.m,r.d)
    print('SPATIAL_END_MATCH',r.d.subtree_linvel[1].tolist(),r.d.subtree_angmom[1].tolist(),flush=True)
    np.savez_compressed(folder/'support_geometry.npz',s=path.grid,qpos=np.array([path.state(float(s))[0] for s in path.grid]),terminal_q=q,terminal_u=us*path.terminal_scale)
    ss,x,t=timing(path,folder,args.cells)
    arrays={key:[] for key in ['times','s','qpos','u','udot','com','velocity','H','body_xpos','body_xmat','contact_positions_m','root_wrench']}
    for k in range(len(ss)-1):
        ds=ss[k+1]-ss[k];acc=(x[k+1]-x[k])/(2*ds)
        for frac in (0.,.5):
            s=ss[k]+frac*ds;speed=np.sqrt((1-frac)*x[k]+frac*x[k+1]);q,us,uds,B,c=path.state(float(s));u=us*speed;udot=uds*speed**2+us*acc;r.set_state(q,u);mech=sample(r,q,B@u,B@udot+c*speed**2);mujoco.mj_subtreeVel(r.m,r.d)
            tt=t[k]+(2*ds*frac/(np.sqrt(x[k])+speed))
            P=np.array([r.d.xpos[bid]+r.d.xmat[bid].reshape(3,3)@local for bid,local,*_ in path.points])
            values=dict(times=tt,s=s,qpos=q,u=u,udot=udot,com=r.d.subtree_com[1].copy(),velocity=r.d.subtree_linvel[1].copy(),H=r.d.subtree_angmom[1].copy(),body_xpos=r.d.xpos.copy(),body_xmat=r.d.xmat.copy(),contact_positions_m=P,root_wrench=mech['body_wrenches'].sum(0))
            for key,val in values.items():arrays[key].append(val)
    q,us,uds,B,c=path.state(path.to);us=us*np.sqrt(x[-1]);r.set_state(q,us);mujoco.mj_subtreeVel(r.m,r.d)
    terminal=dict(actual_velocity_m_s=r.d.subtree_linvel[1].tolist(),angular_momentum_kg_m2_s=r.d.subtree_angmom[1].tolist(),requested_velocity_m_s=[0,args.vy,args.vz],IK_max_error_m_or_rad=path.max_IK_error,ground_duration_s=float(t[-1]),force_analysis_allowed=False)
    np.savez_compressed(folder/'support_motion.npz',**{k:np.array(v) for k,v in arrays.items()})
    np.savez_compressed(folder/'support_geometry.npz',s=path.grid,qpos=np.array([path.state(float(s))[0] for s in path.grid]),terminal_q=q,terminal_u=us)
    (folder/'support_summary.json').write_text(json.dumps(terminal,indent=2));print('SUPPORT_PLAN_COMPLETE',json.dumps(terminal),flush=True)


if __name__=='__main__':main()
