"""Integrate foot/COM constraints and externally generated angular momentum.

The ground-force resultant comes from the prescribed COM acceleration. A
feasible pressure distribution selects its centroidal torque; integrated H
then constrains the whole-body velocity. No root motor or torque correction.
"""
import json
import numpy as np,mujoco,quadprog
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from scipy.linalg import qr

def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])

def solve(r,template,com,velocity,acceleration,foot_frames,points,bottom,toe,tip,to,tip_curve,folder,hz=1000,free_arms=False,free_tip_pitch=False):
    b,m,d=r.b,r.m,r.d;mass=float(m.body_mass.sum());W=mass*9.80665;P=np.array([p[2]for p in points]);n=r.n
    times=np.unique(np.round(np.r_[np.arange(0,to,1/hz),bottom,toe,tip,to],10));torques=[]
    reference_H=[]
    for t in times:
        q=template(float(t));v=np.zeros(m.nv);hh=2e-5
        mujoco.mj_differentiatePos(m,v,2*hh,template(float(t-hh)),template(float(t+hh)))
        b.pose(q,v);mujoco.mj_subtreeVel(m,d);reference_H.append(d.subtree_angmom[1].copy())
    H_reference=CubicSpline(times,np.array(reference_H))
    planned_H=H_reference(0).copy();previous_t=0.;previous_torque=np.zeros(3)
    def active(t):return np.arange(len(points))%6>=4 if t>tip else np.array([p[3]for p in points])if t>toe else np.ones(len(points),bool)
    for t in times:
        C=com(float(t));F=mass*(acceleration(float(t))+np.array([0,0,9.80665]));ids=np.flatnonzero(active(t));N=len(ids)
        A=np.tile(np.eye(3),(1,N));T=np.hstack([skew(P[i]-C)for i in ids]);E=A;rhs=F/W;ineq=[]
        for k in range(N):
            for x,y,z in [(0,0,1)]+[(np.cos(p),np.sin(p),.8*np.cos(np.pi/16))for p in np.linspace(0,2*np.pi,16,endpoint=False)]:
                row=np.zeros(3*N);row[k*3:k*3+3]=[x,y,z];ineq.append(row)
        if np.linalg.norm(F)<1e-5:forces=np.zeros((N,3))
        else:
            # Pressure may recover momentum error whenever the support polygon
            # permits it. This is a contact-feasible feedback target, not an
            # extra body torque or a forced reset of angular momentum.
            desired=(H_reference(float(t),1)+10*(H_reference(float(t))-planned_H))/W
            forces=quadprog.solve_qp(T.T@T+1e-7*np.eye(3*N),T.T@desired,np.vstack([E,ineq]).T,np.r_[rhs,np.zeros(len(ineq))],meq=3)[0].reshape(N,3)*W
        torque=np.cross(P[ids]-C,forces).sum(0)
        planned_H+=.5*(previous_torque+torque)*(t-previous_t)
        previous_t=float(t);previous_torque=torque;torques.append(torque)
    hs=CubicSpline(times,np.array(torques)).antiderivative();H0=hs(0)-H_reference(0)
    def momentum(t):return hs(t)-H0
    (folder/'pressure_momentum_plan.json').write_text(json.dumps(dict(status='external_pressure_plan_only',takeoff_H_target=momentum(to).tolist(),takeoff_reference_H=H_reference(to).tolist(),initial_H=momentum(0).tolist(),not_yet_a_full_body_motion=True),indent=2))
    arms=[i for i,name in enumerate(b.ind_names)if name.endswith(('_r','_l'))and name.split('_')[0]in ['elv','shoulder','elbow','pro','deviation','flexion']]
    # Explicit names avoid accidentally freezing ankle/hip flexion coordinates.
    arms=[i for i,name in enumerate(b.ind_names)if any(name==stem+'_'+side for side in ['r','l']for stem in ['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion'])]
    mtp=[b.ind_names.index('mtp_angle_'+s)for s in ['r','l']];spine=b.ind_names.index('flex_extension')
    lo=m.jnt_range[b.independent,0].copy();hi=m.jnt_range[b.independent,1].copy()
    for i,name in enumerate(b.ind_names):
        if name.startswith('shoulder_elv_'):hi[i]=min(hi[i],1.85)
    weights=np.r_[np.full(3,.01),np.full(3,20.),np.ones(n-6)]
    if free_arms:
        for i in arms:weights[6+i]=.3
    G=np.diag(weights)
    max_eq=0.;steps=[];states=[]
    def targets(t):
        use=1 if t>toe else 0;rows=[]
        for side in ['r','l']:
            bid,p,R=foot_frames[side][use];w=np.zeros(3);v=np.zeros(3)
            if t>tip:
                theta=float(tip_curve.value(t-tip)[0]);rate=float(tip_curve.value(t-tip,1)[0]);A=Rotation.from_rotvec([theta,0,0]).as_matrix();w[0]=rate
                center=np.mean([x[2]for j,x in enumerate(points)if x[4]==side and j%6>=4],axis=0);local=R.T@(center-p);R=A@R;p=center-R@local;v=np.cross(w,p-center)
            rows.append((bid,p,R,v,w))
        return rows
    def flow(q,t):
        nonlocal max_eq
        B,_=r.coordinates(q,np.zeros(n));b.pose(q);Jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,Jc,int(m.jnt_bodyid[0]));AH=np.zeros_like(Jc);mujoco.mj_angmomMat(m,d,AH,1)
        E=[Jc@B,AH@B];rhs=[velocity(t)+60*(com(t)-d.subtree_com[1]),momentum(t)]
        cost=G.copy();linear_cost=None
        pitch_rows=[]
        for side,(bid,p,R,v,w) in zip(['r','l'],targets(t)):
            J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jacBody(m,d,J,Jr,bid)
            err=Rotation.from_matrix(R@d.xmat[bid].reshape(3,3).T).as_rotvec()
            if free_tip_pitch and t>tip:
                _,origin,R0=foot_frames[side][1]
                anchor=np.mean([x[2]for j,x in enumerate(points)if x[4]==side and j%6>=4],axis=0)
                local=R0.T@(anchor-origin);actual=d.xpos[bid]+d.xmat[bid].reshape(3,3)@local
                mujoco.mj_jac(m,d,J,Jr,actual,bid)
                E.extend([J@B,(Jr@B)[1:]])
                rhs.extend([60*(anchor-actual),(60*err)[1:]])
                pitch_rows.append((Jr@B)[0])
            else:
                E.extend([J@B,Jr@B]);rhs.extend([v+60*(p-d.xpos[bid]),w+60*err])
        ref=template(t);hh=1e-5;ref_rate=(template(t+hh)[r.ids]-template(t-hh)[r.ids])/(2*hh)
        delta=np.zeros(m.nv);mujoco.mj_differentiatePos(m,delta,1.,q,ref);goal=np.r_[np.zeros(6),ref_rate+3*delta[r.dofs]]
        root_rate=np.zeros(m.nv);mujoco.mj_differentiatePos(m,root_rate,2*hh,template(t-hh),template(t+hh))
        goal[3:6]=root_rate[3:6]+10*delta[3:6]
        if t>bottom:goal[6+spine]=3*(template(bottom)[r.ids[spine]]-q[r.ids[spine]])
        objective=G@goal
        for row in pitch_rows:
            # Smoothly release pitch after the toe patch shrinks to its edge.
            # There is no external support moment about that material edge.
            u=np.clip((t-tip)/.03,0,1);blend=u*u*u*(10-15*u+6*u*u)
            cost+=(1+1e5*(1-blend))*np.outer(row,row)
        fixed_arms=[i for i in arms if b.ind_names[i].startswith(('pro_sup_','deviation_','flexion_'))]if free_arms else arms
        frozen=fixed_arms+mtp
        for i in frozen:
            row=np.zeros((1,n));row[0,6+i]=1;E.append(row);rhs.append(np.array([ref_rate[i]+60*(ref[r.ids[i]]-q[r.ids[i]])]))
        E=np.vstack(E);rhs=np.concatenate(rhs);sc=np.maximum(np.linalg.norm(E,axis=1),1e-5);En=E/sc[:,None];bn=rhs/sc
        _,Rqr,piv=qr(En.T,mode='economic',pivoting=True);rank=int(np.count_nonzero(abs(np.diag(Rqr))>1e-9));keep=piv[:rank]
        ineq=np.vstack([np.c_[np.zeros((n-6,6)),np.eye(n-6)],np.c_[np.zeros((n-6,6)),-np.eye(n-6)]])
        bound=np.r_[60*(lo-q[r.ids]),-60*(hi-q[r.ids])]
        try:u=quadprog.solve_qp(cost,objective,np.vstack([En[keep],ineq]).T,np.r_[bn[keep],bound],meq=rank)[0]
        except ValueError:
            relaxed=quadprog.solve_qp(G,G@goal,En[keep].T,bn[keep],meq=rank)[0]
            misses=np.flatnonzero(ineq@relaxed<bound-1e-7)
            violations=[dict(coordinate=b.ind_names[int(j%(n-6))],angle=float(q[r.ids[j%(n-6)]]),lower=float(lo[j%(n-6)]),upper=float(hi[j%(n-6)]),required_rate=float(relaxed[6+j%(n-6)]))for j in misses]
            (folder/'momentum_velocity_failure.json').write_text(json.dumps(dict(time_s=float(t),H_target=momentum(t).tolist(),qpos=q.tolist(),rank=rank,relaxed_velocity_bound_violations=violations),indent=2));raise
        residual=float(abs(E@u-rhs).max());max_eq=max(max_eq,residual)
        if residual>1e-4:raise RuntimeError(('Velocity constraint residual',t,residual))
        return B@u
    q=template(0);q[:3]=0;b.pose(q);q[:3]+=com(0)-d.subtree_com[1];q=b.complete(q);states.append(q.copy())
    for k,(t,tn)in enumerate(zip(times[:-1],times[1:])):
        dt=float(tn-t);v=flow(q,float(t));mid=q.copy();mujoco.mj_integratePos(m,mid,v,dt/2);mid=b.complete(mid)
        vmid=flow(mid,float((t+tn)/2));qn=q.copy();mujoco.mj_integratePos(m,qn,vmid,dt);q=b.complete(qn);b.pose(q);q[:3]+=com(float(tn))-d.subtree_com[1];states.append(q.copy())
        if k%200==0:print('MOMENTUM_GROUND',k,len(times),round(float(tn),3),flush=True)
    np.savez_compressed(folder/'ground_momentum_targets.npz',times=times,H=np.array([momentum(t)for t in times]),torque=np.array(torques))
    (folder/'momentum_ground_summary.json').write_text(json.dumps(dict(status='integrated_velocity_constraints',max_velocity_equation_residual=max_eq,takeoff_angular_momentum_kg_m2_s=momentum(to).tolist(),external_force_policy='Prescribed COM acceleration; unilateral friction-constrained pressure distribution; H is its integrated torque.',integration_hz=hz),indent=2))
    return times,np.array(states)
