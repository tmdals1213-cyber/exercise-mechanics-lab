"""Rigid-contact velocity planning with exact COM and zero centroidal H tasks."""
from pathlib import Path
import argparse,json,traceback
import numpy as np,mujoco,quadprog
from scipy.linalg import qr
from scipy.spatial.transform import Rotation
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from centroidal_jump import Polynomial,s5

ROOT=Path(__file__).resolve().parents[1];G=9.80665

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hz',type=int,default=250);ap.add_argument('--height-cm',type=float,default=30)
    ap.add_argument('--drop',type=float,default=.20);ap.add_argument('--rise',type=float,default=.07);ap.add_argument('--label',default='balanced_250hz')
    ap.add_argument('--momentum-peak',type=float,default=6.)
    args=ap.parse_args();out=ROOT/'results/power_comparison/cmj_adult_target'/args.label;out.mkdir(parents=True,exist_ok=True)
    r=MeasuredJumpRig();b,m,d=r.b,r.m,r.d;geometry=contact_geometry(r);q0=geometry['stand'].copy()
    source=np.load(ROOT/'results/power_comparison/cmj_analysis/motion.npz')['qpos'][0]
    free=[i for i,n in enumerate(b.ind_names) if n.startswith(('hip_','knee_','ankle_','subtalar_','mtp_')) or n in ['flex_extension','lat_bending','axial_rotation']]
    held=[i for i in range(len(b.ind_names)) if i not in free]
    for i in held:q0[r.ids[i]]=source[r.ids[i]]
    q0=b.complete(q0)
    b.pose(q0);c0=d.subtree_com[1].copy();mass=float(m.body_mass.sum());W=mass*G
    toes=[m.body('toes_'+s).id for s in ['r','l']];foot_positions=[d.xpos[j].copy() for j in toes];foot_rotations=[d.xmat[j].reshape(3,3).copy() for j in toes]
    bottom=.85;start=.25;launch=np.sqrt(2*G*args.height_cm/100);stroke=args.drop+args.rise
    T=float(max(np.roots([G/12,.5*launch,-stroke])));takeoff=bottom+T;end=takeoff+2*launch/G*.92
    initial_toe=np.mean([p[2][1] for p in geometry['points'] if p[3]])+.01
    forward_start=.50
    forward=Polynomial(c0[1],0,0,initial_toe,0,0,takeoff-forward_start)
    descend=Polynomial(c0[2],0,0,c0[2]-args.drop,0,0,bottom-start)
    mtp={s:b.ind_names.index('mtp_angle_'+s) for s in ['r','l']}
    roll={s:Polynomial(q0[r.ids[i]],0,0,-.40,-3.,0,T) for s,i in mtp.items()}
    low=q0.copy();ext=q0.copy()
    for s in ['r','l']:
        for name,val in [('hip_flexion',1.0),('knee_angle',1.5),('ankle_angle',.45)]:low[m.jnt_qposadr[b.joints[name+'_'+s]]]=val
        for name,val in [('hip_flexion',.08),('knee_angle',.25),('ankle_angle',-.35),('mtp_angle',-.6)]:ext[m.jnt_qposadr[b.joints[name+'_'+s]]]=val
    low[m.jnt_qposadr[b.joints['flex_extension']]]=-.15
    release=None;last_qp_error=None
    def COM(t,k=0):
        if t>=takeoff and release is not None and 'com' in release:
            ss=t-takeoff
            return release['com']+ss*release['comvel']+.5*ss*ss*np.array([0,0,-G]) if k==0 else release['comvel']+ss*np.array([0,0,-G]) if k==1 else np.array([0,0,-G])
        if t<=start:return c0.copy() if k==0 else np.zeros(3)
        yy=(float(c0[1]) if k==0 else 0.) if t<=forward_start else float(forward.value(min(t-forward_start,takeoff-forward_start),k)[0])
        if t>=takeoff and k:yy=0.
        if t<bottom:zz=float(descend.value(t-start,k)[0])
        elif t<=takeoff:
            s=(t-bottom)/T;C=np.pi/2*(launch/(G*T)+.5)
            vals=[c0[2]-args.drop+G*T*T*(-s**3/6+C*(s/np.pi-np.sin(np.pi*s)/np.pi**2)),
                G*T*(-s*s/2+C*(1-np.cos(np.pi*s))/np.pi),G*(-s+C*np.sin(np.pi*s))]
            zz=vals[k]
        else:
            ss=t-takeoff;zz=(c0[2]+args.rise+launch*ss-.5*G*ss*ss) if k==0 else launch-G*ss if k==1 else -G
        return np.array([c0[0] if k==0 else 0.,yy,zz])
    def reference(t):
        if t<=start:return q0.copy(),np.zeros(m.nq)
        if t<bottom:
            p=Polynomial(q0,0,0,low,0,0,bottom-start);return p.value(t-start),p.value(t-start,1)
        p=Polynomial(low,0,0,ext,0,0,T);return p.value(min(t-bottom,T)),p.value(min(t-bottom,T),1)
    def solve(q,t,Hstate=None,free_momentum=False):
        nonlocal last_qp_error
        b.pose(q);B,_=r.coordinates(q,np.zeros(r.n));rows=[];rhs=[]
        goal,qd=reference(t);uref=np.r_[np.zeros(6),qd[r.ids]+8*(goal[r.ids]-q[r.ids])]
        phase=0. if t<=start else float(s5((t-start)/(bottom-start))) if t<bottom else 1-float(s5(min(1.,(t-bottom)/T)))
        phase_rate=0.
        if start<t<bottom:
            ss=(t-start)/(bottom-start);phase_rate=30*ss**2*(1-ss)**2/(bottom-start)
        elif bottom<t<takeoff:
            ss=(t-bottom)/T;phase_rate=-30*ss**2*(1-ss)**2/T
        Rgoal=Rotation.from_quat(np.roll(q0[3:7],-1))*Rotation.from_rotvec([.50*phase,0,0])
        Rnow=Rotation.from_quat(np.roll(q[3:7],-1));uref[3:6]=np.array([.50*phase_rate,0,0])+8*(Rnow.inv()*Rgoal).as_rotvec()
        if t>=takeoff and release is not None:
            ss=t-takeoff;blend=float(s5(min(1,ss/.08)))
            uref=release['u']*(1-blend)
            air=release['q'].copy()
            for s in ['r','l']:
                air[m.jnt_qposadr[b.joints['knee_angle_'+s]]]=.35
                air[m.jnt_qposadr[b.joints['hip_flexion_'+s]]]=.10
                air[m.jnt_qposadr[b.joints['ankle_angle_'+s]]]=-.1
                air[m.jnt_qposadr[b.joints['mtp_angle_'+s]]]=-.25
            uref[6:]+=blend*5*(air[r.ids]-q[r.ids])
        if t<takeoff-1e-10:
            for j,p0,R0 in zip(toes,foot_positions,foot_rotations):
                J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,d.xpos[j],j)
                rows.extend([J@B,Jr@B]);rhs.extend([30*(p0-d.xpos[j]),30*Rotation.from_matrix(R0@d.xmat[j].reshape(3,3).T).as_rotvec()])
            for s,i in mtp.items():
                angle=q0[r.ids[i]] if t<=bottom else float(roll[s].value(t-bottom)[0]);vel=0 if t<=bottom else float(roll[s].value(t-bottom,1)[0])
                row=np.zeros((1,r.n));row[0,6+i]=1;rows.append(row);rhs.append(np.array([vel+30*(angle-q[r.ids[i]])]))
        Jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,Jc,1);com_rhs=COM(t,1)+30*(COM(t)-d.subtree_com[1])
        rows.append((Jc@B) if t>=takeoff else (Jc@B)[2:3]);rhs.append(com_rhs if t>=takeoff else com_rhs[2:3])
        H=np.zeros((3,m.nv));mujoco.mj_angmomMat(m,d,H,1)
        if not free_momentum:rows.append(H@B);rhs.append(np.zeros(3) if Hstate is None else Hstate)
        for i in held:
            row=np.zeros((1,r.n));row[0,6+i]=1;rows.append(row);rhs.append(np.array([20*(q0[r.ids[i]]-q[r.ids[i]])]))
        A=np.vstack(rows);bb=np.concatenate(rhs);sc=np.maximum(np.linalg.norm(A,axis=1),1e-5);A/=sc[:,None];bb/=sc
        _,R,piv=qr(A.T,mode='economic',pivoting=True);rank=int(np.sum(abs(np.diag(R))>1e-8));keep=piv[:rank]
        inequalities=[];limits=[]
        for i,j in enumerate(b.independent):
            if not m.jnt_limited[j]:continue
            lo,hi=m.jnt_range[j];name=b.ind_names[i]
            if name=='flex_extension':lo,hi=max(lo,-.45),min(hi,.45)
            elif name in ['lat_bending','axial_rotation']:lo,hi=q0[r.ids[i]]-.15,q0[r.ids[i]]+.15
            row=np.zeros(r.n);row[6+i]=1;inequalities.extend([row,-row]);limits.extend([60*(lo+1e-6-q[r.ids[i]]),60*(q[r.ids[i]]-hi+1e-6)])
        weights=np.r_[np.full(3,.1),np.full(3,100.),np.ones(r.n-6)]
        for i,n in enumerate(b.ind_names):
            if n in ['flex_extension','lat_bending','axial_rotation']:weights[6+i]=10
        C=np.vstack([A[keep],inequalities]);target=np.r_[bb[keep],limits]
        Pobjective=np.diag(weights);linear=weights*uref
        if t<takeoff:
            XY=(Jc@B)[:2];wanted=COM(t,1)[:2]+6*(COM(t)-d.subtree_com[1])[:2]
            Pobjective+=2*XY.T@XY;linear+=2*XY.T@wanted
        try:u=quadprog.solve_qp(Pobjective,linear,C.T,target,meq=rank)[0]
        except ValueError as e:
            last_qp_error=dict(time_s=float(t),rank=rank,n=r.n,reason=str(e));raise
        if abs(A@u-bb).max()>1e-6:raise RuntimeError(('Velocity closure',t,float(abs(A@u-bb).max())))
        return u,B
    contact_points=geometry['points']
    def flow(q,H,t):
        if t<takeoff:
            natural,B=solve(q,t,free_momentum=True)
            mat=np.zeros((3,m.nv));mujoco.mj_angmomMat(m,d,mat,1);Hgoal=mat@B@natural
            if t>takeoff-.10:Hgoal*=1-float(s5((t-(takeoff-.10))/.10))
        u,B=solve(q,t,H);v=B@u
        eps=1e-4
        def field(qq,HH,tt):
            uv,Bv=solve(qq,tt,HH);return Bv@uv
        qp=q.copy();qm=q.copy();mujoco.mj_integratePos(m,qp,v,eps);mujoco.mj_integratePos(m,qm,v,-eps)
        a0=(field(b.complete(qp),H,t+eps)-field(b.complete(qm),H,t-eps))/(2*eps)
        DH=np.zeros((m.nv,3))
        if t<takeoff:
            for j in range(3):
                hh=np.zeros(3);hh[j]=1e-3
                DH[:,j]=(field(q,H+hh,t)-field(q,H-hh,t))/(2e-3)
        b.pose(q,v);Jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,Jc,1)
        positions=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,_,_ in contact_points])
        forces=np.zeros_like(positions)
        if t>=takeoff:return u,B,np.zeros(3),forces,positions,a0
        active=[i for i,p in enumerate(contact_points) if t<bottom or p[3]]
        rows=[]
        for i in active:
            x,y,zv=positions[i]-d.subtree_com[1];rows.append(np.array([[0,-zv,y],[zv,0,-x],[-y,x,0]]))
        M=np.concatenate(rows,axis=1);inequalities=[];Jext=[]
        for i in active:
            J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,positions[i],contact_points[i][0]);Jext.append(J)
        external_map=np.concatenate([J[:,:6].T for J in Jext],axis=1)
        MM=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,MM)
        d.qacc[:]=a0;bias=np.zeros(m.nv);mujoco.mj_rne(m,d,1,bias)
        E=external_map-MM[:6]@DH@M
        _,rr,piv=qr(E.T,mode='economic',pivoting=True);rank=int(np.sum(abs(np.diag(rr))>1e-5));keep=piv[:rank]
        scale=np.maximum(np.linalg.norm(E[keep],axis=1),1e-8);eq=E[keep]/scale[:,None];beq=bias[keep]/W/scale
        for j in range(len(active)):
            for x,y,zv in [(0,0,1)]+[(float(np.cos(p)),float(np.sin(p)),float(.8*np.cos(np.pi/16))) for p in np.linspace(0,2*np.pi,16,endpoint=False)]:
                row=np.zeros(3*len(active));row[3*j:3*j+3]=[x,y,zv];inequalities.append(row)
        torque=40*(Hgoal-H)
        try:
            pressure=quadprog.solve_qp(M.T@M+1e-5*np.eye(E.shape[1]),M.T@(torque/W),
                np.vstack([eq,inequalities]).T,np.r_[beq,np.zeros(len(inequalities))],meq=rank)[0]
        except ValueError:
            ftest=np.linalg.lstsq(E,bias[:6]/W,rcond=1e-5)[0].reshape(-1,3)*W
            detail=dict(time_s=float(t),rank=rank,singular_values=np.linalg.svd(E,compute_uv=False).tolist(),
                implied_total_force_N=ftest.sum(0).tolist(),root_bias=bias[:6].tolist(),H=H.tolist(),Hgoal=Hgoal.tolist())
            (out/'pressure_failure.json').write_text(json.dumps(detail,indent=2));print('PRESSURE_FAILURE',detail,flush=True);raise
        forces[active]=(W*pressure).reshape(-1,3);Hdot=W*M@pressure
        acceleration=a0+DH@Hdot
        d.qacc[:]=acceleration;check=np.zeros(m.nv);mujoco.mj_rne(m,d,1,check)
        residual=check[:6]-external_map@(W*pressure)
        if abs(residual).max()>.1:raise RuntimeError(('Coupled root closure',t,float(abs(residual).max()),rank))
        return u,B,Hdot,forces,positions,acceleration
    times=np.unique(np.r_[np.arange(0,end,1/args.hz),start,bottom,takeoff,end]);q=q0.copy();Hstate=np.zeros(3);states=[]
    try:
        for k,t in enumerate(times):
            if t>=takeoff and release is None:
                # Limit from the last contact-constrained instant.
                uto,Bto=solve(q,takeoff-1e-9,Hstate);b.pose(q,Bto@uto);mujoco.mj_subtreeVel(m,d)
                release=dict(q=q.copy(),u=uto.copy(),com=d.subtree_com[1].copy(),comvel=d.subtree_linvel[1].copy())
            u,B,Hdot,F,P,acc=flow(q,Hstate,float(t));states.append((t,q.copy(),(B@u).copy(),u.copy(),Hstate.copy(),F.copy(),P.copy(),Hdot.copy(),acc.copy()))
            if k%max(1,args.hz//2)==0:print('BALANCED_IK',round(float(t),3),'COM',d.subtree_com[1].round(3),'speed',round(float(abs(u).max()),2),flush=True)
            if k+1==len(times):break
            dt=times[k+1]-t;qm=q.copy();mujoco.mj_integratePos(m,qm,B@u,dt/2);qm=b.complete(qm)
            um,Bm,Hdm,_,_,_=flow(qm,Hstate+dt*Hdot/2,float(t+dt/2));qn=q.copy();mujoco.mj_integratePos(m,qn,Bm@um,dt);q=b.complete(qn);Hstate+=dt*Hdm
    except Exception:
        if states:np.savez_compressed(out/'rejected_kinematics.npz',times=[x[0] for x in states],qpos=[x[1] for x in states],qvel=[x[2] for x in states])
        (out/'failure.txt').write_text(traceback.format_exc());print('IK_FAILURE',last_qp_error,flush=True);raise
    np.savez_compressed(out/'planned_kinematics.npz',times=[x[0] for x in states],qpos=[x[1] for x in states],qvel=[x[2] for x in states],u=[x[3] for x in states],
        angular_momentum_state=[x[4] for x in states],contact_forces_N=[x[5] for x in states],contact_positions_m=[x[6] for x in states],qacc=[x[8] for x in states])
    (out/'plan.json').write_text(json.dumps(dict(takeoff_s=takeoff,bottom_s=bottom,duration_s=end,push_duration_s=T,
        target_height_cm=args.height_cm,drop_m=args.drop,rise_m=args.rise,COM0=c0.tolist(),source='Velocity-level COM / rigid feet / zero angular momentum planning.'),indent=2))
    print('BALANCED_IK_COMPLETE',takeoff,end,flush=True)

if __name__=='__main__':main()
