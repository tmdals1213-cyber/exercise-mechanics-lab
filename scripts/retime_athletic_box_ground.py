"""Retiming an already audited support path for an athletic high-box task.

The squared path speed and all sole reactions are solved together. Force,
moment, unilateral pressure and friction are hard conditions. No force cap.
Release can change acceleration, but never resets pose or velocity.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicHermiteSpline
from scipy.integrate import quad
from scipy.optimize import linprog
import osqp

ROOT=Path(__file__).resolve().parents[1];G=9.80665;W=87*G


def skew(p):
    x,y,z=p;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--vz',type=float,default=4.8);ap.add_argument('--end-s',type=float,default=1.032);ap.add_argument('--label',default='yun_160_retimed_v1');args=ap.parse_args()
    src=ROOT/'results/power_comparison/yun_sungbin/forward_v480_coordinated';out=ROOT/'results/athletic_box_jump'/args.label;out.mkdir(parents=True,exist_ok=True)
    z=np.load(src/'motion.npz');keep=np.flatnonzero(z['times']<=args.end_s+1e-8);s=z['times'][keep];N=len(s)
    D=np.zeros((N,N));D[0,:2]=[-1,1]/np.diff(s[:2])[0];D[-1,-2:]=[-1,1]/np.diff(s[-2:])[0]
    for i in range(1,N-1):D[i,i-1]=-1/(s[i+1]-s[i-1]);D[i,i+1]=1/(s[i+1]-s[i-1])
    active=[np.flatnonzero(z['contact_positions_m'][k,:,2]<.00015) for k in keep];offsets=[];nv=N
    for ids in active:offsets.append(nv);nv+=3*len(ids)
    rows=[];cols=[];vals=[];lower=[];upper=[]
    def row(entries,lo,hi):
        i=len(lower)
        for j,v in entries:
            if abs(v)>1e-14:rows.append(i);cols.append(int(j));vals.append(float(v))
        lower.append(lo);upper.append(hi)
    for i,k in enumerate(keep):
        P=z['contact_positions_m'][k];C=z['com'][k];F=z['contact_forces_N'][k];V=z['velocity'][k];H=z['angular_momentum'][k]
        torque=np.cross(P-C,F).sum(0);basis=np.r_[F.sum(0)/W-[0,0,1],torque/W];slope=np.r_[V/(2*G),H/(2*W)]
        A=np.hstack([np.vstack([np.eye(3),skew(P[j]-C)]) for j in active[i]])
        for axis in range(6):
            entries=[(i,-basis[axis])]+[(j,-slope[axis]*D[i,j]) for j in np.flatnonzero(D[i])]+[(offsets[i]+j,A[axis,j]) for j in range(A.shape[1])]
            value=1. if axis==2 else 0.;row(entries,value,value)
        for j in range(len(active[i])):
            off=offsets[i]+3*j;row([(off+2,1)],0,np.inf)
            for phi in np.linspace(0,2*np.pi,12,endpoint=False):row([(off,np.cos(phi)),(off+1,np.sin(phi)),(off+2,-.8*np.cos(np.pi/12))],-np.inf,0)
    terminal=(args.vz/z['velocity'][keep[-1],2])**2
    for i in range(N):
        fixed=1. if s[i]<=.20 else terminal if i==N-1 else None
        row([(i,1)],fixed if fixed is not None else .5,fixed if fixed is not None else 8.)
    A=sp.csc_matrix((vals,(rows,cols)),shape=(len(lower),nv));Ds=sp.csc_matrix(D)
    objective=sp.block_diag([5*sp.eye(N)+.01*Ds.T@Ds,.1*sp.eye(nv-N)],format='csc');linear=np.r_[np.full(N,-5.),np.zeros(nv-N)]
    qp=osqp.OSQP();qp.setup(P=objective,q=linear,A=A,l=np.array(lower),u=np.array(upper),verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=100000,polishing=True)
    result=qp.solve(raise_error=False);report=dict(status=result.info.status,requested_launch_vz_m_s=args.vz,source=str(src),source_motion_sha256=hashlib.sha256((src/'motion.npz').read_bytes()).hexdigest(),strength_ceiling=False,force_clipping=False,root_actuator=False)
    if not result.info.status.startswith('solved'):
        l=np.array(lower);h=np.array(upper);eq=np.isfinite(l)&(l==h);iu=np.isfinite(h)&~eq;il=np.isfinite(l)&~eq
        lp=linprog(np.zeros(nv),A_ub=sp.vstack([A[iu],-A[il]]),b_ub=np.r_[h[iu],-l[il]],A_eq=A[eq],b_eq=l[eq],bounds=[(None,None)]*nv,method='highs')
        report.update(independent_LP_feasible=bool(lp.success),independent_LP_message=lp.message)
        if not lp.success:(out/'retiming_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
        solution=lp.x;report['solver_used']='Independent feasibility LP; no optimality claim'
    else:solution=result.x
    x=solution[:N];xp=D@x;curve=CubicHermiteSpline(s,x,xp)
    dense=np.linspace(s[0],s[-1],5001)
    if curve(dense).min()<=0:raise RuntimeError('Retiming interpolation is not positive')
    times=np.r_[0,np.cumsum([quad(lambda t:1/np.sqrt(curve(t)),a,b,epsabs=1e-10)[0] for a,b in zip(s[:-1],s[1:])])]
    forces=np.zeros((N,12,3))
    for i in range(N):forces[i,active[i]]=W*solution[offsets[i]:offsets[i]+3*len(active[i])].reshape(-1,3)
    u=z['u'][keep]*np.sqrt(x[:,None]);ud=z['udot'][keep]*x[:,None]+.5*xp[:,None]*z['u'][keep]
    np.savez_compressed(out/'dynamic_ground_integrated.npz',times=times,qpos=z['qpos'][keep],u=u,udot=ud,contact_forces_N=forces)
    np.savez_compressed(out/'retiming_path.npz',source_times_s=s,speed_squared=x,speed_squared_derivative=xp,new_times_s=times,source_indices=keep)
    violation=max(float(np.maximum(np.array(lower)-A@solution,0).max()),float(np.maximum(A@solution-np.array(upper),0).max()))
    report.update(status='retimed_ground_candidate_dense_audit_pending',constraint_violation=violation,duration_s=float(times[-1]),takeoff_velocity_m_s=(np.sqrt(x[-1])*z['velocity'][keep[-1]]).tolist(),takeoff_H_kg_m2_s=(np.sqrt(x[-1])*z['angular_momentum'][keep[-1]]).tolist(),takeoff_COM_m=z['com'][keep[-1]].tolist(),peak_GRF_N=float(forces[:,:,2].sum(1).max()),release_GRF_N=forces[-1].sum(0).tolist(),release='Prescribed separating contact-mode transition; velocity continuous, joint acceleration can change.',force_analysis_allowed=False,human_motion_validated=False,box_height_m=1.6)
    (out/'retiming_summary.json').write_text(json.dumps(report,indent=2));print('ATHLETIC_RETIMING',json.dumps(report),flush=True)


if __name__=='__main__':main()
