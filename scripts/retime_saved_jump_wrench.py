"""Reduced support-wrench envelope; full 3-D contact allocation is a later check.

Uses the already independently assembled full-body wrenches. Eliminating the
point-force variables avoids near-singular matrices at the final toe line.
"""
from pathlib import Path
import argparse,json
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import osqp
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder');args=ap.parse_args();folder=ROOT/args.folder
    A=sp.load_npz(folder/'timing_constraint_matrix.npz');z=np.load(folder/'timing_constraints.npz');lo=z['lower'];hi=z['upper'];eq=np.isfinite(lo)&(lo==hi);E=A[eq];rhs=lo[eq]
    mid=np.load(folder/'timing_node_metadata.npz')['s'];K=len(mid);N=K+1;assert E.shape[0]==6*K+2
    s=[0.]
    for v in mid:s.append(2*v-s[-1])
    s=np.array(s);B=-E[:6*K,:N].toarray().reshape(K,6,N);w0=rhs[:6*K].reshape(K,6)
    rows=[];lower=[];upper=[]
    def inequality(i,c,lb=0.,ub=np.inf):
        rows.append(c@B[i]);offset=c@w0[i];lower.append(lb-offset);upper.append(ub-offset)
    axis=np.eye(6)
    for i in range(K):
        e=E[6*i:6*i+6];fz=e[2].toarray().ravel();columns=np.flatnonzero(abs(fz[N:]-1)<1e-9)+N
        P=np.array([[-float(e[4,j]),float(e[3,j]),-float(e[3,j-1])] for j in columns])
        if not len(P):raise RuntimeError(('No support points',i))
        xlo,ylo=P[:,:2].min(0);xhi,yhi=P[:,:2].max(0)
        inequality(i,axis[2])
        for phi in np.linspace(0,2*np.pi,16,endpoint=False):inequality(i,.8*np.cos(np.pi/16)*axis[2]-np.cos(phi)*axis[0]-np.sin(phi)*axis[1])
        inequality(i,-axis[4]-xlo*axis[2]);inequality(i,xhi*axis[2]+axis[4])
        if yhi-ylo<1e-5:inequality(i,axis[3]-(yhi+ylo)/2*axis[2],0.,0.)
        else:inequality(i,axis[3]-ylo*axis[2]);inequality(i,yhi*axis[2]-axis[3])
    for j in range(N):
        row=np.zeros(N);row[j]=1;rows.append(row)
        value=rhs[-2] if j==0 else rhs[-1] if j==N-1 else None
        lower.append(value if value is not None else .0001);upper.append(value if value is not None else np.inf)
    C=np.array(rows);l=np.array(lower);u=np.array(upper);ce=np.isfinite(l)&(l==u);iu=np.isfinite(u)&~ce;il=np.isfinite(l)&~ce
    feasible=linprog(np.zeros(N),A_ub=np.vstack([C[iu],-C[il]]),b_ub=np.r_[u[iu],-l[il]],A_eq=C[ce],b_eq=l[ce],bounds=[(None,None)]*N,method='highs')
    report=dict(status=feasible.message,envelope_feasible=bool(feasible.success),full_3d_contact_validated=False,force_analysis_allowed=False)
    if not feasible.success:
        # Keep speed endpoints hard and diagnose, never apply, missing support.
        slack=np.eye(int(ce.sum()));soft=linprog(np.r_[np.zeros(N),np.ones(2*len(C))],A_ub=np.vstack([np.c_[C,-np.eye(len(C)),np.eye(len(C))],np.c_[-C,np.eye(len(C)),-np.eye(len(C))]])[np.r_[np.isfinite(u),np.isfinite(l)]],b_ub=np.r_[u[np.isfinite(u)],-l[np.isfinite(l)]],bounds=[(None,None)]*N+[(0,None)]*(2*len(C)),method='highs') if False else None
        (folder/'envelope_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    D=sp.diags([np.ones(N-1),-np.ones(N-1)],[0,1],shape=(N-1,N));D2=sp.diags([np.ones(N-2),-2*np.ones(N-2),np.ones(N-2)],[0,1,2],shape=(N-2,N))
    Q=sp.eye(N)*.1+D.T@D+D2.T@D2*50
    qp=osqp.OSQP();qp.setup(P=Q.tocsc(),q=np.full(N,-.1),A=sp.csc_matrix(C),l=l,u=u,eps_abs=1e-8,eps_rel=1e-8,verbose=False,max_iter=100000,polishing=True);qp.warm_start(x=feasible.x);result=qp.solve(raise_error=False)
    if not result.info.status.startswith('solved'):raise RuntimeError(result.info.status)
    x=result.x;times=np.r_[0,np.cumsum(2*np.diff(s)/(np.sqrt(x[:-1])+np.sqrt(x[1:])))]
    wrench=w0+np.einsum('ijk,k->ij',B,x);cop_y=np.divide(wrench[:,3],wrench[:,2],out=np.full(K,np.nan),where=wrench[:,2]>1e-8)
    report.update(status='envelope_solved_pending_full_contact_check',duration_s=float(times[-1]),peak_GRF_bodyweights=float(wrench[:,2].max()),minimum_normal_bodyweights=float(wrench[:,2].min()),primal_residual=float(result.info.prim_res),terminal_speed_squared=float(x[-1]),cop_y_m_range=[float(np.nanmin(cop_y)),float(np.nanmax(cop_y))])
    np.savez_compressed(folder/'path_timing.npz',s=s,speed_squared=x,times=times)
    (folder/'envelope_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
