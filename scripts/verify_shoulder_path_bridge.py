"""Unit-tension virtual work, force closure and released shoulder equilibrium."""
import json,numpy as np,mujoco
from scipy.optimize import linprog
from rigid_baseline_profile import RigidBaselineRig
from conservative_ellipsoid_paths import ConservativeEllipsoidPaths as ShoulderPaths,force_basis
from jump_rig import ROOT

def main():
    rig=RigidBaselineRig();b,m,d=rig.b,rig.m,rig.d;bridge=ShoulderPaths(rig);results=[]
    for elev in [.2,.7,1.4]:
        q=b.q0.copy();rig.arm(q,1.4,elev,.25);q=b.complete(q);b.pose(q);B,names=force_basis(rig,q,np.zeros(rig.n));Rb,A,L,paths=bridge.operators();Rn=b.moments();target=np.zeros(m.nv);mujoco.mj_rne(m,d,1,target)
        rng=np.random.default_rng(72);v=B@rng.normal(size=B.shape[1]);eps=1e-6;lengths=[]
        for sign in [-1,1]:
            qp=q.copy();mujoco.mj_integratePos(m,qp,v,sign*eps);d.qpos[:]=qp;d.qvel[:]=0;mujoco.mj_forward(m,d);bridge.update();lengths.append(np.array([mus.getLength(bridge.state) for mus in bridge.muscles]))
        derivative=(lengths[1]-lengths[0])/(2*eps);errors=derivative+Rb.T@v;vw=float(abs(errors).max());print('PATH_ERRORS',[(bridge.names[j],round(float(errors[j]),6)) for j in range(len(errors)) if abs(errors[j])>1e-4],flush=True);b.pose(q)
        R=(B.T@np.column_stack([Rn,Rb]))[6:];tau=(B.T@target)[6:];ref=np.r_[m.actuator_gainprm[:,2],bridge.refs];mat=R*ref;scale=np.maximum(np.linalg.norm(mat,axis=1),1)
        lp=linprog(np.ones(len(ref)),A_eq=mat/scale[:,None],b_eq=tau/scale,bounds=(0,None),method='highs')
        results.append(dict(elevation_rad=elev,virtual_work_max_error_m_s=vw,unit_force_closure=float(abs(A.sum(0)).max()),transform_error=bridge.max_transform_error,force_balance_feasible=bool(lp.success),moment_residual_Nm=float(abs(mat@lp.x-tau).max()) if lp.success else None,maximum_required_force_N=float((lp.x*ref).max()) if lp.success else None))
        print('SHOULDER_CHECK',results[-1],flush=True)
    report=dict(status='passed' if all(x['virtual_work_max_error_m_s']<1e-4 and x['unit_force_closure']<1e-8 and x['force_balance_feasible'] for x in results) else 'failed',muscles=bridge.names,shoulder_force_coordinates=names[-10:],source='https://doi.org/10.3389/fnbot.2019.00090',scope='Registered wrapped-path mechanics; anatomical registration and human recruitment not empirically validated',checks=results)
    (ROOT/'results/jump_v2/shoulder_bridge_checks.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
