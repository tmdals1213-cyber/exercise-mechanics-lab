"""Replay one failed long-jump step; diagnose, never relax the real motion."""
import json,hashlib,numpy as np,mujoco
from scipy.optimize import linprog
from jump_rig import JumpRig,ROOT,OUT
from centroidal_jump import force_plan
from jump_acceleration import AccelerationMotion

def main():
    folder=OUT/'long/acceleration_600hz';old=dict(np.load(folder/'partial_motion.npz'));t=float(old['times'][-1]);setup=json.loads((folder/'setup.json').read_text('utf-8'));hz=setup['physics_hz'];dt=1/hz
    rig=JumpRig();plan=force_plan(rig,'long',hz);motion=AccelerationMotion(rig,plan,'long',hz);q=old['qpos'][-1].copy();u=old['u'][-1].copy();a=old['udot'][-1];B,_=rig.coordinates(q,u)
    report={'status':'diagnostic_only','source_last_valid_time_s':t,'source_sha256':hashlib.sha256((folder/'partial_motion.npz').read_bytes()).hexdigest(),'real_joint_bounds_changed':False,'completed_motion_created':False}
    try:
        mid=q.copy();mujoco.mj_integratePos(rig.m,mid,B@u,dt/2);mid=rig.b.complete(mid);umid=u+dt*a/2
        amid,Bmid,_=motion.acceleration(mid,umid,t+dt/2,motion.mode(t));nq=q.copy();mujoco.mj_integratePos(rig.m,nq,Bmid@umid,dt);nq=rig.b.complete(nq);nu=u+dt*amid
        motion.acceleration(nq,nu,t+dt,motion.mode(t+dt));report['failure_reproduced']=False;report['next']='Re-run full trajectory; one successful replay step is not a completed recovery.'
    except RuntimeError as error:
        fail=dict(np.load(OUT/'debug/acceleration_failure.npz'));A=fail['A'];b=fail['target'];lo=fail['lower'];hi=fail['upper'];n=len(lo)
        assert abs(float(fail['t'])-t)<2*dt+1e-7 and n==rig.n,'Unexpected failure snapshot'
        lp=linprog(np.zeros(n),A_eq=A,b_eq=b,bounds=list(zip(lo,hi)),method='highs')
        report.update(failure_reproduced=True,failure_time_s=float(fail['t']),error=str(error),linear_feasibility_status=int(lp.status),linear_feasibility_message=lp.message)
        if not lp.success:
            # Slack exists only in this diagnostic LP, not in the trajectory.
            il=np.where(np.isfinite(lo))[0];iu=np.where(np.isfinite(hi))[0];k=len(il)+len(iu);U=np.zeros((k,n+k));rhs=[]
            for row,i in enumerate(il):U[row,i]=-1;U[row,n+row]=-1;rhs.append(-lo[i])
            for j,i in enumerate(iu):row=len(il)+j;U[row,i]=1;U[row,n+row]=-1;rhs.append(hi[i])
            costs=np.r_[np.zeros(n),[1/max(1,hi[i]-lo[i]) for i in il],[1/max(1,hi[i]-lo[i]) for i in iu]]
            elastic=linprog(costs,A_ub=U,b_ub=rhs,A_eq=np.c_[A,np.zeros((len(b),k))],b_eq=b,bounds=[(None,None)]*n+[(0,None)]*k,method='highs');relaxed=[]
            if elastic.success:
                aa=elastic.x[:n]
                for j,jid in enumerate(rig.b.independent):
                    i=6+j;v=max(float(lo[i]-aa[i]),float(aa[i]-hi[i]),0.)
                    if v>1e-4:relaxed.append({'joint':rig.m.joint(jid).name,'minimum_diagnostic_acceleration_bound_slack_rad_s2':v,'angle_deg':float(np.rad2deg(fail['q'][rig.ids[j]])),'native_range_deg':np.rad2deg(rig.m.jnt_range[jid]).tolist()})
            report.update(diagnostic_slack_problem_solved=bool(elastic.success),bound_conflicts=sorted(relaxed,key=lambda x:-x['minimum_diagnostic_acceleration_bound_slack_rad_s2']),next='Revise the prescribed landing/recovery task and contact compatibility; do not drop the real joint bounds to export a successful video.')
        else:report['next']='Investigate QP conditioning; LP feasibility alone does not validate the trajectory.'
    (folder/'recovery_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
