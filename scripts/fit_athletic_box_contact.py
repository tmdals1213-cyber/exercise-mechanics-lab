"""Fit a high-box contact configuration at the actual ballistic COM state.

Root attitude is held to the previous conservative flight solution. The
result is a shooting target, not a substituted flight pose or applied force.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,RotationSpline
from retarget_measured_jump import MeasuredJumpRig
from high_flex_profile import apply
from build_contact_jump import contact_geometry

ROOT=Path(__file__).resolve().parents[1]


def skew(x):
    a,b,c=x;return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])


def main():
    p=argparse.ArgumentParser();p.add_argument('flight');p.add_argument('--output',required=True);p.add_argument('--time',type=float,default=.55);p.add_argument('--advance',type=float,default=.15);p.add_argument('--height',type=float,default=1.6);p.add_argument('--toeout',type=float,default=15.);args=p.parse_args()
    folder=ROOT/args.flight;out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True);a=np.load(folder/'motion.npz');meta=json.loads((folder/'flight_audit.json').read_text());r=MeasuredJumpRig(87,1.78);apply(r);b,m,d=r.b,r.m,r.d
    time=meta['takeoff_s']+args.time;seed=np.array([np.interp(time,a['times'],a['qpos'][:,j]) for j in range(m.nq)]);seed[3:7]=np.roll(RotationSpline(a['times'],Rotation.from_quat(np.roll(a['qpos'][:,3:7],-1,axis=1)))(time).as_quat(),1)
    C=np.array(meta['takeoff_COM_m'])+args.time*np.array(meta['takeoff_velocity_m_s'])+[0,0,-.5*9.80665*args.time**2]
    r.contact_stand_override=a['qpos'][0];points=contact_geometry(r)['points'];P0=np.array([p[2] for p in points]);target=P0.copy()
    for first,last,sign in [(0,6,-1),(6,12,1)]:
        target[first:last]-=P0[first:last].mean(0);target[first:last]=Rotation.from_rotvec([0.,0.,sign*np.deg2rad(args.toeout)]).apply(target[first:last]);target[first:last]+=np.array([C[0]+sign*.13,C[1]-args.advance,args.height])
    b.pose(a['qpos'][0]);torso=m.body('torso').id;local_up=d.xmat[torso].reshape(3,3).T@[0,0,1.];wanted=Rotation.from_rotvec([.65,0,0]).apply([0,0,1.])
    selected=[i for i,n in enumerate(b.ind_names) if n.startswith(('hip_','knee_','ankle_','subtalar_','mtp_angle_')) or n=='flex_extension'];qids=np.array(r.ids)[selected];jid=np.array(b.independent)[selected];prior=seed[qids].copy();lo=m.jnt_range[jid,0]+1e-6;hi=m.jnt_range[jid,1]-1e-6
    for k,i in enumerate(selected):
        if b.ind_names[i]=='flex_extension':lo[k]=max(lo[k],-.6);hi[k]=min(hi[k],.6)
    cache={}
    def evaluate(x):
        if cache.get('key')==x.tobytes():return cache['value']
        q=seed.copy();q[qids]=x;q[:3]=0;q=b.complete(q);b.pose(q);q[:3]=C-d.subtree_com[1];b.pose(q)
        B,_=r.coordinates(q,np.zeros(r.n));columns=B[:,6+np.array(selected)].copy();b.pose(q);Jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,Jc,1);columns[:3]-=Jc@columns
        residual=[];rows=[];P=[]
        for index,(bid,local,*_) in enumerate(points):
            J,p=b.jacobian(bid,local);residual.extend(1000*(p-target[index]));rows.append(1000*J@columns);P.append(p.copy())
        jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,torso);up=d.xmat[torso].reshape(3,3)@local_up
        residual.extend(5*(up-wanted));rows.append(5*(-skew(up))@jr@columns)
        residual.extend(.1*(x-prior));rows.append(.1*np.eye(len(x)))
        cache['key']=x.tobytes();cache['value']=(np.array(residual),np.vstack(rows),q.copy(),np.array(P));return cache['value']
    initial=np.clip(prior,lo+1e-6,hi-1e-6);direction=np.linspace(-.2,.3,len(initial));eps=1e-6
    fd=(evaluate(initial+eps*direction)[0]-evaluate(initial-eps*direction)[0])/(2*eps);jacerr=float(abs(fd-evaluate(initial)[1]@direction).max());assert jacerr<.01,jacerr
    opt=least_squares(lambda x:evaluate(x)[0],initial,jac=lambda x:evaluate(x)[1],bounds=(lo,hi),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-9)
    residual,jac,q,P=evaluate(opt.x);err=float(np.linalg.norm(P-target,axis=1).max());report=dict(status='contact_configuration_fitted' if err<.003 else 'contact_configuration_not_fitted',maximum_foot_error_m=err,COM_m=C.tolist(),contact_after_takeoff_s=args.time,foot_advance_from_COM_m=args.advance,root_attitude_is_previous_flight=True,shooting_consistency_verified=False,source_flight=args.flight,jacobian_directional_error=jacerr,optimizer=opt.message,selected_coordinates=[b.ind_names[i] for i in selected],values_rad=opt.x.tolist())
    np.savez_compressed(out/'contact_pose.npz',qpos=q,target_foot_points_m=target,COM_m=C,contact_after_takeoff_s=args.time);(out/'contact_pose_report.json').write_text(json.dumps(report,indent=2));print('CONTACT_POSE',json.dumps(report),flush=True)


if __name__=='__main__':main()
