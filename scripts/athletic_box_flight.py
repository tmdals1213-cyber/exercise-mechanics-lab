"""High-box flight from a solved takeoff, with conserved centroidal momentum.

Internal joint coordination is an input. Root attitude follows conserved
angular momentum; root translation follows ballistic COM. Independent inverse
dynamics and velocity-derivative checks audit this conservative reconstruction.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,hashlib
import numpy as np,mujoco,quadprog
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from centroidal_jump import Polynomial
from mechanics_engine_adapter import sample
from high_flex_profile import apply
from box_geometry_audit import segments,minimum_clearance

ROOT=Path(__file__).resolve().parents[1];G=9.80665


def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--label',default='yun_160_flight_v1');p.add_argument('--duration',type=float,default=.75);p.add_argument('--hip',type=float,default=2.05);p.add_argument('--knee',type=float,default=2.40);p.add_argument('--ankle',type=float);p.add_argument('--shoulder',type=float,default=.65);p.add_argument('--elbow',type=float,default=.4);p.add_argument('--lumbar',type=float);p.add_argument('--fold-time',type=float,default=.22);p.add_argument('--reach-time',type=float,default=.43);p.add_argument('--height',type=float,default=1.6);args=p.parse_args()
    source=ROOT/args.source;out=ROOT/'results/athletic_box_jump'/args.label;out.mkdir(parents=True,exist_ok=True)
    z=np.load(source/'dynamic_ground_integrated.npz');r=MeasuredJumpRig(87,1.78);profile=apply(r);b,m,d=r.b,r.m,r.d;names=b.ind_names;ids=np.array(r.ids)
    release=np.flatnonzero((z['times'][:-1]>.85)&(z['contact_forces_N'][:,:,2].sum(1)<1e-4));k0=int(release[0]) if len(release) else len(z['times'])-1
    q0=z['qpos'][k0].copy();u0=z['u'][k0].copy();to=float(z['times'][k0]);r.set_state(q0,u0);mujoco.mj_subtreeVel(m,d);C0=d.subtree_com[1].copy();V0=d.subtree_linvel[1].copy();H0=d.subtree_angmom[1].copy()
    r.contact_stand_override=z['qpos'][0];points=contact_geometry(r)['points'];r.set_state(q0,u0)
    angle=q0[ids];rate=u0[6:];lo=m.jnt_range[b.independent,0];hi=m.jnt_range[b.independent,1]
    margin=np.where(rate<0,angle-lo,hi-angle);brake=np.minimum(.035,1.6*np.maximum(margin,1e-9)/np.maximum(abs(rate),1e-9))
    if brake.min()<.002:raise RuntimeError('No smooth release within the source joint ranges')
    stop=angle+.5*brake*rate;desired=-rate/brake;B,curve=r.set_state(q0,u0);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);K=M@B
    root=-np.linalg.solve(K[:6,:6],(d.qfrc_bias+M@curve)[:6]+K[:6,6:]@desired);base=B@np.r_[root,desired]+curve
    chosen=[i for i,n in enumerate(names) if n.startswith(('hip_flexion','knee_angle','ankle_angle','mtp_angle'))];D=np.zeros((r.n,len(chosen)));D[6+np.array(chosen),np.arange(len(chosen))]=1;D[:6]=-np.linalg.solve(K[:6,:6],K[:6,6+np.array(chosen)])
    rows=[];limits=[]
    for bid,local,*_ in points:
        J,P=b.jacobian(bid,local)
        if P[2]>.003:continue
        jd=J.copy();jrd=J.copy();mujoco.mj_jacDot(m,d,jd,jrd,P,bid);rows.append((J@B@D)[2]);limits.append(1.-float((J@base+jd@d.qvel)[2]))
    if rows:desired[chosen]+=quadprog.solve_qp(np.eye(len(chosen)),np.zeros(len(chosen)),np.array(rows).T,np.array(limits))[0]
    fold=stop.copy();end=stop.copy()
    landing_q,_=r.planted_pose(args.hip,args.knee,.12,arms=(np.pi/2,.65,.4),trunk=0.)
    for i,name in enumerate(names):
        if name.startswith(('hip_adduction','hip_rotation','ankle_angle','subtalar_angle')):fold[i]=end[i]=landing_q[ids[i]]
    for side in ['r','l']:
        for n,v,w in [('hip_flexion',.75,args.hip),('knee_angle',args.knee,args.knee),('mtp_angle',0.,0.),('shoulder_elv',args.shoulder,args.shoulder),('elbow_flexion',args.elbow,args.elbow)]:
            i=names.index(n+'_'+side);fold[i]=v;end[i]=w
        if args.ankle is not None:end[names.index('ankle_angle_'+side)]=args.ankle
    if args.lumbar is not None:
        i=names.index('flex_extension');fold[i]=(stop[i]+args.lumbar)/2;end[i]=args.lumbar
    extension_time=.10
    extension=stop.copy()
    for side in ['r','l']:
        for n,value in [('hip_flexion',.15),('knee_angle',.18),('ankle_angle',-.20),('mtp_angle',0.)]:extension[names.index(n+'_'+side)]=value
        fold[names.index('hip_flexion_'+side)]=.30
        for n in ['shoulder_elv','elbow_flexion']:
            i=names.index(n+'_'+side);extension[i]=(stop[i]+fold[i])/2
    paths=[]
    for i in range(len(names)):
        paths.append((Polynomial(angle[i],rate[i],desired[i],stop[i],0,0,brake[i]),Polynomial(stop[i],0,0,extension[i],0,0,extension_time-brake[i]),Polynomial(extension[i],0,0,fold[i],0,0,args.fold_time-extension_time),Polynomial(fold[i],0,0,end[i],0,0,args.reach_time-args.fold_time)))
    def joints(t):
        return np.array([float((a.value(t) if t<brake[i] else c.value(t-brake[i]) if t<extension_time else f.value(t-extension_time) if t<args.fold_time else e.value(t-args.fold_time) if t<args.reach_time else np.array([end[i]]))[0]) for i,(a,c,f,e) in enumerate(paths)])
    grid=np.unique(np.r_[np.arange(0,min(.06,args.duration),.0005),np.arange(.06,args.duration,.002),args.duration,brake,extension_time,args.fold_time,args.reach_time]);grid=grid[(grid>=0)&(grid<=args.duration)]
    table=[]
    for t in grid:
        q=q0.copy();q[ids]=joints(float(t));table.append(b.complete(q)[7:])
    Q=CubicSpline(grid,np.array(table),axis=0)
    def pose_rot(t,quat):
        q=q0.copy();q[:3]=0;q[3:7]=quat/np.linalg.norm(quat);q[7:]=Q(t);d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);return q
    def omega(t,quat):
        pose_rot(t,quat);A=np.zeros((3,m.nv));mujoco.mj_angmomMat(m,d,A,1);v=np.r_[np.zeros(6),Q(t,1)];return np.linalg.solve(A[:,3:6],H0-A@v)
    def rhs(t,quat):
        q=quat/np.linalg.norm(quat);w=omega(t,q);return .5*np.r_[-q[1:]@w,q[0]*w+np.cross(q[1:],w)]
    sol=solve_ivp(rhs,[0,args.duration],q0[3:7],method='DOP853',rtol=1e-9,atol=1e-10,max_step=.004,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    def direct_pose(q,v=None,a=None):
        d.qpos[:]=q;d.qvel[:]=0 if v is None else v;mujoco.mj_forward(m,d);d.qacc[:]=0 if a is None else a
    proxy=SimpleNamespace(m=m,d=d,b=SimpleNamespace(pose=direct_pose))
    def state(t):
        q=pose_rot(t,sol.sol(t));com=C0+V0*t+np.array([0.,0.,-.5*G*t*t]);q[:3]=com-d.subtree_com[1];direct_pose(q)
        A=np.zeros((3,m.nv));mujoco.mj_angmomMat(m,d,A,1);v=np.r_[np.zeros(6),Q(t,1)];v[3:6]=np.linalg.solve(A[:,3:6],H0-A@v);J=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,J,1);v[:3]=V0+[0,0,-G*t]-J@v;direct_pose(q,v)
        M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);a=np.r_[np.zeros(6),Q(t,2)];a[:6]=-np.linalg.solve(M[:6,:6],d.qfrc_bias[:6]+M[:6,6:]@a[6:]);d.qacc[:]=a;mujoco.mj_subtreeVel(m,d)
        return q,v,a
    keys=['times','qpos','qvel','qacc','body_xpos','body_xmat','com','velocity','angular_momentum','force_world_N','contact_forces_N','contact_positions_m','mode'];arr={key:[] for key in keys};max_NE=0.;derivative=0.;ROM=0.;closure=0.;Herror=0.
    work=[(float(z['times'][i]),int(i)) for i in range(0,k0,4)]+[(to+float(t),None) for t in np.r_[np.arange(0,args.duration,.004),args.duration]]
    for k,(time,index) in enumerate(work):
        air=index is None;t=time-to
        if air:
            q,v,a=state(t);forces=np.zeros((len(points),3))
            if k%5==0:
                ne=sample(proxy,q,v,a);max_NE=max(max_NE,float(abs(ne['body_wrenches'].sum(0)).max()))
                if .0002<t<args.duration-.0002:
                    vm=state(t-.00001)[1];vp=state(t+.00001)[1];derivative=max(derivative,float(abs((vp-vm)/.00002-a)[:6].max()))
                closed=b.complete(q);closure=max(closure,float(abs(closed-q).max()));direct_pose(q,v,a)
            Herror=max(Herror,float(abs(d.subtree_angmom[1]-H0).max()))
        else:
            q=z['qpos'][index];u=z['u'][index];aind=z['udot'][index];B,c=r.set_state(q,u);v=B@u;a=B@aind+c;direct_pose(q,v,a);forces=z['contact_forces_N'][index]
        mujoco.mj_subtreeVel(m,d);P=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);ROM=max(ROM,float(np.maximum(lo-q[ids],q[ids]-hi).max()))
        values=dict(times=time,qpos=q,qvel=v,qacc=a,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),force_world_N=forces.sum(0),contact_forces_N=forces,contact_positions_m=P,mode=2 if air else 0)
        for key,value in values.items():arr[key].append(value)
    data={key:np.array(value) for key,value in arr.items()};capsules=[]
    for q in data['qpos']:
        direct_pose(q);caps,rad=segments(m,d);capsules.append(caps)
    data['collision_segments_m']=np.array(capsules);data['collision_radii_m']=rad
    np.savez_compressed(out/'motion.npz',**data);b.pose(b.q0);np.savez_compressed(out/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
    air=data['mode']==2;P=data['contact_positions_m'][air];ts=data['times'][air];minimum=P[:,:,2].min(1);top_candidates=[]
    for front in np.arange(-.20,-.701,-.01):
        inside=(P[:,:,1]<=front)&(P[:,:,1]>=front-.80)&(abs(P[:,:,0]+.025)<=.50)
        crossing=np.flatnonzero((minimum[:-1]>args.height)&(minimum[1:]<=args.height))
        for i in crossing:
            j=i+1
            if not inside[j].all():continue
            hit=bool(np.any(inside[:j]&(P[:j,:,2]<args.height-.003)))
            if not hit:
                end_index=int(np.flatnonzero(air)[j]);clearance=min(minimum_clearance(caps,rad,front,args.height) for caps in data['collision_segments_m'][:end_index+1])
                if clearance>=.003:top_candidates.append(dict(front_y_m=float(front),contact_s=float(ts[j]),lowest_sole_z_m=float(minimum[j]),foot_height_spread_m=float(np.ptp(P[j,:,2])),body_capsule_clearance_m=float(clearance)))
                break
    selected=top_candidates[0] if top_candidates else None
    report=dict(status='athletic_high_box_flight_candidate' if selected else 'high_box_clearance_not_yet_passed',subject_height_m=1.78,subject_mass_kg=87,box_height_m=args.height,source_ground=str(source),takeoff_s=to,takeoff_COM_m=C0.tolist(),takeoff_velocity_m_s=V0.tolist(),takeoff_H_kg_m2_s=H0.tolist(),maximum_lowest_sole_height_m=float(minimum.max()),first_candidate=selected,all_candidate_front_edges=top_candidates,flight_root_NE_residual_max_SI=max_NE,root_acceleration_derivative_error_max=derivative,coordinate_closure_error_max=closure,angular_momentum_drift_max=Herror,ROM_excess_rad=max(0,ROM),root_actuator=False,force_clipping=False,muscle_force_analysis_allowed=False,landing_simulated=False,full_mesh_box_collision_checked=False,high_flexion_profile=profile,parameters=vars(args))
    meta=dict(status=report['status'],movement='box_jump_reference_replay',scale=b.report['global_normalization_scale'],body_names=[m.body(i).name for i in range(m.nbody)],phase_times_s=dict(bottom=.85,start_ascent=.85,end_ascent=to,end=float(data['times'][-1])),force_analysis_allowed=False,physical_task=True,landing_simulated=False,display_platform_only=True,box_geometry=dict(top_height_m=args.height,y_bounds_m=[(selected['front_y_m'] if selected else -.35)-.80,selected['front_y_m'] if selected else -.35]))
    (out/'summary.json').write_text(json.dumps(meta,indent=2));(out/'flight_audit.json').write_text(json.dumps(report,indent=2));print('ATHLETIC_FLIGHT',json.dumps({key:value for key,value in report.items() if key not in ['all_candidate_front_edges','high_flexion_profile']}),flush=True)


if __name__=='__main__':main()
