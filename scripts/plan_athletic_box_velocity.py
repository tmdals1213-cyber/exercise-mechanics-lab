"""Resolved-rate whole-body high-box motion under centroidal constraints.

Ballistic COM velocity and conserved angular momentum are hard equalities.
Feet, torso and quiet arm coordination are motion-design objectives. They do
not introduce root actuation. A failed path is retained as a rejected trial.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco,quadprog
from scipy.spatial.transform import Rotation
from scipy.linalg import qr
from centroidal_jump import Polynomial
from retarget_measured_jump import MeasuredJumpRig
from high_flex_profile import apply
from build_contact_jump import contact_geometry
from box_geometry_audit import segments,minimum_clearance,segment_box_distance
from mechanics_engine_adapter import sample
from continue_jump_landing import box_distance

ROOT=Path(__file__).resolve().parents[1];G=9.80665


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='results/physical_jump_task/yun_high_push_v7');p.add_argument('--label',default='yun_160_resolved_v1');p.add_argument('--hz',type=int,default=500);p.add_argument('--front',type=float,default=-.5);p.add_argument('--contact-time',type=float,default=.56);p.add_argument('--clear-time',type=float,default=.35);p.add_argument('--clearance',type=float,default=.05);p.add_argument('--advance',type=float,default=.08);p.add_argument('--settle',type=float,default=0.);p.add_argument('--quiet-arms',action='store_true');p.add_argument('--body-barriers',action='store_true');p.add_argument('--release-time',type=float);p.add_argument('--reach-hip',type=float);p.add_argument('--reach-knee',type=float);p.add_argument('--foot-weight',type=float,default=10000.);p.add_argument('--foot-rotation-weight',type=float,default=4000.);p.add_argument('--quiet-toes',action='store_true');p.add_argument('--touchdown-speed',type=float,default=0.);p.add_argument('--prescribed-legs',action='store_true');p.add_argument('--torso-lean',type=float,default=.65);p.add_argument('--leg-release-time',type=float);p.add_argument('--contact-slip-limit',type=float);p.add_argument('--quiet-leg-alignment',action='store_true');p.add_argument('--reach-hip-rate',type=float,default=0.);p.add_argument('--reach-knee-rate',type=float,default=0.);p.add_argument('--landing-coordination',action='store_true');args=p.parse_args()
    out=ROOT/'results/athletic_box_jump'/args.label;out.mkdir(parents=True,exist_ok=True);source=ROOT/args.source;z=np.load(source/'dynamic_ground_integrated.npz');r=MeasuredJumpRig(87,1.78);profile=apply(r);b,m,d=r.b,r.m,r.d
    release_index=len(z['times'])-1 if args.release_time is None else int(abs(z['times']-args.release_time).argmin())
    q0=z['qpos'][release_index].copy();u0=z['u'][release_index].copy();to=float(z['times'][release_index]);B,_=r.set_state(q0,u0);mujoco.mj_subtreeVel(m,d);C0=d.subtree_com[1].copy();V0=d.subtree_linvel[1].copy();H0=d.subtree_angmom[1].copy();r.contact_stand_override=z['qpos'][0];geometry=contact_geometry(r);points=geometry['points'];r.set_state(q0,u0)
    t_land=args.contact_time;height=1.6;Cland=C0+V0*t_land+[0,0,-.5*G*t_land*t_land];foot_targets=[]
    # Feet retain their material geometry. A frame at the heel is raised before
    # passing the box face, then advanced onto the top surface.
    qsave=q0.copy();b.pose(z['qpos'][0]);quiet_R={s:d.xmat[m.body('calcn_'+s).id].reshape(3,3).copy() for s in ['r','l']};quiet_torso=d.xmat[m.body('torso').id].reshape(3,3).copy();r.set_state(q0,u0)
    for side,sign in [('r',-1),('l',1)]:
        bid=m.body('calcn_'+side).id;local=m.site_pos[m.site('subject_sole_reference_'+side).id].copy();J,P=b.jacobian(bid,local);Jr=np.zeros((3,m.nv));jp=Jr.copy();mujoco.mj_jacBody(m,d,jp,Jr,bid);R0=d.xmat[bid].reshape(3,3).copy();w0=R0.T@(Jr@d.qvel)
        target=np.array([C0[0]+sign*.13,Cland[1]-args.advance+.137,height]);high=target.copy();high[1]=args.front+.265;high[2]=height+args.clearance
        up=Polynomial(P,J@d.qvel,0,high,0,0,args.clear_time);reach=Polynomial(high,0,0,target,np.array([0.,0.,-args.touchdown_speed]),0,t_land-args.clear_time)
        phi=Rotation.from_matrix(R0.T@quiet_R[side]).as_rotvec();turn=Polynomial(np.zeros(3),w0,0,phi,0,0,args.clear_time)
        foot_targets.append((bid,local,up,reach,R0,turn))
    torso=m.body('torso').id;Rtorso0=d.xmat[torso].reshape(3,3).copy();jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,torso);target_torso=Rotation.from_rotvec([args.torso_lean,0,0]).as_matrix()@quiet_torso
    torso_phi=Rotation.from_matrix(Rtorso0.T@target_torso).as_rotvec();torso_path=Polynomial(np.zeros(3),Rtorso0.T@(jr@d.qvel),0,torso_phi,0,0,.30)
    references=[];reference_durations=[];frozen={}
    for i,name in enumerate(b.ind_names):
        goal=q0[r.ids[i]]
        if name.startswith('shoulder_elv'):goal=1.05
        elif name.startswith('elbow_flexion'):goal=.45 if args.quiet_arms else 1.45
        elif name.startswith('hip_flexion'):goal=2.3
        elif name.startswith('knee_angle'):goal=2.50
        elif name.startswith('mtp_angle'):goal=0.
        ref_duration=.23 if name.startswith('knee_angle') else (.35 if args.quiet_arms and name.startswith(('shoulder_elv','elbow_flexion')) else t_land)
        references.append(Polynomial(q0[r.ids[i]],u0[6+i],0,goal,0,0,ref_duration))
        reference_durations.append(ref_duration)
        if name.startswith(('elv_angle','shoulder_rot','pro_sup','flexion_','deviation_')) or name in ['lat_bending','axial_rotation']:frozen[i]=float(q0[r.ids[i]])
    lo=m.jnt_range[b.independent,0].copy();hi=m.jnt_range[b.independent,1].copy()
    reach_paths={}
    for i,name in enumerate(b.ind_names):
        end=args.reach_hip if name.startswith('hip_flexion') else args.reach_knee if name.startswith('knee_angle') else None
        if end is not None:
            folded=2.30 if name.startswith('hip_flexion') else 2.48
            endrate=args.reach_hip_rate if name.startswith('hip_flexion') else args.reach_knee_rate
            reach_paths[i]=(Polynomial(q0[r.ids[i]],u0[6+i],0,folded,0,0,.28),Polynomial(folded,0,0,end,endrate,0,t_land-.28))
    for i,name in enumerate(b.ind_names):
        if name=='flex_extension':lo[i]=max(lo[i],-.6);hi[i]=min(hi[i],.6)
        if name.startswith('shoulder_elv'):hi[i]=min(hi[i],1.84)
    toe_paths={}
    if args.quiet_toes or args.quiet_leg_alignment:
        for i,name in enumerate(b.ind_names):
            if (args.quiet_toes and name.startswith('mtp_angle')) or (args.quiet_leg_alignment and name.startswith(('hip_rotation_','subtalar_angle_'))):
                angle=float(q0[r.ids[i]]);rate=float(u0[6+i]);distance=(hi[i]-angle if rate>0 else angle-lo[i])-.002;T=min(.03,1.5*max(distance,.001)/max(abs(rate),.01));stop=angle+.5*T*rate;toe_paths[i]=(Polynomial(angle,rate,0,stop,0,0,T),Polynomial(stop,0,0,float(z['qpos'][0,r.ids[i]]) if not name.startswith('mtp_angle') else 0.,0,0,.20-T),T)
    prescribed_legs={}
    if args.prescribed_legs:
        for i,name in enumerate(b.ind_names):
            end=args.reach_hip if name.startswith('hip_flexion') else args.reach_knee if name.startswith('knee_angle') else None
            if end is None:continue
            endrate=args.reach_hip_rate if name.startswith('hip_flexion') else args.reach_knee_rate
            angle=float(q0[r.ids[i]]);rate=float(u0[6+i]);distance=(hi[i]-angle if rate>0 else angle-lo[i])-.002;T=min(.035,1.5*max(distance,.001)/max(abs(rate),.01));stop=angle+.5*T*rate;folded=2.30 if name.startswith('hip_flexion') else 2.48;prescribed_legs[i]=[(0.,T,Polynomial(angle,rate,0,stop,0,0,T)),(T,.28,Polynomial(stop,0,0,folded,0,0,.28-T)),(.28,t_land,Polynomial(folded,0,0,end,endrate,0,t_land-.28))]
    previous=u0.copy();maximum_task_error=0.
    def desired_rotation(R0,poly,t,T):
        tt=min(max(t,0.),T);v=poly.value(tt);rate=poly.value(tt,1) if t<T else np.zeros(3);Rt=R0@Rotation.from_rotvec(v).as_matrix()
        # The spline rotation vector is nearly sagittal; use the exact right
        # Jacobian for its angular-rate map.
        from build_forward_jump_posthoc import right_jacobian
        return Rt,Rt@right_jacobian(v)@rate
    def velocity(q,t,preferred):
        B,_=r.set_state(q,np.zeros(r.n));Jc=np.zeros((3,m.nv));AH=Jc.copy();mujoco.mj_jacSubtreeCom(m,d,Jc,1);mujoco.mj_angmomMat(m,d,AH,1)
        if t<1e-12:return u0.copy(),B,0.
        C=C0+V0*t+[0,0,-.5*G*t*t];V=V0+[0,0,-G*t]
        E=[Jc@B,AH@B];rhs=[V+20*(C-d.subtree_com[1]),H0]
        for i,value in frozen.items():
            row=np.zeros((1,r.n));row[0,6+i]=1;E.append(row);rhs.append(np.array([20*(value-q[r.ids[i]])]))
        ref=np.array([float(poly.value(min(t,T))[0]) for T,poly in zip(reference_durations,references)]);vr=np.array([float(poly.value(t,1)[0]) if t<T else 0. for T,poly in zip(reference_durations,references)]);goal=np.r_[u0[:6]*np.exp(-t/.03),vr+12*(ref-q[r.ids])]
        for i,(fold,reach) in reach_paths.items():
            poly=fold if t<.28 else reach;tt=t if t<.28 else min(t-.28,t_land-.28);endrate=args.reach_hip_rate if b.ind_names[i].startswith('hip_flexion') else args.reach_knee_rate;goal[6+i]=float(poly.value(tt,1)[0] if t<t_land else endrate)+20*(float(poly.value(tt)[0])+max(0,t-t_land)*endrate-q[r.ids[i]])
        soft_arms=[]
        if args.quiet_arms:
            for i,name in enumerate(b.ind_names):
                if name.startswith(('shoulder_elv','elbow_flexion')):
                    row=np.zeros((1,r.n));row[0,6+i]=1.
                    if args.landing_coordination and t>=.30:
                        from centroidal_jump import s5
                        blend=float(s5(np.clip((t-.30)/.12,0,1)));soft_arms.append((row,np.array([goal[6+i]]),np.exp((1-blend)*np.log(1e8)+blend*np.log(150.))))
                    else:E.append(row);rhs.append(np.array([goal[6+i]]))
        if args.quiet_toes or args.quiet_leg_alignment:
            for i,(brake,relax,T) in toe_paths.items():
                poly=brake if t<T else relax;tt=t if t<T else min(t-T,.20-T);rate=float(poly.value(tt,1)[0]) if t<.20 else 0.;angle=float(poly.value(tt)[0]);row=np.zeros((1,r.n));row[0,6+i]=1.;E.append(row);rhs.append(np.array([rate+20*(angle-q[r.ids[i]])]))
        soft_legs=[]
        for i,pieces in prescribed_legs.items():
            start,end,poly=next((piece for piece in pieces if t<piece[1]),pieces[-1]);tt=np.clip(t-start,0,end-start);endrate=args.reach_hip_rate if b.ind_names[i].startswith('hip_flexion') else args.reach_knee_rate;rate=float(poly.value(tt,1)[0]) if t<t_land else endrate;angle=float(poly.value(tt)[0])+max(0,t-t_land)*endrate;row=np.zeros((1,r.n));row[0,6+i]=1.;target=np.array([rate+20*(angle-q[r.ids[i]])])
            if args.leg_release_time is not None and t>=args.leg_release_time:
                from centroidal_jump import s5
                blend=float(s5(np.clip((t-args.leg_release_time)/.1,0,1)));weight=np.exp((1-blend)*np.log(1e8)+blend*np.log(3000.));soft_legs.append((row,target,weight))
            else:E.append(row);rhs.append(target)
        weights=np.r_[np.full(6,.001),np.full(r.n-6,.4)];Q=np.diag(weights)+np.eye(r.n)*.01;linear=weights*goal+.01*preferred;error=0.
        def task(A,target,w):
            nonlocal Q,linear
            Q+=w*A.T@A;linear+=w*A.T@target
        for row,target,weight in soft_legs+soft_arms:task(row,target,weight)
        from centroidal_jump import s5
        landing_blend=float(s5(np.clip((t-.32)/.14,0,1))) if args.landing_coordination else 0.
        for bid,local,up,reach,R0,turn in foot_targets:
            J,P=b.jacobian(bid,local);poly=up if t<args.clear_time else reach;tt=t if t<args.clear_time else min(t-args.clear_time,t_land-args.clear_time);target=poly.value(tt);rate=poly.value(tt,1) if t<t_land else np.zeros(3)
            if t>=t_land and args.touchdown_speed:
                target[2]-=args.touchdown_speed*(t-t_land);rate[2]=-args.touchdown_speed
            task(J@B,rate+25*(target-P),args.foot_weight*(1+999*landing_blend));error=max(error,float(np.linalg.norm(target-P)))
            jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);Rt,wt=desired_rotation(R0,turn,t,args.clear_time);err=Rotation.from_matrix(Rt@d.xmat[bid].reshape(3,3).T).as_rotvec();task(jr@B,wt+15*err,args.foot_rotation_weight*(1+99*landing_blend))
        jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,torso);Rt,wt=desired_rotation(Rtorso0,torso_path,t,.30);err=Rotation.from_matrix(Rt@d.xmat[torso].reshape(3,3).T).as_rotvec();task(jr@B,wt+12*err,10000.)
        if args.landing_coordination and landing_blend:
            task(jr@B,np.zeros(3),1e6*landing_blend)
            for side in ['r','l']:
                bid=m.body('lunate_'+side).id;jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);task(jp@B,V,2000*landing_blend)
        for i,name in enumerate(b.ind_names):
            w=150. if name.startswith(('shoulder_elv','elbow_flexion','knee_angle')) else (600. if name.startswith(('hip_adduction','hip_rotation','subtalar_angle','mtp_angle')) else 0.)
            if i in reach_paths:w=3000.
            if w:
                row=np.zeros((1,r.n));row[0,6+i]=1;task(row,np.array([goal[6+i]]),w)
        E=np.vstack(E);rhs=np.concatenate(rhs);scale=np.maximum(np.linalg.norm(E,axis=1),1e-5);E/=scale[:,None];rhs/=scale;_,tri,piv=qr(E.T,pivoting=True,mode='economic');rank=int((abs(np.diag(tri))>1e-8).sum());keep=piv[:rank]
        limits=[];bounds=[]
        for bid,local,*_ in points:
            J,P=b.jacobian(bid,local);gap,normal=box_distance(P,args.front,height,width=1.,depth=.8)
            if gap<.30:limits.append(normal@J@B);bounds.append(-30*gap)
            if P[2]<.03:limits.append((J@B)[2]);bounds.append(-30*P[2])
        head=m.body('head').id;J,P=b.jacobian(head,m.body_ipos[head]);gap,normal=box_distance(P,args.front,height,width=1.,depth=.8)
        if gap<.35:limits.append(normal@J@B);bounds.append(-25*(gap-.10))
        if args.body_barriers:
            anchors={}
            for side in ['r','l']:
                for name,stem in [('hip','hip_flexion'),('knee','knee_angle'),('ankle','ankle_angle'),('shoulder','shoulder_elv'),('elbow','elbow_flexion'),('wrist','flexion')]:
                    jid=m.joint(stem+'_'+side).id;bid=m.jnt_bodyid[jid];point=d.xanchor[jid].copy();jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,point,bid);anchors[name+'_'+side]=(point,jp)
            pairs=[]
            for side in ['r','l']:
                for a,c,radius in [('hip','knee',.07),('knee','ankle',.05),('shoulder','elbow',.045),('elbow','wrist',.035)]:pairs.append((anchors[a+'_'+side],anchors[c+'_'+side],radius))
            hips=tuple((a+c)/2 for a,c in zip(anchors['hip_r'],anchors['hip_l']));shoulders=tuple((a+c)/2 for a,c in zip(anchors['shoulder_r'],anchors['shoulder_l']));pairs.append((hips,shoulders,.12))
            lower=np.array([-.525,args.front-.8,0.]);upper=np.array([.475,args.front,height])
            for (p0,j0),(p1,j1),radius in pairs:
                distance,s=segment_box_distance(np.array([p0,p1]),lower,upper,return_t=True)
                if distance>radius+.20:continue
                point=(1-s)*p0+s*p1;gap,normal=box_distance(point,args.front,height,width=1.,depth=.8);jp=(1-s)*j0+s*j1;limits.append(normal@jp@B);bounds.append(-25*(gap-radius-.003))
        for i in range(r.n-6):
            row=np.zeros(r.n);row[6+i]=1;limits.extend([row,-row]);bounds.extend([100*(lo[i]+1e-7-q[r.ids[i]]),100*(q[r.ids[i]]-hi[i]+1e-7)])
        u=quadprog.solve_qp((Q+Q.T)/2,linear,np.vstack([E[keep],limits]).T,np.r_[rhs[keep],bounds],meq=rank)[0]
        if abs(E@u-rhs).max()>1e-6:raise RuntimeError('Centroidal velocity equality failed')
        return u,B,error
    arrays={k:[] for k in ['times','qpos','qvel','u','body_xpos','body_xmat','com','velocity','angular_momentum','contact_positions_m','collision_segments_m']};q=q0.copy();failure=None
    for k in range(round((t_land+args.settle)*args.hz)+1):
        t=k/args.hz
        try:u,B,error=velocity(q,t,previous)
        except (ValueError,RuntimeError) as e:failure=str(e);break
        r.set_state(q,u);mujoco.mj_subtreeVel(m,d);P=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);caps,radii=segments(m,d)
        vals=dict(times=to+t,qpos=q.copy(),qvel=B@u,u=u,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),contact_positions_m=P,collision_segments_m=caps)
        for key,value in vals.items():arrays[key].append(value)
        maximum_task_error=max(maximum_task_error,error)
        if k%100==0:print('RESOLVED_BOX',round(t,3),'foot_error',round(error,4),'COM',np.round(d.subtree_com[1],3),flush=True)
        on_top=(P[:,1]<args.front-.003)&(P[:,1]>args.front-.8)&(abs(P[:,0]+.025)<.5)
        slip=max(np.linalg.norm((b.jacobian(bid,local)[0]@d.qvel)[:2]) for bid,local,*_ in points)
        if (args.contact_slip_limit is None or slip<args.contact_slip_limit) and t>.20 and on_top.all() and P[:6,2].min()<=height+.003 and P[6:,2].min()<=height+.003 and P[:,2].min()>=height-.003 and P[:,2].max()<height+.010 and minimum_clearance(caps,radii,args.front,height)>=0.:
            print('BOX_CONTACT_READY',round(t,4),flush=True);break
        if k==round((t_land+args.settle)*args.hz):break
        dt=1/args.hz;mid=q.copy();mujoco.mj_integratePos(m,mid,B@u,dt/2);mid=b.complete(mid)
        try:um,Bm,_=velocity(mid,t+dt/2,u)
        except (ValueError,RuntimeError) as e:failure=str(e);break
        nextq=q.copy();mujoco.mj_integratePos(m,nextq,Bm@um,dt)
        try:q=b.complete(nextq)
        except RuntimeError as e:failure=str(e);break
        previous=um
    data={k:np.array(v) for k,v in arrays.items()};data['collision_radii_m']=radii;np.savez_compressed(out/'flight_motion.npz',**data)
    actualt=data['times']-to;ballistic=C0+actualt[:,None]*V0+[0,0,0]+.5*actualt[:,None]**2*np.array([0,0,-G]);position_error=float(np.linalg.norm(data['com']-ballistic,axis=1).max());Herror=float(np.linalg.norm(data['angular_momentum']-H0,axis=1).max());P=data['contact_positions_m'];bodyclear=min(minimum_clearance(caps,radii,args.front,height) for caps in data['collision_segments_m'])
    geometry_ok=bool(failure is None and P[-1,:,2].min()>height-.003 and P[-1,:,2].max()<height+.01 and P[-1,:,1].max()<args.front-.003 and bodyclear>=0.)
    report=dict(status='precontact_geometry_candidate' if geometry_ok else 'rejected_resolved_high_box',failure=failure,precontact_geometry_pass=geometry_ok,box_height_m=height,box_front_y_m=args.front,source_ground=args.source,takeoff_s=to,takeoff_COM_m=C0.tolist(),takeoff_velocity_m_s=V0.tolist(),takeoff_H_kg_m2_s=H0.tolist(),duration_s=float(actualt[-1]),maximum_foot_tracking_error_m=maximum_task_error,final_sole_height_range_m=[float(P[-1,:,2].min()),float(P[-1,:,2].max())],maximum_lowest_sole_height_m=float(P[:,:,2].min(1).max()),final_foot_y_bounds_m=[float(P[-1,:,1].min()),float(P[-1,:,1].max())],final_COM_m=data['com'][-1].tolist(),ballistic_COM_error_m=position_error,angular_momentum_error_kg_m2_s=Herror,body_capsule_clearance_m=float(bodyclear),root_actuator=False,force_clipping=False,muscle_force_analysis_allowed=False,landing_simulated=False,parameters=vars(args))
    (out/'resolved_audit.json').write_text(json.dumps(report,indent=2));print('RESOLVED_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()










