"""Contact-acceleration continuation from a near-touch high-box state.

No pose/velocity reset, root actuator, prescribed reaction, or strength limit.
The 3 mm contact tolerance and Baumgarte stabilization are numerical contact
assumptions. Failed support solves remain rejected and retain partial motion.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco,quadprog
from scipy.spatial.transform import Rotation
from scipy.linalg import qr
from retarget_measured_jump import MeasuredJumpRig
from high_flex_profile import apply
from build_contact_jump import contact_geometry
from box_geometry_audit import segments,minimum_clearance
from centroidal_jump import Polynomial,s5
ROOT=Path(__file__).resolve().parents[1];G=9.80665


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder');ap.add_argument('--label',default='supported_landing');ap.add_argument('--hz',type=int,default=500);ap.add_argument('--duration',type=float,default=1.2);ap.add_argument('--quiet-upper-body',action='store_true');ap.add_argument('--quiet-upper-mode',choices=['fixed','guided','wrists'],default='guided');ap.add_argument('--coordinated-trunk',action='store_true');ap.add_argument('--recovery-duration',type=float,default=.9);ap.add_argument('--free-com-height',action='store_true');ap.add_argument('--landing-hinge',type=float,default=0.);ap.add_argument('--absorb-duration',type=float,default=.15);ap.add_argument('--absorb-drop',type=float,default=.08);ap.add_argument('--soft-horizontal',action='store_true');ap.add_argument('--upper-speed-limit',type=float);ap.add_argument('--posture-priority',type=float,default=1.);ap.add_argument('--trunk-weight',type=float,default=100.);ap.add_argument('--joint-tracking-duration',type=float,default=.9);ap.add_argument('--soft-upper-weight',type=float,default=8.);ap.add_argument('--contact-impact',action='store_true');ap.add_argument('--rocker-contact',action='store_true');ap.add_argument('--recovery-delay',type=float,default=0.);ap.add_argument('--hand-task-weight',type=float,default=0.);ap.add_argument('--hand-task-duration',type=float,default=1.2);args=ap.parse_args()
    folder=ROOT/args.folder;out=folder/args.label;out.mkdir(exist_ok=True);meta=json.loads((folder/'resolved_audit.json').read_text());air=np.load(folder/'flight_motion.npz');ground=np.load(ROOT/meta['source_ground']/'dynamic_ground_integrated.npz');front=meta['box_front_y_m'];height=1.6
    P=air['contact_positions_m'];inside=(P[:,:,1]<front)&(P[:,:,1]>front-.8)&(abs(P[:,:,0]+.025)<.5);ready=inside.all(1)&(P[:,:6,2].min(1)<=height+.003)&(P[:,6:,2].min(1)<=height+.003)&(P[:,:,2].min(1)>=height-.003)
    candidates=np.flatnonzero(ready&(air['times']>meta['takeoff_s']+.2))
    if not len(candidates):
        report=dict(status='rejected_no_actual_box_contact',landing_simulated=False,minimum_inside_sole_gap_m=float((P[:,:,2].min(1)-height)[inside.all(1)].min(initial=np.inf)));(out/'landing_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    index=int(candidates[0]);q=air['qpos'][index].copy();u=air['u'][index].copy();t0=float(air['times'][index]);r=MeasuredJumpRig(87,1.78);apply(r);b,m,d=r.b,r.m,r.d;r.contact_stand_override=ground['qpos'][0];geometry=contact_geometry(r);points=geometry['points'];n=r.n;W=m.body_mass.sum()*G
    b.pose(ground['qpos'][0]);Rquiet=d.xmat[m.body('torso').id].reshape(3,3).copy();footR={s:d.xmat[m.body('calcn_'+s).id].reshape(3,3).copy() for s in ['r','l']};r.set_state(q,u);mujoco.mj_subtreeVel(m,d);C0=d.subtree_com[1].copy();V0=d.subtree_linvel[1].copy();joint0=q[r.ids].copy();stand=ground['qpos'][0,r.ids].copy();anchors={}
    for side in ['r','l']:
        bid=m.body('calcn_'+side).id;local=m.site_pos[m.site('subject_sole_reference_'+side).id].copy();_,anchor=b.jacobian(bid,local);anchor[2]=height;anchors[side]=(bid,local,anchor,footR[side])
    impact=None
    if args.contact_impact:
        B,_=r.set_state(q,u);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);Mr=B.T@M@B
        J=np.vstack([b.jacobian(bid,local)[0]@B for bid,local,*_ in points]);size=n+J.shape[0]
        E=np.zeros((n+J.shape[0],size));E[:n,:n]=Mr;E[:n,n:]=-J.T;E[n:,:n]=J;rhs=np.r_[Mr@u,np.zeros(J.shape[0])]
        scale=np.maximum(np.linalg.norm(E,axis=1),1e-6);E/=scale[:,None];rhs/=scale;_,tri,piv=qr(E.T,pivoting=True,mode='economic');rank=int((abs(np.diag(tri))>1e-9).sum());keep=piv[:rank];rows=[]
        for j in range(len(points)):
            for x,y,z in [(0,0,1)]+[(-np.cos(phi),-np.sin(phi),.8*np.cos(np.pi/16)) for phi in np.linspace(0,2*np.pi,16,endpoint=False)]:
                row=np.zeros(size);row[n+3*j:n+3*j+3]=[x,y,z];rows.append(row)
        Q=np.eye(size)*1e-9;Q[:n,:n]+=Mr;sol=quadprog.solve_qp(Q,np.zeros(size),np.vstack([E[keep],rows]).T,np.r_[rhs[keep],np.zeros(len(rows))],meq=rank)[0]
        before=u.copy();u=sol[:n];impulses=sol[n:].reshape(-1,3);res=float(abs(Mr@(u-before)-J.T@impulses.ravel()).max());impact=dict(type='perfectly_inelastic_sticking_impulse',friction=.8,pre_u=before.tolist(),post_u=u.tolist(),contact_impulses_Ns=impulses.tolist(),reduced_momentum_residual_max_SI=res,post_sole_velocity_max_m_s=float(abs(J@u).max()),kinetic_energy_loss_J=float(.5*(before@Mr@before-u@Mr@u)),pose_reset=False,arbitrary_velocity_reset=False)
        joint0=q[r.ids].copy();r.set_state(q,u);mujoco.mj_subtreeVel(m,d);C0=d.subtree_com[1].copy();V0=d.subtree_linvel[1].copy()
        print('CONTACT_IMPULSE',impact['kinetic_energy_loss_J'],impulses.sum(0),flush=True)
    torso_bid=m.body('torso').id;Rcontact=d.xmat[torso_bid].reshape(3,3).copy();jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,torso_bid);omega_contact=Rcontact.T@(jr@d.qvel);phi_hold=.075*omega_contact+np.array([args.landing_hinge,0.,0.]);phi_end=Rotation.from_matrix(Rcontact.T@Rotation.from_rotvec([.15,0,0]).as_matrix()@Rquiet).as_rotvec();torso_brake=Polynomial(np.zeros(3),omega_contact,0,phi_hold,0,0,.15);torso_rise=Polynomial(phi_hold,0,0,phi_end,0,0,.90)
    def torso_reference(t):
        from build_forward_jump_posthoc import right_jacobian
        def orientation_rate(time):
            poly=torso_brake if time<.15 else torso_rise;s=max(time,0.) if time<.15 else np.clip(time-.15-args.recovery_delay,0,.90);phi=poly.value(s);phidot=poly.value(s,1) if time<1.05+args.recovery_delay else np.zeros(3);Rt=Rcontact@Rotation.from_rotvec(phi).as_matrix();return Rt,Rt@right_jacobian(phi)@phidot
        Rt,wr=orientation_rate(t);eps=1e-5;ar=(orientation_rate(t+eps)[1]-orientation_rate(t-eps)[1])/(2*eps);return Rt,wr,ar
    absorb=Polynomial(C0[2],V0[2],-G,max(height+.42,C0[2]-args.absorb_drop),0,0,args.absorb_duration);stand_z=height+float(np.mean(ground['qpos'][:1,2]))
    b.pose(ground['qpos'][0]);stand_z=height+float(d.subtree_com[1,2]);recover=Polynomial(float(absorb.value(args.absorb_duration)[0]),0,0,stand_z,0,0,args.recovery_duration)
    capture=C0[:2]+V0[:2]/np.sqrt(G/max(.35,C0[2]-height));capture[0]=np.clip(capture[0],P[index,:,0].min()+.04,P[index,:,0].max()-.04);capture[1]=np.clip(capture[1],P[index,:,1].min()+.04,P[index,:,1].max()-.04)
    lo=m.jnt_range[b.independent,0].copy();hi=m.jnt_range[b.independent,1].copy()
    for i,name in enumerate(b.ind_names):
        if name.startswith('shoulder_elv'):hi[i]=min(hi[i],1.84)
        if name=='flex_extension':lo[i]=max(lo[i],-.6);hi[i]=min(hi[i],.6)
    quiet_indices=[i for i,name in enumerate(b.ind_names) if name.startswith(('elv_angle_','shoulder_','elbow_','pro_sup_','deviation_','flexion_')) or name in ['lat_bending','axial_rotation']]
    quiet_paths={i:(Polynomial(joint0[i],u[6+i],0.,joint0[i],0.,0.,.15),Polynomial(joint0[i],0.,0.,stand[i],0.,0.,.90)) for i in quiet_indices}
    def upper_reference(i,t):
        hold,lower=quiet_paths[i];poly=hold if t<.15 else lower;s=t if t<.15 else np.clip(t-.15-args.recovery_delay,0,.90)
        return tuple(float(poly.value(s,j)[0]) if t<1.05+args.recovery_delay or j==0 else 0. for j in range(3))
    # Use world-space hand paths, not singular shoulder Euler-rate limits.
    hand_paths=[]
    if args.hand_task_weight:
        b.pose(ground['qpos'][0]);oldfeet=np.mean([d.site_xpos[m.site('subject_sole_reference_'+side).id] for side in ['r','l']],axis=0);newfeet=np.mean([anchor for _,_,anchor,_ in anchors.values()],axis=0);shift=newfeet-oldfeet;goals={}
        for side in ['r','l']:
            for stem in ['lunate','ulna']:
                bid=m.body(stem+'_'+side).id;goals[bid]=(d.xpos[bid].copy()+shift,d.xmat[bid].reshape(3,3).copy())
        r.set_state(q,u)
        for bid,(end,Rend) in goals.items():
            jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);R0=d.xmat[bid].reshape(3,3).copy();path=Polynomial(d.xpos[bid].copy(),jp@d.qvel,0,end,0,0,args.hand_task_duration);phi=Rotation.from_matrix(R0.T@Rend).as_rotvec();turn=Polynomial(np.zeros(3),R0.T@(jr@d.qvel),0,phi,0,0,args.hand_task_duration);hand_paths.append((bid,path,R0,turn,1. if m.body(bid).name.startswith('lunate') else .15))
    def solve(q,u,t,contact_mode='full'):
        B,curve=r.set_state(q,u);mujoco.mj_subtreeVel(m,d);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);Mr=B.T@M@B;bias=B.T@(M@curve+d.qfrc_bias-d.qfrc_passive);nv=n+3*len(points);E=[];rhs=[];root=np.zeros((6,nv));root[:,:n]=Mr[:6];Js=[];positions=[]
        for j,(bid,local,*_) in enumerate(points):
            J,p=b.jacobian(bid,local);Js.append(J@B);positions.append(p);root[:,n+3*j:n+3*j+3]=-W*(J@B)[:,:6].T
        E.append(root);rhs.append(-bias[:6]);limits=[];bounds=[]
        if args.quiet_upper_body:
            for i in quiet_indices:
                if args.quiet_upper_mode=='guided' and (b.ind_names[i].startswith(('shoulder_elv_','elbow_flexion_')) or b.ind_names[i] in ['lat_bending','axial_rotation']):continue
                if args.quiet_upper_mode=='wrists' and not b.ind_names[i].startswith(('pro_sup_','deviation_','flexion_')):continue
                angle,rate,acceleration=upper_reference(i,t);row=np.zeros((1,nv));row[0,6+i]=1.;E.append(row);rhs.append(np.array([acceleration+225*(angle-q[r.ids[i]])+30*(rate-u[6+i])]))
        if contact_mode=='full':
            for side,(bid,local,anchor,R0) in anchors.items():
                J,p=b.jacobian(bid,local);jr=np.zeros_like(J);jdot=np.zeros_like(J);jrdot=np.zeros_like(J);mujoco.mj_jac(m,d,J,jr,p,bid);mujoco.mj_jacDot(m,d,jdot,jrdot,p,bid);row=np.zeros((3,nv));row[:,:n]=J@B;E.append(row);rhs.append(np.array([1000.,1000.,4000.])*(anchor-p)-np.array([60.,60.,100.])*(J@d.qvel)-jdot@d.qvel-J@curve)
                err=Rotation.from_matrix(R0@d.xmat[bid].reshape(3,3).T).as_rotvec();row=np.zeros((3,nv));row[:,:n]=jr@B;E.append(row);rhs.append(1000*err-60*(jr@d.qvel)-jrdot@d.qvel-jr@curve)
                i=b.ind_names.index('mtp_angle_'+side);row=np.zeros((1,nv));row[0,6+i]=1.;E.append(row);rhs.append(np.array([1000*(stand[i]-q[r.ids[i]])-60*u[6+i]]))
        else:
            byside={'heel':[[0,1],[6,7]],'ball':[[2,3],[8,9]],'tip':[[4,5],[10,11]]}[contact_mode]
            active=set(sum(byside,[]))
            for j in range(len(points)):
                if j not in active:
                    row=np.zeros((3,nv));row[:,n+3*j:n+3*j+3]=np.eye(3);E.append(row);rhs.append(np.zeros(3))
                bid,local,*_=points[j];J,p=b.jacobian(bid,local);jd=np.zeros_like(J);jrd=jd.copy();mujoco.mj_jacDot(m,d,jd,jrd,p,bid)
                if j not in active:
                    row=np.zeros(nv);row[:n]=(J@B)[2];limits.append(row);bounds.append(float(4000*(height-.003-p[2])-100*(J@d.qvel)[2]-(jd@d.qvel+J@curve)[2]))
            for side,ids in zip(['r','l'],byside):
                bid=points[ids[0]][0];local=np.mean([points[j][1] for j in ids],axis=0);anchor=P[index,ids].mean(0).copy();anchor[2]=height;J,p=b.jacobian(bid,local);jr=np.zeros_like(J);jd=jr.copy();jrd=jr.copy();mujoco.mj_jac(m,d,J,jr,p,bid);mujoco.mj_jacDot(m,d,jd,jrd,p,bid);row=np.zeros((3,nv));row[:,:n]=J@B;E.append(row);rhs.append(np.array([1000.,1000.,4000.])*(anchor-p)-np.array([60.,60.,100.])*(J@d.qvel)-jd@d.qvel-J@curve)
                bpose=ground['qpos'][0];saved=d.qpos.copy();b.pose(bpose);R0=d.xmat[bid].reshape(3,3).copy();r.set_state(saved,u)
                _,_,Vt=np.linalg.svd(P[index,ids]-P[index,ids].mean(0),full_matrices=True);rotrows=Vt[1:];err=Rotation.from_matrix(R0@d.xmat[bid].reshape(3,3).T).as_rotvec();row=np.zeros((2,nv));row[:,:n]=rotrows@jr@B;E.append(row);rhs.append(rotrows@(1000*err-60*(jr@d.qvel)-jrd@d.qvel-jr@curve))
        blend=float(s5(np.clip((t-.10-args.recovery_delay)/args.joint_tracking_duration,0,1)));target=joint0+blend*(stand-joint0);goal=np.r_[np.zeros(6),100*(target-q[r.ids])-20*u[6:],np.zeros(3*len(points))];weights=np.r_[np.full(6,.01),np.full(n-6,args.posture_priority),np.full(3*len(points),.1)]
        if args.quiet_upper_body and args.quiet_upper_mode in ['guided','wrists']:
            for i in quiet_indices:
                if args.quiet_upper_mode=='guided' and not (b.ind_names[i].startswith(('shoulder_elv_','elbow_flexion_')) or b.ind_names[i] in ['lat_bending','axial_rotation']):continue
                if args.quiet_upper_mode=='wrists' and b.ind_names[i].startswith(('pro_sup_','deviation_','flexion_')):continue
                angle,rate,acceleration=upper_reference(i,t);goal[6+i]=acceleration+225*(angle-q[r.ids[i]])+30*(rate-u[6+i]);weights[6+i]=(100. if b.ind_names[i].startswith(('shoulder_elv_','elbow_flexion_')) else 30.) if args.quiet_upper_mode=='guided' else args.soft_upper_weight
        if args.quiet_upper_body:weights[6:n]*=np.sqrt(args.posture_priority)
        Q=np.diag(weights);lin=weights*goal
        def task(A,v,w):
            nonlocal Q,lin
            Q+=w*A.T@A;lin+=w*A.T@v
        if args.hand_task_weight:
            from build_forward_jump_posthoc import right_jacobian
            for bid,path,R0,turn,factor in hand_paths:
                jp=np.zeros((3,m.nv));jr=jp.copy();jd=jp.copy();jrd=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);mujoco.mj_jacDot(m,d,jd,jrd,d.xpos[bid],bid);tt=min(t,args.hand_task_duration);P=path.value(tt);V=path.value(tt,1) if t<args.hand_task_duration else np.zeros(3);A=path.value(tt,2) if t<args.hand_task_duration else np.zeros(3);row=np.zeros((3,nv));row[:,:n]=jp@B;task(row,A+100*(P-d.xpos[bid])+20*(V-jp@d.qvel)-jd@d.qvel-jp@curve,args.hand_task_weight*factor)
                phi=turn.value(tt);rate=turn.value(tt,1) if t<args.hand_task_duration else np.zeros(3);Rt=R0@Rotation.from_rotvec(phi).as_matrix();wr=Rt@right_jacobian(phi)@rate;err=Rotation.from_matrix(Rt@d.xmat[bid].reshape(3,3).T).as_rotvec();row=np.zeros((3,nv));row[:,:n]=jr@B;task(row,100*err+20*(wr-jr@d.qvel)-jrd@d.qvel-jr@curve,args.hand_task_weight*factor*.06)
        C=d.subtree_com[1].copy();V=d.subtree_linvel[1];poly=absorb if t<args.absorb_duration else recover;tt=t if t<args.absorb_duration else np.clip(t-args.absorb_duration-args.recovery_delay,0,args.recovery_duration);zr,vr,ar=[float(poly.value(tt,k)[0]) for k in range(3)];acc=np.r_[30*(capture-C[:2])-10*V[:2],ar+100*(zr-C[2])+20*(vr-V[2])]
        row=np.zeros((3,nv))
        for j in range(len(points)):row[:,n+3*j:n+3*j+3]=np.eye(3)
        if args.free_com_height:
            task(row[:2],acc[:2]/G,10000000.);task(row[2:],np.ones(1),10000.)
        else:task(row,acc/G+[0,0,1],10000000./np.sqrt(args.posture_priority))
        if t>=.10 and not args.soft_horizontal:E.append(row[:2]);rhs.append(acc[:2]/G)
        momentum=np.zeros((3,nv))
        for j,p in enumerate(positions):
            x,y,z=p-C;momentum[:,n+3*j:n+3*j+3]=W*np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        task(momentum,-8*d.subtree_angmom[1],.001 if t<.25 else .005)
        bid=m.body('torso').id;jp=np.zeros((3,m.nv));jr=jp.copy();jd=jp.copy();jrd=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid);mujoco.mj_jacDot(m,d,jd,jrd,d.xpos[bid],bid);Rt,wr,ar=torso_reference(t) if args.coordinated_trunk else (Rotation.from_rotvec([.15,0,0]).as_matrix()@Rquiet,np.zeros(3),np.zeros(3));err=Rotation.from_matrix(Rt@d.xmat[bid].reshape(3,3).T).as_rotvec();row=np.zeros((3,nv));row[:,:n]=jr@B;task(row,ar+100*err+20*(wr-jr@d.qvel)-jrd@d.qvel-jr@curve,args.trunk_weight*args.posture_priority)
        for j in range(len(points)):
            for x,y,z in [(0,0,1)]+[(-np.cos(phi),-np.sin(phi),.8*np.cos(np.pi/16)) for phi in np.linspace(0,2*np.pi,16,endpoint=False)]:
                row=np.zeros(nv);row[n+3*j:n+3*j+3]=[x,y,z];limits.append(row);bounds.append(0.)
        if args.upper_speed_limit is not None:
            for i in quiet_indices:
                limit=np.deg2rad(args.upper_speed_limit);row=np.zeros(nv);row[6+i]=1.;limits.extend([row,-row]);bounds.extend([-80*(limit+u[6+i]),-80*(limit-u[6+i])])
        for i in range(n-6):
            row=np.zeros(nv);row[6+i]=1;limits.extend([row,-row]);bounds.extend([-200*u[6+i]+10000*(lo[i]+1e-6-q[r.ids[i]]),200*u[6+i]-10000*(hi[i]-1e-6-q[r.ids[i]])])
        E=np.vstack(E);rhs=np.concatenate(rhs);sc=np.maximum(np.linalg.norm(E,axis=1),.02);E/=sc[:,None];rhs/=sc;_,T,piv=qr(E.T,pivoting=True,mode='economic');rank=int((abs(np.diag(T))>1e-8).sum());keep=piv[:rank];scale=max(float(np.diag(Q).max()),1.);solution=quadprog.solve_qp((Q+Q.T)/2/scale,lin/scale,np.vstack([E[keep],limits]).T,np.r_[rhs[keep],bounds],meq=rank)[0]
        if abs(E@solution-rhs).max()>1e-5:raise RuntimeError('Redundant contact or root equation inconsistency')
        return solution[:n],B,W*solution[n:].reshape(-1,3),np.array(positions),float(abs(E[:6]@solution-rhs[:6]).max())
    mode_history=[]
    def solve_contact(q,u,t):
        failures=[]
        for mode in (['full','heel','ball','tip'] if args.rocker_contact else ['full']):
            try:
                result=solve(q,u,t,mode);mode_history.append((float(t),mode));return result
            except (ValueError,RuntimeError) as error:failures.append((mode,str(error)))
        raise RuntimeError(str(failures))
    arrays={k:[] for k in ['times','qpos','u','body_xpos','body_xmat','com','velocity','angular_momentum','contact_positions_m','contact_forces_N']};failure=None;max_res=0.;bodyclear=np.inf;minheight=np.inf;mid_forces=[];mid_torques=[]
    for k in range(round(args.duration*args.hz)+1):
        t=k/args.hz
        try:
            acc,B,F,positions,res=solve_contact(q,u,t);max_res=max(max_res,res);caps,rad=segments(m,d);bodyclear=min(bodyclear,minimum_clearance(caps,rad,front,height));minheight=min(minheight,float(positions[:,2].min()));mujoco.mj_subtreeVel(m,d)
            vals=dict(times=t0+t,qpos=q.copy(),u=u.copy(),body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),contact_positions_m=positions,contact_forces_N=F)
            for key,value in vals.items():arrays[key].append(value)
            if k%100==0:print('BOX_SUPPORT',round(t,3),'C',np.round(d.subtree_com[1],3),'F',round(F[:,2].sum()),flush=True)
            if k==round(args.duration*args.hz):break
            dt=1/args.hz;mid=q.copy();mujoco.mj_integratePos(m,mid,B@u,dt/2);mid=b.complete(mid);um=u+dt*acc/2;am,Bm,Fm,Pm,_=solve_contact(mid,um,t+dt/2);mid_forces.append(Fm.sum(0));mid_torques.append(np.cross(Pm-d.subtree_com[1],Fm).sum(0));nq=q.copy();mujoco.mj_integratePos(m,nq,Bm@um,dt);q=b.complete(nq);u+=dt*am
        except (ValueError,RuntimeError) as error:failure=str(error);break
    data={key:np.array(value) for key,value in arrays.items()};count=max(0,len(data['times'])-1);data['midpoint_force_world_N']=np.array(mid_forces[:count]);data['midpoint_torque_about_COM_Nm']=np.array(mid_torques[:count]);np.savez_compressed(out/'supported_motion.npz',**data)
    finalspeed=float(np.linalg.norm(data['velocity'][-1])) if len(data['times']) else np.inf;success=bool(failure is None and bodyclear>=0 and minheight>=height-.005 and finalspeed<.1)
    report=dict(status='supported_box_landing_candidate' if success else 'rejected_supported_box_landing',failure=failure,completed_supported_landing=success,landing_simulated=bool(len(data['times'])),contact_time_s=t0,contact_index=index,contact_tolerance_m=.003,initial_foot_height_range_m=[float(P[index,:,2].min()),float(P[index,:,2].max())],duration_s=float(data['times'][-1]-t0) if len(data['times']) else 0,final_COM_speed_m_s=finalspeed,body_capsule_clearance_m=float(bodyclear),maximum_sole_penetration_m=max(0.,height-minheight),maximum_scaled_root_balance_residual=max_res,peak_landing_GRF_N=float(np.linalg.norm(data['contact_forces_N'].sum(1),axis=1).max()) if len(data['times']) else None,root_actuator=False,pose_reset=False,velocity_reset=False,muscle_force_analysis_allowed=False)
    if count:
        dt=np.diff(data['times'])[:,None];force_residual=87*np.diff(data['velocity'],axis=0)/dt-data['midpoint_force_world_N']-np.array([0.,0.,-87*G]);moment_residual=np.diff(data['angular_momentum'],axis=0)/dt-data['midpoint_torque_about_COM_Nm'];report['independent_discrete_momentum_force_equivalent_max_N']=float(abs(force_residual).max());report['independent_discrete_momentum_force_equivalent_RMS_N']=float(np.sqrt(np.mean(force_residual**2)));report['independent_discrete_angular_momentum_torque_equivalent_max_Nm']=float(abs(moment_residual).max());report['integration_hz']=args.hz
    report['contact_modes_used']=sorted(set(mode for _,mode in mode_history));report['impact']=impact;report['quiet_upper_body']=args.quiet_upper_body;report['parameters']=vars(args)
    report['coordinated_trunk']=args.coordinated_trunk;report['COM_height_is_motion_output']=args.free_com_height;report['landing_hinge_rad']=args.landing_hinge
    report['quiet_upper_mode']=args.quiet_upper_mode if args.quiet_upper_body else None
    if len(data['times']):
        report['upper_joint_peak_speed_deg_s']={name:float(np.rad2deg(abs(data['u'][:,6+i]).max())) for i,name in enumerate(b.ind_names) if i in quiet_indices}
        if args.quiet_upper_body:report['upper_reference_max_error_rad']=float(max(abs(float(data['qpos'][k,r.ids[i]])-upper_reference(i,float(t-t0))[0]) for k,t in enumerate(data['times']) for i in quiet_indices))
    (out/'landing_audit.json').write_text(json.dumps(report,indent=2));print('SUPPORTED_BOX_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()





