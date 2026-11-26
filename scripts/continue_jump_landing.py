"""Hybrid foot/box contact continuation with an unactuated floating root.

Rigid, inelastic contact impulses and Coulomb-cone friction are resolved in
reduced coordinates. Internal posture/balance torques are scenario controls,
not physiological recruitment. Impact impulse is not a resolved peak force.
Only feet contact the environment; body/box penetration therefore rejects a
task rather than being silently accepted as a valid landing.
"""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco,quadprog
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation,RotationSpline
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from centroidal_jump import s5,Polynomial
from scipy.linalg import qr
ROOT=Path(__file__).resolve().parents[1]


def box_distance(p,front,height,width=.8,depth=.75):
    center=np.array([-.025,front-depth/2,height/2]);half=np.array([width/2,depth/2,height/2]);delta=p-center;a=abs(delta)-half;outside=np.maximum(a,0);length=np.linalg.norm(outside)
    if length>1e-12:return float(length),np.sign(delta)*outside/length
    axis=int(np.argmax(a));normal=np.zeros(3);normal[axis]=1 if delta[axis]>=0 else -1
    return float(a[axis]),normal


class Landing:
    def __init__(self,r,points,front,height,dt):self.r=r;self.points=points;self.front=front;self.height=height;self.dt=dt;self.max_constraint=0.;self.max_dynamics=0.;self.max_penetration=0.;self.joint_stop_impulse=0.;self.max_normal_complementarity=0.

    def complementarity_refine(self,H,w,impulse,blocks,normal_rows):
        nr=np.array(normal_rows,dtype=int);tr=np.array([start+j for start,count in blocks for j in [0,1]],dtype=int);pn=impulse[nr].copy();pt=impulse[tr].copy();N=H[np.ix_(nr,nr)]+np.eye(len(nr))*1e-7;T=H[np.ix_(tr,tr)]+np.eye(len(tr))*1e-7;NT=H[np.ix_(nr,tr)]
        D=np.zeros((12*len(blocks),len(tr)))
        for j in range(len(blocks)):
            for k,phi in enumerate(np.linspace(0,2*np.pi,12,endpoint=False)):D[12*j+k,2*j:2*j+2]=[-np.cos(phi),-np.sin(phi)]
        normal_lookup={value:j for j,value in enumerate(nr)};contact_n=[normal_lookup[start+2] for start,count in blocks]
        for iteration in range(100):
            newn=quadprog.solve_qp(N,-w[nr]-NT@pt,np.eye(len(nr)),np.zeros(len(nr)))[0]
            newn=np.maximum(newn,0.)
            if len(tr):
                bound=-.8*np.cos(np.pi/12)*np.repeat(newn[contact_n],12)
                newt=quadprog.solve_qp(T,-w[tr]-NT.T@newn,D.T,bound)[0]
            else:newt=pt
            change=max(float(abs(newn-pn).max()),float(abs(newt-pt).max()) if len(pt) else 0.)
            pn=.5*(pn+newn);pt=.5*(pt+newt);vn=H[np.ix_(nr,nr)]@pn+NT@pt+w[nr]
            if max(float(abs(pn*vn).max()),float(np.maximum(-vn,0).max()),change)<1e-5:break
        impulse[nr]=pn;impulse[tr]=pt
        self.max_contact_iterations=max(getattr(self,'max_contact_iterations',0),iteration+1)
        if iteration==99:self.contact_iteration_limit_steps=getattr(self,'contact_iteration_limit_steps',0)+1
        return impulse

    def stance_torque(self,q,u,target,pitch,Mr,bias,B,curve,Jcom,P,operators,gaps,blocks,point_ids):
        r=self.r;b,m,d=r.b,r.m,r.d;W=m.body_mass.sum()*9.80665;n=r.n;nc=len(blocks);nv=n+3*nc;E=[];rhs=[]
        root=np.zeros((6,nv));root[:,:n]=Mr[:6]
        for j,(start,count) in enumerate(blocks):root[:,n+3*j:n+3*j+3]=-W*np.array(operators[start:start+3])[:,:6].T
        E.append(root);rhs.append(-bias[:6]);ineq=[];limits=[];contact_tasks=[]
        frozen=set(getattr(self,'frozen_controller_joints',[]))
        for name in frozen:
            i=b.ind_names.index(name);row=np.zeros((1,nv));row[0,6+i]=1;E.append(row);rhs.append(np.zeros(1))
        for j,((start,count),(i,frame,surface)) in enumerate(zip(blocks,point_ids)):
            bid,local,_,_=self.points[i];J,p=b.jacobian(bid,local);Jd=J.copy();Jrd=J.copy();mujoco.mj_jacDot(m,d,Jd,Jrd,p,bid);conv=frame@(Jd@d.qvel+J@curve);C=np.array(operators[start:start+3]);v=C@u
            row=np.zeros((1,nv));row[0,:n]=C[2];normal_goal=np.array([-conv[2]-40*v[2]-4000*float(gaps[start+2])])
            if getattr(self,'soft_contact_tracking',False):
                if getattr(self,'native_compliant_contact',False):normal_goal=np.array([-conv[2]-20*v[2]])
                contact_tasks.append((row,normal_goal,getattr(self,'normal_tracking_weight',1000.)))
                tangent=np.zeros((2,nv));tangent[:,:n]=C[:2];contact_tasks.append((tangent,-conv[:2]-30*v[:2],getattr(self,'tangent_tracking_weight',100.)))
            else:E.append(row);rhs.append(normal_goal)
            if getattr(self,'soft_contact_tracking',False):pass
            elif np.linalg.norm(v[:2])<.02:
                row=np.zeros((2,nv));row[:,:n]=C[:2];E.append(row);rhs.append(-conv[:2]-30*v[:2])
            else:
                direction=v[:2]/np.linalg.norm(v[:2]);phis=np.linspace(0,2*np.pi,12,endpoint=False);radial_mu=.8*np.cos(np.pi/12)/np.max(np.cos(phis)*direction[0]+np.sin(phis)*direction[1]);row=np.zeros((2,nv));row[:,n+3*j:n+3*j+3]=np.c_[np.eye(2),radial_mu*direction];E.append(row);rhs.append(np.zeros(2))
            row=np.zeros(nv);row[n+3*j+2]=1;ineq.append(row);limits.append(0.)
            for phi in np.linspace(0,2*np.pi,12,endpoint=False):
                row=np.zeros(nv);row[n+3*j:n+3*j+3]=[-np.cos(phi),-np.sin(phi),.8*np.cos(np.pi/12)];ineq.append(row);limits.append(0.)
        goal=np.r_[np.zeros(6),100*(target-q[r.ids])-20*u[6:],np.zeros(3*nc)];weights=np.r_[np.full(3,.01),np.full(3,2.),np.ones(n-6),np.full(3*nc,.02)];Q=np.diag(weights);linear=weights*goal
        for C,ref,weight in contact_tasks:Q+=weight*C.T@C;linear+=weight*C.T@ref
        C0=d.subtree_com[1].copy();V=Jcom@B@u;t=getattr(self,'elapsed',0.)
        if not hasattr(self,'impact_z'):
            self.impact_z=float(C0[2]);self.impact_vz=float(V[2]);self.bottom_z=max(self.height+.55,self.impact_z-.20)
            self.capture_omega=np.sqrt(9.80665/max(.3,self.impact_z-self.height));self.capture_goal_xy=C0[:2]+V[:2]/self.capture_omega
            self.absorb=Polynomial(self.impact_z,self.impact_vz,-9.80665,self.bottom_z,0,0,.20)
            self.recover=Polynomial(self.bottom_z,0,0,self.stand_COM_z,0,0,.9)
        poly=self.absorb if t<.20 else self.recover;s=t if t<.20 else min(t-.20,.9);zref,vref,az=[float(poly.value(s,k)[0]) for k in range(3)]
        # Capture-point target is a constant-height balance heuristic, not a
        # replacement for the full multibody/contact equations below.
        om=self.capture_omega;acc=np.r_[om**2*(self.capture_goal_xy-C0[:2])-2*om*V[:2],az+80*(zref-C0[2])+18*(vref-V[2])]
        conv=bias[:3]/m.body_mass.sum()+np.array([0.,0.,-9.80665]);C=np.zeros((3,nv));C[:,:n]=Jcom@B;Q+=10000*C.T@C;linear+=10000*C.T@(acc-conv)
        R=Rotation.from_quat(np.roll(q[3:7],-1));err=(R.inv()*Rotation.from_rotvec([pitch,0,0])).as_rotvec();C=np.zeros((3,nv));C[:,3:6]=np.eye(3);Q+=1000*C.T@C;linear+=1000*C.T@(80*err-18*u[3:6])
        for i,(x,v,lo,hi) in enumerate(zip(q[r.ids],u[6:],m.jnt_range[b.independent,0],m.jnt_range[b.independent,1])):
            if b.ind_names[i] in frozen:continue
            row=np.zeros(nv);row[6+i]=1;ineq.extend([row,-row]);limits.extend([-80*v+1600*(lo+1e-5-x),80*v-1600*(hi-1e-5-x)])
        E=np.vstack(E);rhs=np.concatenate(rhs);scale=np.maximum(np.linalg.norm(E,axis=1),.02);E/=scale[:,None];rhs/=scale;_,tri,piv=qr(E.T,pivoting=True,mode='economic');rank=int((abs(np.diag(tri))>1e-8).sum());keep=piv[:rank]
        sol=quadprog.solve_qp(Q,linear,np.vstack([E[keep],ineq]).T,np.r_[rhs[keep],limits],meq=rank)[0]
        force=np.zeros(n)
        for j,(start,count) in enumerate(blocks):force+=np.array(operators[start:start+3]).T@(W*sol[n+3*j:n+3*j+3])
        tau=Mr@sol[:n]+bias-force
        if abs(tau[:6]).max()>1e-4:raise RuntimeError('Landing controller root wrench is nonzero')
        tau[:6]=0.;return tau

    def step(self,q,u,target,pitch_target):
        r=self.r;b,m,d=r.b,r.m,r.d;dt=self.dt;B,curve=r.set_state(q,u);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);Mr=B.T@M@B;bias=B.T@(d.qfrc_bias+M@curve);Minv=np.linalg.inv(Mr)
        Jcom=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,Jcom,1);C=d.subtree_com[1].copy();V=Jcom@B@u
        P=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,*_ in self.points]);support_y=float(P[:,1].mean()+.03)
        desired=target.copy()
        for side in ['r','l']:
            j=b.ind_names.index('ankle_angle_'+side);desired[j]+=1.6*(C[1]-support_y)-.20*V[1]
        lo=m.jnt_range[b.independent,0];hi=m.jnt_range[b.independent,1];desired=np.minimum(np.maximum(desired,lo+.015),hi-.015)
        aj=100*(desired-q[r.ids])-20*u[6:];tau=Mr@np.r_[np.zeros(6),aj]+bias;tau[:6]=0.
        # This is an internal hip torque pair. The pelvis receives the opposite
        # moment; environmental angular momentum can change only via contact.
        R=Rotation.from_quat(np.roll(q[3:7],-1));error=(R.inv()*Rotation.from_rotvec([pitch_target,0,0])).as_rotvec();moment=120*error-25*u[3:6]
        hip=[(i,b.joints[n]) for i,n in enumerate(b.ind_names) if n.startswith(('hip_flexion_','hip_adduction_','hip_rotation_'))]
        axes=np.column_stack([-R.as_matrix().T@d.xaxis[j] for i,j in hip]);extra=axes.T@np.linalg.solve(axes@axes.T+1e-6*np.eye(3),moment)
        for (i,j),value in zip(hip,extra):tau[6+i]+=value
        free=u+dt*(Minv@(tau-bias));operators=[];gaps=[];blocks=[];point_ids=[]
        for i,((bid,local,_,_),p) in enumerate(zip(self.points,P)):
            J,_=b.jacobian(bid,local);J=J@B
            candidates=[(float(p[2]),np.array([0.,0.,1.]),'floor')]
            if self.height>0:candidates.append((*box_distance(p,self.front,self.height),'box'))
            for gap,normal,surface in candidates:
                vn=float(normal@J@free)
                if gap>.002 and gap+dt*vn>.0001:continue
                if gap>.004:continue
                if normal[2]>.9:t1=np.array([1.,0,0])
                else:t1=np.cross(normal,[0,0,1.]);t1/=np.linalg.norm(t1)
                t2=np.cross(normal,t1);frame=np.vstack([t1,t2,normal]);start=len(operators);operators.extend(frame@J);gaps.extend([0.,0.,gap]);blocks.append((start,3));point_ids.append((i,frame,surface))
                self.max_penetration=max(self.max_penetration,max(0.,-gap))
        physical_size=len(operators);limit_blocks=[]
        if blocks:
            try:tau=self.stance_torque(q,u,desired,pitch_target,Mr,bias,B,curve,Jcom,P,operators,gaps,blocks,point_ids)
            except (ValueError,RuntimeError) as e:
                self.controller_fallbacks=getattr(self,'controller_fallbacks',0)+1
                if not hasattr(self,'first_controller_failure'):self.first_controller_failure=str(e)
            free=u+dt*(Minv@(tau-bias))
        for i,(x,v,l,h) in enumerate(zip(q[r.ids],free[6:],lo,hi)):
            for sign,gap in [(1.,x-l),(-1.,h-x)]:
                if gap<.002 or gap+dt*sign*v<0:
                    row=np.zeros(r.n);row[6+i]=sign;limit_blocks.append(len(operators));operators.append(row);gaps.append(gap)
        impulse_world=np.zeros((len(P),3));root_impulse=np.zeros(r.n);normal_error=0.
        if operators:
            J=np.array(operators);H=J@Minv@J.T;H=(H+H.T)/2;w=J@free;gap=np.array(gaps);normal_rows=[start+2 for start,count in blocks]+limit_blocks
            for j in normal_rows:w[j]+=max(gap[j],-.003)/dt
            constraints=[];bounds=[]
            for start,count in blocks:
                row=np.zeros(len(J));row[start+2]=1;constraints.append(row);bounds.append(0.)
                for phi in np.linspace(0,2*np.pi,12,endpoint=False):
                    row=np.zeros(len(J));row[start:start+3]=[-np.cos(phi),-np.sin(phi),.8*np.cos(np.pi/12)];constraints.append(row);bounds.append(0.)
            for j in limit_blocks:
                row=np.zeros(len(J));row[j]=1;constraints.append(row);bounds.append(0.)
            for j in normal_rows:constraints.append(H[j]);bounds.append(-w[j])
            impulse=quadprog.solve_qp(H+np.eye(len(J))*1e-7,-w,np.array(constraints).T,np.array(bounds))[0]
            try:impulse=self.complementarity_refine(H,w,impulse,blocks,normal_rows)
            except ValueError as e:
                self.contact_refinement_fallbacks=getattr(self,'contact_refinement_fallbacks',0)+1
                self.contact_refinement_failure=str(e)
            normal_velocity=H@impulse+w
            for start,count in blocks:self.max_normal_complementarity=max(self.max_normal_complementarity,float(abs(impulse[start+2]*normal_velocity[start+2])))
            normal_error=float(max(0.,np.max(np.array(bounds)-np.array(constraints)@impulse)));root_impulse=J.T@impulse;post=free+Minv@root_impulse
            for (start,count),(i,frame,surface) in zip(blocks,point_ids):impulse_world[i]+=frame.T@impulse[start:start+3]
            if limit_blocks:self.joint_stop_impulse+=float(abs(impulse[limit_blocks]).sum())
        else:post=free
        residual=Mr@(post-u)-dt*(tau-bias)-root_impulse;self.max_dynamics=max(self.max_dynamics,float(abs(residual).max()));self.max_constraint=max(self.max_constraint,normal_error)
        nq=q.copy();mujoco.mj_integratePos(m,nq,B@post,dt);nq=b.complete(nq)
        return nq,post,impulse_world,P


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('--output-dir',required=True);ap.add_argument('--height',type=float,default=.3);ap.add_argument('--front',type=float,default=-.14);ap.add_argument('--seconds',type=float,default=1.6);ap.add_argument('--hz',type=int,default=1000);args=ap.parse_args();source=ROOT/args.source;out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
    data=np.load(source/'motion.npz');meta=json.loads((source/'summary.json').read_text());r=MeasuredJumpRig(87,1.78);b,m,d=r.b,r.m,r.d;ids=np.array(r.ids);to=meta['phase_times_s']['end_ascent'];ts=data['times'];pos=CubicSpline(ts,data['qpos'][:,:3],axis=0);angles=CubicSpline(ts,data['qpos'][:,ids],axis=0);rot=RotationSpline(ts,Rotation.from_quat(np.roll(data['qpos'][:,3:7],-1,axis=1)));vel=CubicSpline(ts,np.c_[data['qvel'][:,:6],data['qvel'][:,r.dofs]],axis=0)
    r.contact_stand_override=data['qpos'][0].copy();points=contact_geometry(r)['points']
    def state(t):
        q=data['qpos'][0].copy();q[:3]=pos(t);q[3:7]=np.roll(rot(t).as_quat(),1);q[ids]=angles(t);return b.complete(q),vel(t)
    def clearance(t):
        q,u=state(t);b.pose(q);P=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);values=[(float(p[2]),'floor',i,np.array([0.,0,1.])) for i,p in enumerate(P)]
        if args.height>0:
            for i,p in enumerate(P):
                gap,n=box_distance(p,args.front,args.height);values.append((gap,'box',i,n))
        return min(values,key=lambda x:x[0])
    flight_times=ts[ts>to+.015];event=None
    previous=to+.015;old=clearance(previous)[0]
    for t in flight_times:
        gap=clearance(float(t))[0]
        if old>0 and gap<=0:
            contact_time=brentq(lambda t:clearance(t)[0],previous,float(t),xtol=1e-10);event=(contact_time,clearance(contact_time));break
        previous=float(t);old=gap
    if event is None:
        (out/'landing_status.json').write_text(json.dumps(dict(status='no_contact_in_available_flight',height_m=args.height,front_y_m=args.front,landing_simulated=False),indent=2));print('NO_CONTACT');return
    contact_time,(_,surface,point,normal)=event
    if surface=='box' and normal[2]<.9:
        (out/'landing_status.json').write_text(json.dumps(dict(status='rejected_front_or_side_collision',contact_time_s=contact_time,normal=normal.tolist(),height_m=args.height,front_y_m=args.front,landing_simulated=False),indent=2));print('BOX_FRONT_COLLISION',contact_time,normal);return
    q,u=state(contact_time);qimpact=q.copy();uimpact=u.copy();sim=Landing(r,points,args.front,args.height,1/args.hz);sim.stand_COM_z=args.height+float(data['com'][0,2]);low,_=r.planted_pose(1.5,1.3,.09,arms=(np.pi/2,.1,.2),trunk=0.);crouch=low[ids].copy();stand=data['qpos'][0,ids].copy();pitch_low=float(Rotation.from_quat(np.roll(low[3:7],-1)).as_rotvec()[0]);arr={k:[] for k in ['times','qpos','u','body_xpos','body_xmat','com','contact_impulses_Ns','contact_positions_m']};failure=None
    for k in range(round(args.seconds*args.hz)+1):
        t=k/args.hz
        sim.elapsed=t
        if t<.20:f=float(s5(t/.20));target=(1-f)*qimpact[ids]+f*crouch;pitch=pitch_low
        else:f=float(s5((t-.20)/.9));target=(1-f)*crouch+f*stand;pitch=(1-f)*pitch_low
        try:nq,nu,impulse,P=sim.step(q,u,target,pitch)
        except Exception as e:failure=str(e);break
        if k%max(1,args.hz//125)==0:
            r.set_state(q,u);values=dict(times=contact_time+t,qpos=q.copy(),u=u.copy(),body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),contact_impulses_Ns=impulse.copy(),contact_positions_m=P.copy())
            for key,v in values.items():arr[key].append(v)
        if k%250==0:print('CONTACT_LANDING',round(t,3),'COM',np.round(d.subtree_com[1],3),'impulse',round(float(impulse[:,2].sum()),3),flush=True)
        q,u=nq,nu
        if q[2]<args.height+.2:failure='Pelvis dropped below viable landing posture; body collisions are not modeled.';break
    arrays={key:np.array(value) for key,value in arr.items()};np.savez_compressed(out/'landing_motion.npz',**arrays);r.set_state(q,u);mujoco.mj_subtreeVel(m,d);endV=d.subtree_linvel[1].copy();P=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);on_top=bool(np.all(P[:,1]<=args.front+.005)&np.all(P[:,1]>=args.front-.75-.005)&np.all(abs(P[:,0]+.025)<=.405)&np.all(abs(P[:,2]-args.height)<.015));ROM=max(0.,float(np.maximum(m.jnt_range[b.independent,0]-q[ids],q[ids]-m.jnt_range[b.independent,1]).max()))
    if args.height<=0:on_top=bool(np.all(abs(P[:,2])<.015))
    report=dict(status='landing_candidate' if failure is None else 'failed_landing',failure=failure,contact_surface=surface,contact_time_s=contact_time,contact_point=int(point),box_height_m=args.height,box_front_y_m=args.front,landing_simulated=True,completed_supported_landing=bool(failure is None and on_top and np.linalg.norm(endV)<.1 and ROM<.002),feet_on_top=on_top,final_COM_velocity_m_s=endV.tolist(),max_contact_inequality_residual=sim.max_constraint,max_discrete_dynamics_residual=sim.max_dynamics,max_contact_penetration_m=sim.max_penetration,max_normal_complementarity_J=sim.max_normal_complementarity,joint_stop_impulse_total_Nm_s=sim.joint_stop_impulse,ROM_excess_rad=ROM,contact_model='Convex impulse/friction-cone approximation; strict normal complementarity audited separately.',impacts='Rigid, inelastic; impulse is resolved, physical peak force is not.',root_actuator=False,strength_ceiling=False,force_analysis_allowed=False,human_motion_validated=False)
    report['controller_fallback_steps']=getattr(sim,'controller_fallbacks',0)
    report.update(first_controller_failure=getattr(sim,'first_controller_failure',None),max_contact_iterations=getattr(sim,'max_contact_iterations',0),contact_iteration_limit_steps=getattr(sim,'contact_iteration_limit_steps',0))
    report.update(contact_refinement_fallbacks=getattr(sim,'contact_refinement_fallbacks',0),contact_refinement_failure=getattr(sim,'contact_refinement_failure',None))
    (out/'landing_status.json').write_text(json.dumps(report,indent=2));print('LANDING_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
