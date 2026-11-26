"""Moderate vertical-jump mechanics reference, isolated from rejected runs.

GRF/COP and movement goals are declared inputs. Exact floating-root dynamics,
sticking support and numerical convergence are required before tension fitting.
No force clipping or anatomical strength limit is used.
"""
import argparse, hashlib, json
import numpy as np
import mujoco, quadprog
from scipy.linalg import qr
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation
from rigid_baseline_profile import RigidBaselineRig
from centroidal_jump import Polynomial, integrate, cop_plan, s5, ReferenceMotion, G
from jump_acceleration import AccelerationMotion
from verify_jump_reference import normal_weights
from jump_rig import ROOT


def plan_jump(rig,hz):
    b,m=rig.b,rig.m; mass=float(m.body_mass.sum());gscale=b.report['global_normalization_scale']
    stand,_=rig.planted_pose(.05,.12,.09,arms=(1.4,.25))
    b.pose(stand);c0=rig.d.subtree_com[1].copy();feet={};points=[];ground=.025*gscale
    for side in ['r','l']:
        bid=m.body('calcn_'+side).id;sid=m.site('subject_sole_reference_'+side).id
        feet[side]={'bid':bid,'local':m.site_pos[sid].copy(),'p':rig.d.site_xpos[sid].copy(),'R':rig.d.xmat[bid].reshape(3,3).copy()}
        for x,y in [(-.037,-.245),(.037,-.245),(-.037,.010),(.037,.010)]:
            p=feet[side]['p']+np.array([x,y,.025])*gscale
            points.append((bid,feet[side]['R'].T@(p-rig.d.xpos[bid]),p.copy(),y<0))
    hold=.25;bottom=.70;braked=1.05;propT=.35;takeoff=braked+propT
    t=np.unique(np.r_[np.arange(0,takeoff,1/hz),hold,bottom,braked,takeoff]);az=np.zeros(len(t))
    mask=(t>=hold)&(t<bottom);u=(t[mask]-hold)/(bottom-hold);az[mask]=-2.8*np.sin(np.pi*u)**2
    mask=(t>=bottom)&(t<braked);u=(t[mask]-bottom)/(braked-bottom);az[mask]=3.6*np.sin(np.pi*u)**2
    mask=t>=braked;u=(t[mask]-braked)/propT;vto=1.6;P=np.pi/2*(vto/propT+G/2);az[mask]=-G*u+P*np.sin(np.pi*u)
    z,vz=integrate(t,az,c0[2]);F=np.c_[np.zeros((len(t),2)),mass*(az+G)];F[-1]=0
    com=np.c_[np.full(len(t),c0[0]),np.full(len(t),c0[1]),z];vel=np.c_[np.zeros((len(t),2)),vz]
    toe_start=takeoff-.10;miny=min(p[2][1] for p in points);maxy=max(p[2][1] for p in points);toey=np.mean([p[2][1] for p in points if p[3]])
    forward=Polynomial(c0[1],0,0,toey+.025,0,0,takeoff-hold)
    for i,ti in enumerate(t):
        if ti>=hold:
            ss=ti-hold;com[i,1]=forward.value(ss)[0];vel[i,1]=forward.value(ss,1)[0];F[i,1]=mass*forward.value(ss,2)[0]
    F[-1]=0
    ix=np.unique(np.r_[np.arange(0,len(t),max(1,hz//120)),len(t)-1]);lo=np.full(len(ix),miny+.001);hi=np.full(len(ix),maxy-.001)
    # COP travels to the toe smoothly; it is not moved discontinuously at release.
    blend=s5((t[ix]-(toe_start-.15))/.15);lo=(1-blend)*lo+blend*toey;hi=(1-blend)*hi+blend*toey
    cs,_=cop_plan(t[ix],com[ix]-[0,0,ground],F[ix],lo,hi);cop=np.interp(t,t[ix],cs);cop[t>=toe_start]=toey
    H=cumulative_trapezoid((cop-com[:,1])*F[:,2]+(com[:,2]-ground)*F[:,1],t,initial=0)
    zland=c0[2]-.08;Tf=(vz[-1]+np.sqrt(vz[-1]**2+2*G*(z[-1]-zland)))/G
    Ta=.30;at=np.unique(np.r_[np.arange(0,Ta,1/hz),Ta]);u=at/Ta;vland=vz[-1]-G*Tf
    normal=mass*G*(u+(1-2*vland/(G*Ta))*np.sin(np.pi*u)**2)
    yz,yv=integrate(at,np.c_[np.zeros(len(at)),normal/mass-G],[com[-1,1],zland],[0,vland])
    cl=np.c_[np.full(len(at),c0[0]),yz];vl=np.c_[np.zeros(len(at)),yv];Fl=np.c_[np.zeros((len(at),2)),normal]
    shift=com[-1,1]-c0[1];land_feet={s:{**p,'p':p['p']+np.array([0,shift,0])} for s,p in feet.items()}
    ix=np.unique(np.r_[np.arange(0,len(at),max(1,hz//120)),len(at)-1]);cs,_=cop_plan(at[ix],cl[ix]-[0,0,ground],Fl[ix],np.full(len(ix),miny+shift+.001),np.full(len(ix),maxy+shift-.001),H[-1]);copl=np.interp(at,at[ix],cs)
    Hl=H[-1]+cumulative_trapezoid((copl-cl[:,1])*normal,at,initial=0)
    return dict(stand=stand,feet=feet,points=points,ground=ground,t_ground=t,com_ground=com,vel_ground=vel,F_ground=F,cop_ground=cop,H_ground=H,takeoff=takeoff,propT=propT,toe_start=toe_start,flight_T=Tf,t_land=at,com_land=cl,vel_land=vl,F_land=Fl,cop_land=copl,H_land=Hl,land_feet=land_feet,max_friction_ratio=0.,hold=hold,bottom=bottom,braked=braked)


class JumpMotion(AccelerationMotion):
    def __init__(self,rig,plan,hz):
        super().__init__(rig,plan,'vertical',hz)
        self.crouch=rig.planted_pose(.65,.95,.09,arms=(1.4,.12),trunk=.10)[0]
        self.extend=plan['stand'].copy();self.reach=plan['stand'].copy();self.land=self.crouch.copy()
        rig.arm(self.extend,1.4,1.35,.25);rig.arm(self.reach,1.4,1.45,.25);rig.arm(self.land,1.4,.70,.3)
        for side in ['r','l']:
            for name,val in [('hip_flexion',.20),('knee_angle',.35),('ankle_angle',-.15)]:
                self.reach[self.m.jnt_qposadr[self.b.joints[name+'_'+side]]]=val

    def reference(self,t):
        p=self.p
        if t<p['hold']:return p['stand'].copy()
        if t<p['bottom']:return p['stand']+s5((t-p['hold'])/(p['bottom']-p['hold']))*(self.crouch-p['stand'])
        if t<p['takeoff']:
            # Arm acceleration is spread over braking AND push-off.
            return self.crouch+s5((t-p['bottom'])/(p['takeoff']-p['bottom']))*(self.extend-self.crouch)
        if t<self.t_land:
            u=(t-p['takeoff'])/p['flight_T']
            if u<.45:return self.extend+s5(u/.45)*(self.reach-self.extend)
            return self.reach+s5((u-.45)/.55)*(self.land-self.reach)
        if t<self.t_absorb:return self.land.copy()
        return self.land+s5((t-self.t_absorb)/.85)*(p['stand']-self.land)

    def capture_flight(self,q,u):
        super().capture_flight(q,u)
        for path in self.flight_paths.values():path['bump']=.18

    def acceleration(self,q,u,t,mode):
        m,d=self.m,self.d;B,curve=self.rig.set_state(q,u);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M)
        *_,force,H,cop=self.state_target(t);external=np.zeros(m.nv);points=[];jac=[]
        for bid,local,_,fore in self.p['points']:
            J,p=self.b.jacobian(bid,local);points.append(p);jac.append(J)
        points=np.array(points);active=[i for i,p in enumerate(self.p['points']) if mode!='toe' or p[3]]
        if force[2]>1e-8:
            weights=normal_weights(points[active],d.subtree_com[1,0],cop)
            for i,w in zip(active,weights):external+=jac[i].T@(w*force)
        rows=[(M@B)[:6]];rhs=[(external+d.qfrc_passive-d.qfrc_bias-M@curve)[:6]];soft=[]
        for side,foot in self.p['feet'].items():
            bid=foot['bid'];targets=[(p[1],p[2]) for p in self.p['points'] if p[0]==bid and p[3]] if mode=='toe' else [(foot['local'],self.foot_target(side,t,mode)[0])]
            for local,target in targets:
                J,p=self.b.jacobian(bid,local);Jdot=np.zeros_like(J);Jrdot=np.zeros_like(J);mujoco.mj_jacDot(m,d,Jdot,Jrdot,p,bid)
                targetv=np.zeros(3);targeta=np.zeros(3)
                if mode=='flight':
                    eps=1e-5;targetv=self.foot_target(side,t,mode)[1];targeta=(self.foot_target(side,t+eps,mode)[1]-self.foot_target(side,t-eps,mode)[1])/(2*eps)
                C=J@B;v=targeta+20*(targetv-J@d.qvel)+100*(target-p)-Jdot@d.qvel-J@curve
                rows.append(C);rhs.append(v)
            if mode not in ['toe','flight']:
                _,_,Rt,omega=self.foot_target(side,t,mode);J=np.zeros((3,m.nv));Jr=np.zeros_like(J);Jdot=np.zeros_like(J);Jrdot=np.zeros_like(J)
                mujoco.mj_jac(m,d,J,Jr,d.xpos[bid],bid);mujoco.mj_jacDot(m,d,Jdot,Jrdot,d.xpos[bid],bid)
                err=Rotation.from_matrix(Rt@d.xmat[bid].reshape(3,3).T).as_rotvec();rows.append(Jr@B);rhs.append(20*(omega-Jr@d.qvel)+100*err-Jrdot@d.qvel-Jr@curve)
            elif mode=='flight':
                # Airborne orientation is a soft landing goal, not a constraint.
                _,_,Rt,omega=self.foot_target(side,t,mode);J=np.zeros((3,m.nv));Jr=np.zeros_like(J);mujoco.mj_jac(m,d,J,Jr,d.xpos[bid],bid)
                err=Rotation.from_matrix(Rt@d.xmat[bid].reshape(3,3).T).as_rotvec();soft.append((Jr@B,20*(omega-Jr@d.qvel)+100*err,.2))
        A=np.vstack(rows);target=np.concatenate(rhs);scale=np.maximum(np.linalg.norm(A,axis=1),.02);A/=scale[:,None];target/=scale
        desired=self.reference(t);eps=1e-4;qd=(self.reference(t+eps)-self.reference(t-eps))/(2*eps)
        posture=np.zeros(self.rig.n);R=Rotation.from_quat(np.r_[q[4:7],q[3]]);Rd=Rotation.from_quat(np.r_[desired[4:7],desired[3]])
        posture[3:6]=36*(R.inv()*Rd).as_rotvec()-12*u[3:6]
        posture[6:]=100*(desired[self.rig.ids]-q[self.rig.ids])-20*(u[6:]-qd[self.rig.ids])
        weights=np.r_[np.full(3,.001),np.full(3,.2),np.ones(self.rig.n-6)];P=np.diag(weights);objective=-weights*posture
        for C,value,strength in soft:P+=strength*C.T@C;objective-=strength*C.T@value
        lower=np.full(self.rig.n,-np.inf);upper=np.full(self.rig.n,np.inf)
        for i,jid in enumerate(self.b.independent):
            if m.jnt_limited[jid]:
                lo,hi=m.jnt_range[jid];x=q[self.rig.ids[i]];speed=u[6+i];k=40.
                lower[6+i]=-2*k*speed+k*k*(lo+1e-5-x);upper[6+i]=-2*k*speed+k*k*(hi-1e-5-x)
        _,tri,piv=qr(A.T,pivoting=True,mode='economic');rank=int(np.sum(abs(np.diag(tri))>1e-8));keep=piv[:rank];I=np.eye(self.rig.n);il=np.where(np.isfinite(lower))[0];iu=np.where(np.isfinite(upper))[0]
        C=np.vstack([A[keep],I[il],-I[iu]]);limit=np.r_[target[keep],lower[il],-upper[iu]]
        try:sol=quadprog.solve_qp(P,-objective,C.T,limit,meq=rank)[0]
        except ValueError:
            out=ROOT/'results/jump_v2/failure.npz';np.savez(out,q=q,u=u,t=t,A=A,target=target,lower=lower,upper=upper)
            raise RuntimeError(('Infeasible acceleration',t,mode))
        error=float(abs(A@sol-target).max())
        if error>1e-4:raise RuntimeError(('Root/contact residual',t,error))
        return sol,B,curve


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hz',type=int,default=600);a=ap.parse_args();rig=RigidBaselineRig();plan=plan_jump(rig,a.hz);motion=JumpMotion(rig,plan,a.hz);dest=ROOT/f'results/jump_v2/{a.hz}hz';dest.mkdir(parents=True,exist_ok=True)
    data=motion.run();np.savez_compressed(dest/'motion.npz',**data)
    deps=['build_jump_v2.py','jump_acceleration.py','centroidal_jump.py','jump_rig.py','parametric_body.py','rigid_baseline_profile.py']
    (dest/'setup.json').write_text(json.dumps(dict(profile=rig.profile,physics_hz=a.hz,coordinate_names=rig.b.coordinate_names,body_names=[rig.m.body(i).name for i in range(rig.m.nbody)],takeoff_s=plan['takeoff'],landing_s=motion.t_land,duration_s=motion.duration,human_motion_validated=False,GRF_and_COP_are_inputs=True,airborne_foot_orientation='soft landing goal',airborne_foot_position='prescribed geometry, not physical contact',source_sha256={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in deps}),indent=2))
    (dest/'contact_definition.json').write_text(json.dumps(dict(points=[dict(body_id=bid,body=rig.m.body(bid).name,local_m=local.tolist(),initial_world_m=p.tolist(),forefoot=fore) for bid,local,p,fore in plan['points']],mass_kg=float(rig.m.body_mass.sum()),gravity_m_s2=G,ground_z_m=plan['ground']),indent=2))
    print('JUMP_V2_COMPLETE',dest,flush=True)


if __name__=='__main__':main()
