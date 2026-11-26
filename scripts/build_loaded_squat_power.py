"""Continuous full-body squat with planted feet and explicit bar inertia.

Posture depth and cadence are inputs. Contact wrench is a Newton--Euler output,
distributed inside friction cones without a floating-root support actuator.
This is a mechanically checked scenario, not a measured personal performance.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco,osqp
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from retarget_measured_jump import MeasuredJumpRig
from mechanics_engine_adapter import sample
from newton_euler_reference import accumulate,at_point

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/power_comparison'
G=9.80665


def skew(p):
    x,y,z=p
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def smooth7(u):
    u=np.clip(u,0,1)
    return u**4*(35-84*u+70*u*u-20*u**3)


class SquatPath:
    def __init__(self,rig,bar_mass,arm_seed=None):
        self.rig=rig;self.b=rig.b;self.m=rig.m;self.d=rig.d;self.bar_mass=bar_mass
        m,d,b=self.m,self.d,self.b
        self.length_scale=b.subject.height_m/1.8
        self.seed=b.q0.copy() if arm_seed is None else arm_seed.copy()
        if arm_seed is None:rig.arm(self.seed,1.4,1.0,1.9)
        if bar_mass>0:
            for j in range(m.njnt):
                name=m.joint(j).name;value=None
                if name.startswith('mcp') and 'flexion' in name:value=1.1
                elif name.startswith(('pm2','pm3','pm4','pm5')):value=1.2
                elif name.startswith(('md2','md3','md4','md5')):value=.8
                elif name.startswith('cmc_flexion'):value=.3
                elif name.startswith('cmc_abduction'):value=.55
                elif name.startswith('mp_flexion'):value=.9
                elif name.startswith('ip_flexion'):value=.8
                if value is not None:self.seed[m.jnt_qposadr[j]]=np.clip(value,*m.jnt_range[j])
        b.pose(self.seed)
        shoulder=np.mean([d.xanchor[b.joints['shoulder_rot_'+s]] for s in ['r','l']],axis=0)
        self.chest=m.body('chest_r').id
        self.bar_reference_R=d.xmat[self.chest].reshape(3,3).copy()
        self.bar0=shoulder+np.array([0,.065,.035])*self.length_scale
        self.bar_local=d.xmat[self.chest].reshape(3,3).T@(self.bar0-d.xpos[self.chest])
        preset=ROOT/'models/grips/high_bar_28mm.json'
        if bar_mass>0 and preset.exists():
            grip=json.loads(preset.read_text())
            if grip['source_anatomy_sha256']!=hashlib.sha256((ROOT/'assets/fullbody/fullbody_muscles_skin.glb').read_bytes()).hexdigest():raise RuntimeError('Grip preset belongs to different skin geometry')
            for name,value in grip['pose_joint_values_rad'].items():self.seed[m.jnt_qposadr[b.joints[name]]]=value
            b.pose(self.seed);errors=[]
            for side in ['r','l']:
                local=np.array(grip['grasp_centres_unscaled_m'][side])*b.report['global_normalization_scale']
                _,p=b.jacobian(m.body('lunate_'+side).id,local)
                target=self.bar0+np.array([-.48 if side=='r' else .48,0,0])*self.length_scale;errors.append(np.linalg.norm(p-target))
            self.arm_fit=dict(status='surface_checked_grasp_preset',max_component_hand_error_m=float(max(errors)),grip_width_m=.96*self.length_scale,
                              grasp_centres_in_lunate_frame_m=grip['grasp_centres_unscaled_m'],surface_audit=grip['surface_audit'],
                              original_joint_ranges_retained=True,skin_geometry_modified=False)
            if max(errors)>.0001:raise RuntimeError(('Grasp frame mismatch',errors))
        else:self.arm_fit=self.rack_arms() if bar_mass>0 else {'status':'source upper-body posture retained'}
        b.pose(self.seed)
        self.feet=[];self.points=[]
        for side in ['r','l']:
            bid=m.body('calcn_'+side).id;sid=m.site('subject_sole_reference_'+side).id
            origin=d.site_xpos[sid].copy();R=d.xmat[bid].reshape(3,3).copy()
            toe_out=Rotation.from_rotvec([0,0,np.deg2rad(-15 if side=='r' else 15)]).as_matrix()
            heel_pitch=Rotation.from_rotvec([np.arctan(.025/.255),0,0]).as_matrix()
            target=origin.copy();target[0]=-.17 if side=='r' else .17;target[2]=.024
            target_R=toe_out@heel_pitch@R
            self.feet.append((bid,m.site_pos[sid].copy(),target,target_R))
            target_body_origin=target-target_R@m.site_pos[sid]
            for x,y in [(-.04,-.245),(.04,-.245),(-.04,.01),(.04,.01)]:
                world=target+toe_out@np.array([x,y,0]);world[2]=0
                # A rigid shoe wedge: external sole contacts are on the floor;
                # anatomical heel sits 25 mm higher than the forefoot.
                local=target_R.T@(world-target_body_origin)
                self.points.append((bid,local,world,side))
        self.support_center=np.mean([p[2] for p in self.points],axis=0)[:2]
        self.names=[stem+'_'+s for stem in ['hip_adduction','hip_rotation','ankle_angle','subtalar_angle'] for s in ['r','l']]
        self.jids=[b.joints[n] for n in self.names];self.adrs=[int(m.jnt_qposadr[j]) for j in self.jids]
        self.lo=np.r_[[-.5,-.5,.15],[-1.,-.35,-.35],[m.jnt_range[j,0] for j in self.jids]]
        self.hi=np.r_[[.5,.5,1.5],[1.,.35,.35],[m.jnt_range[j,1] for j in self.jids]]
        self.last=np.r_[self.seed[:3],np.zeros(3),self.seed[self.adrs]]
        self.max_fit=0

    def rack_arms(self):
        b,m,d=self.b,self.m,self.d
        names=[stem+'_'+s for s in ['r','l'] for stem in ['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion']]
        adrs=[m.jnt_qposadr[b.joints[n]] for n in names]
        lo=np.array([m.jnt_range[b.joints[n],0] for n in names]);hi=np.array([m.jnt_range[b.joints[n],1] for n in names])
        for i,n in enumerate(names):
            if n.startswith('shoulder_elv'):hi[i]=min(hi[i],1.85)
        x0=np.maximum(np.minimum(self.seed[adrs],hi-1e-6),lo+1e-6)
        targets=[self.bar0+np.array([-.48 if s=='r' else .48,0,0]) for s in ['r','l']]
        elbow_targets=[self.bar0+np.array([-.35 if s=='r' else .35,.10,-.20]) for s in ['r','l']]
        def place(x):
            q=self.seed.copy();q[adrs]=x;d.qpos[:]=b.complete(q);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);return q
        def fun(x):
            place(x);errs=[];elbows=[];align=[]
            for s,target,elbow_target in zip(['r','l'],targets,elbow_targets):
                # The grasp centre lies inside the curled fingers, offset
                # from the palm. Putting the bar through the palm centre is
                # not a contact constraint.
                grasp_local=np.array([0,-.090,-.015 if s=='r' else .015])*b.report['global_normalization_scale']
                _,p=b.jacobian(m.body('lunate_'+s).id,grasp_local)
                errs.extend((p-target)*10000)
                elbows.extend((d.xanchor[b.joints['elbow_flexion_'+s]]-elbow_target)*30)
                elbows.append(max(0.,d.xanchor[b.joints['elbow_flexion_'+s],2]-(self.bar0[2]-.1))*200)
                hand_R=d.xmat[m.body('lunate_'+s).id].reshape(3,3)
                align.extend(np.cross(hand_R[:,0],[1.,0.,0.])*20)
            return np.r_[errs,align,elbows,.03*(x-x0)]
        guesses=[x0]
        for plane,rot in [(0.,-1.2),(0.,1.2),(-1.2,1.2),(1.5,-1.2)]:
            x=x0.copy()
            for side in range(2):x[side*7:side*7+4]=[plane,1.1,rot,2.1]
            guesses.append(np.maximum(np.minimum(x,hi-1e-6),lo+1e-6))
        opt=None
        for guess in guesses:
            fit=least_squares(fun,guess,bounds=(lo,hi),max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-8)
            if opt is None or np.linalg.norm(fun(fit.x))<np.linalg.norm(fun(opt.x)):opt=fit
            if np.max(abs(fun(opt.x)[:6]))<.01:break
        self.seed=place(opt.x);error=float(np.max(abs(fun(opt.x)[:6]))) /10000
        if error>.005:raise RuntimeError(('Rack hand contact not fitted',error))
        return dict(max_component_hand_error_m=error,grip_width_m=.96,grasp_centres_in_lunate_frame_m={'r':[0,-.090,-.015],'l':[0,-.090,.015]},bar_support='upper-back torso; hands stabilize',bar_mass_kg=self.bar_mass)

    def pose_at_depth(self,s):
        b,m,d=self.b,self.m,self.d;q=self.seed.copy()
        for side in ['r','l']:
            q[m.jnt_qposadr[b.joints['hip_flexion_'+side]]]=.08+1.97*s
            q[m.jnt_qposadr[b.joints['knee_angle_'+side]]]=.12+1.93*s
            q[m.jnt_qposadr[b.joints['mtp_angle_'+side]]]=0
        q[m.jnt_qposadr[b.joints['flex_extension']]]=.01-.15*s
        def place(x):
            z=q.copy();z[:3]=x[:3];z[3:7]=np.roll(Rotation.from_rotvec(x[3:6]).as_quat(),1);z[self.adrs]=x[6:]
            d.qpos[:]=b.complete(z);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            return d.qpos.copy()
        def residual(x):
            place(x);r=[]
            for bid,local,target,R0 in self.feet:
                R=d.xmat[bid].reshape(3,3);p=d.xpos[bid]+R@local
                r.extend((p-target)*10)
                r.extend(Rotation.from_matrix(R0.T@R).as_rotvec())
            bar=d.xpos[self.chest]+d.xmat[self.chest].reshape(3,3)@self.bar_local
            system=(m.body_mass.sum()*d.subtree_com[1]+self.bar_mass*bar)/(m.body_mass.sum()+self.bar_mass)
            r.extend((system[:2]-self.support_center)*10)
            return r
        x0=np.maximum(np.minimum(self.last,self.hi-1e-7),self.lo+1e-7)
        opt=least_squares(residual,x0,bounds=(self.lo,self.hi),max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-10)
        error=max(abs(np.array(residual(opt.x))));self.last=opt.x;result=place(opt.x)
        self.max_fit=max(self.max_fit,float(error))
        if error>.002:raise RuntimeError(('Planted squat fit',s,error,opt.message,opt.x.tolist()))
        return result

    def build(self,count=101):
        ss=np.linspace(0,1,count);poses=[]
        for i,s in enumerate(ss):
            q=self.pose_at_depth(float(s));poses.append(q)
            if i%25==0:print('SQUAT_PATH',self.bar_mass,i,flush=True)
        self.spline=CubicSpline(ss,np.array(poses),axis=0)
        return self

    def q(self,s):
        q=self.spline(np.clip(s,0,1));q[3:7]/=np.linalg.norm(q[3:7]);return self.b.complete(q)


def contact_solver(points,mass):
    W=mass*G;N=len(points)
    A=np.hstack([np.vstack([np.eye(3),skew(p)]) for p in points])
    constraints=[sp.csc_matrix(A)];lo=[np.zeros(6)];hi=[np.zeros(6)]
    for i in range(N):
        for phi in np.linspace(0,2*np.pi,16,endpoint=False):
            row=np.zeros(3*N);row[3*i:3*i+3]=[np.cos(phi),np.sin(phi),-.8*np.cos(np.pi/16)]
            constraints.append(sp.csc_matrix(row[None]));lo.append(np.array([-np.inf]));hi.append(np.array([0.]))
        row=np.zeros(3*N);row[3*i+2]=1
        constraints.append(sp.csc_matrix(row[None]));lo.append(np.array([0.]));hi.append(np.array([np.inf]))
    lower=np.concatenate(lo);upper=np.concatenate(hi)
    qp=osqp.OSQP();qp.setup(P=sp.eye(3*N,format='csc'),q=np.zeros(3*N),A=sp.vstack(constraints,format='csc'),l=lower,u=upper,
                           verbose=False,eps_abs=1e-9,eps_rel=1e-9,polishing=True,max_iter=20000)
    def solve(wrench):
        lower[:6]=upper[:6]=wrench/W;qp.update(l=lower,u=upper)
        r=qp.solve(raise_error=False)
        if not r.info.status.startswith('solved'):raise RuntimeError(('Contact allocation',r.info.status,wrench))
        forces=r.x.reshape(N,3)*W
        error=float(max(abs(A@forces.ravel()-wrench)))
        if error>1e-5:raise RuntimeError(('Ground wrench balance',error))
        return forces,error
    return solve


def build_trial(bar_mass,hz=250,descent=1.25,ascent=None,hold=.15,mass_kg=88.,height_m=1.8,label=None,bottom_knee_deg=None):
    # This varies depth within the existing coordinated high-bar path; it does
    # not predict a new movement strategy or extrapolate beyond that path.
    peak_depth=1. if bottom_knee_deg is None else (np.deg2rad(bottom_knee_deg)-.12)/1.93
    if not 0.<peak_depth<=1.+1e-10:
        raise ValueError(f'Bottom knee flexion must be above {np.degrees(.12):.3f} and at most {np.degrees(2.05):.3f} degrees for this path.')
    peak_depth=min(peak_depth,1.)
    rig=MeasuredJumpRig(mass_kg=mass_kg,height_m=height_m);b,m,d=rig.b,rig.m,rig.d
    default_label=f'squat_{bar_mass:g}'+(f'_knee{bottom_knee_deg:g}' if bottom_knee_deg is not None else '')
    path=SquatPath(rig,bar_mass).build();folder=OUT/(label or default_label);folder.mkdir(parents=True,exist_ok=True)
    if ascent is None:ascent={60:.80,100:.90,140:1.15}[bar_mass]
    assert descent>0 and ascent>0 and hold>=0 and bar_mass>=0
    start=.25;bottom=start+descent;up=bottom+hold;end=up+ascent;T=end+.25
    def depth(t):
        if t<start:return 0.
        if t<bottom:return peak_depth*float(smooth7((t-start)/descent))
        if t<up:return peak_depth
        if t<end:return peak_depth*float(1-smooth7((t-up)/ascent))
        return 0.
    def position(t):return path.q(depth(t))
    def state(t):
        eps=2e-5;q=position(t);qm=position(t-eps);qp=position(t+eps)
        v=np.zeros(m.nv);mujoco.mj_differentiatePos(m,v,2*eps,qm,qp)
        vm=np.zeros(m.nv);vp=np.zeros(m.nv)
        mujoco.mj_differentiatePos(m,vm,eps,qm,q);mujoco.mj_differentiatePos(m,vp,eps,q,qp)
        a=(vp-vm)/eps
        return q,v,a
    times=np.unique(np.round(np.r_[np.arange(0,T+1e-9,1/hz),start,bottom,up,end,T],10))
    allocate=contact_solver([p[2] for p in path.points],mass_kg+bar_mass)
    arrays={k:[] for k in ['times','qpos','qvel','qacc','body_xpos','body_xmat','com','velocity','system_COM_m','system_velocity_m_s','GRF_N','COP_m','bar_position_m','bar_rotation','bar_velocity_m_s','bar_acceleration_m_s2','bar_support_force_N','contact_positions_m','contact_forces_N','net_joint_moments_Nm','net_joint_powers_W','net_joint_reactions_N','COM_power_W','bar_power_W','segment_energy_J','segment_energy_rate_W','internal_power_W','balance_error_SI','engine_error_SI','foot_error_m','foot_speed_m_s','grasp_error_m']}
    selected=[stem+'_'+side for stem in ['hip_flexion','knee_angle','ankle_angle'] for side in ['r','l']]
    for i,t in enumerate(times):
        q,v,a=state(float(t));mech=sample(rig,q,v,a);R=d.xmat[path.chest].reshape(3,3)
        bar=d.xpos[path.chest]+R@path.bar_local;r=bar-d.xipos[path.chest]
        omega=mech['omega'][path.chest];alpha=mech['alpha'][path.chest]
        bv=mech['velocity'][path.chest]+np.cross(omega,r)
        ba=mech['acceleration'][path.chest]+np.cross(alpha,r)+np.cross(omega,np.cross(omega,r))
        # A 2.2 m rigid rod, negligible axial inertia but finite radius.
        bar_R=R@path.bar_reference_R.T
        Ibar=bar_R@np.diag([bar_mass*.014**2/2,bar_mass*2.2**2/12,bar_mass*2.2**2/12])@bar_R.T
        Fbar=bar_mass*(ba-m.opt.gravity);Tbar=Ibar@alpha+np.cross(omega,Ibar@omega)
        bar_wrench=np.r_[Fbar,np.cross(bar,Fbar)+Tbar]
        required=mech['body_wrenches'].sum(0)+bar_wrench
        ground,err=allocate(required)
        ext=np.zeros_like(mech['body_wrenches']);ext[path.chest]-=bar_wrench
        external_gen=np.zeros(m.nv)
        for (bid,local,p0,side),f in zip(path.points,ground):
            ext[bid,:3]+=f;ext[bid,3:]+=np.cross(p0,f)
            J,p=b.jacobian(bid,local);external_gen+=J.T@f
        J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,bar,path.chest)
        external_gen-=J.T@Fbar+Jr.T@Tbar
        generalized=mech['generalized']-external_gen
        net=mech['body_wrenches']-ext;cuts=accumulate(m.body_parentid,net)
        B,_=rig.coordinates(q,np.zeros(rig.n));reduced=B.T@generalized
        moments=[];powers=[];reactions=[]
        for name in selected:
            j=b.joints[name];col=6+b.independent.index(j);moment=float(reduced[col]);moments.append(moment)
            powers.append(moment*v[m.jnt_dofadr[j]])
            reactions.append(at_point(cuts[m.jnt_bodyid[j]],d.xanchor[j])[:3])
        body_v=m.body_mass@mech['velocity']/mass_kg;body_com=m.body_mass@d.xipos/mass_kg
        sys_com=(mass_kg*body_com+bar_mass*bar)/(mass_kg+bar_mass);sys_v=(mass_kg*body_v+bar_mass*bv)/(mass_kg+bar_mass)
        F=ground.sum(0);cop=np.r_[-required[4]/F[2],required[3]/F[2],0.]
        kinetic=.5*np.sum(m.body_mass[:,None]*mech['velocity']**2)+.5*np.einsum('bi,bij,bj->',mech['omega'],mech['inertia_world'],mech['omega'])
        E=kinetic-np.sum(m.body_mass[:,None]*d.xipos*m.opt.gravity)+.5*bar_mass*(bv@bv)+.5*omega@Ibar@omega-bar_mass*(m.opt.gravity@bar)
        rate=np.sum(mech['body_wrenches'][:,:3]*mech['velocity'])
        for bid in range(m.nbody):rate+=(mech['body_wrenches'][bid,3:]-np.cross(d.xipos[bid],mech['body_wrenches'][bid,:3]))@mech['omega'][bid]
        rate+=Fbar@bv+Tbar@omega
        errs=[];speeds=[]
        for bid,local,p0,_ in path.points:
            J,p=b.jacobian(bid,local);errs.append(np.linalg.norm(p-p0));speeds.append(np.linalg.norm(J@v))
        grasp_errors=[]
        for side in ['r','l']:
            local=np.array(path.arm_fit['grasp_centres_in_lunate_frame_m'][side])*b.report['global_normalization_scale']
            _,p=b.jacobian(m.body('lunate_'+side).id,local)
            grasp_errors.append(np.linalg.norm(p-(bar+(-.5 if side=='r' else .5)*path.arm_fit['grip_width_m']*bar_R[:,0])))
        vals=dict(times=t,qpos=q,qvel=v,qacc=a,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=body_com,velocity=body_v,system_COM_m=sys_com,system_velocity_m_s=sys_v,GRF_N=F,COP_m=cop,bar_position_m=bar,bar_rotation=bar_R,bar_velocity_m_s=bv,bar_acceleration_m_s2=ba,bar_support_force_N=Fbar,contact_positions_m=np.array([p[2] for p in path.points]),contact_forces_N=ground,net_joint_moments_Nm=moments,net_joint_powers_W=powers,net_joint_reactions_N=reactions,COM_power_W=F[2]*sys_v[2],bar_power_W=Fbar@bv+Tbar@omega,segment_energy_J=E,segment_energy_rate_W=rate,internal_power_W=generalized@v,balance_error_SI=max(err,abs(cuts[0]).max()),engine_error_SI=mech['crosscheck_max'],foot_error_m=max(errs),foot_speed_m_s=max(speeds))
        vals['grasp_error_m']=max(grasp_errors)
        for k,z in vals.items():arrays[k].append(z)
        if i%125==0:print('SQUAT_DYNAMICS',bar_mass,round(float(t),3),'GRF',round(F[2]),flush=True)
    arrays={k:np.asarray(v) for k,v in arrays.items()};np.savez_compressed(folder/'motion.npz',**arrays)
    b.pose(b.q0);np.savez_compressed(folder/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
    ascent_mask=(times>=up)&(times<=end);pp=arrays['COM_power_W'][ascent_mask];pt=times[ascent_mask]
    rate_work=np.trapezoid(arrays['segment_energy_rate_W'][ascent_mask],pt)
    delta=arrays['segment_energy_J'][ascent_mask][-1]-arrays['segment_energy_J'][ascent_mask][0]
    power_identity=float(max(abs(arrays['segment_energy_rate_W']-arrays['internal_power_W'])))
    report=dict(status='mechanically_checked_prescribed_squat',bar_mass_kg=bar_mass,body_mass_kg=mass_kg,height_m=height_m,physics_hz=hz,scale=b.report['global_normalization_scale'],
                phase_times_s=dict(start_descent=start,bottom=bottom,start_ascent=up,end_ascent=end,end=T),
                prescribed_timing_s=dict(descent=descent,bottom_hold=hold,ascent=ascent),
                squat_style='high_bar',depth=dict(path_fraction=peak_depth,bottom_knee_flexion_deg=float(np.degrees(.12+1.93*peak_depth)),
                    definition='Anatomical knee flexion coordinate, not thigh parallel or hip-crease depth.',
                    independently_predicted_strategy=False),
                balance_assumption='Prescribed coupled hip/knee path; torso and root fit planted feet and horizontal combined body-plus-bar COM at support centre. COP is calculated dynamically.',
                cadence_is_predicted=False,human_movement_validated=False,muscle_forces_computed=False,
                bar_support='Upper-back torso, with stabilizing hands. Full rod translational and rotational inertia.',
                footwear=dict(heel_to_toe_elevation_m=.025,shoe_mass_included=False,toe_out_deg=15,stance_site_width_m=.34),
                arm_fit=path.arm_fit,coordinates=selected,body_names=[m.body(i).name for i in range(m.nbody)],
                peak_COM_vertical_power_W=float(pp.max()),mean_ascent_COM_vertical_power_W=float(np.trapezoid(pp,pt)/ascent),
                peak_bar_power_W=float(arrays['bar_power_W'][ascent_mask].max()),peak_segment_energy_rate_W=float(arrays['segment_energy_rate_W'][ascent_mask].max()),
                peak_GRF_N=float(arrays['GRF_N'][:,2].max()),bar_travel_m=float(arrays['bar_position_m'][ascent_mask][-1,2]-arrays['bar_position_m'][ascent_mask][0,2]),
                peak_net_joint_moments_Nm=np.max(abs(arrays['net_joint_moments_Nm']),axis=0).tolist(),
                checks=dict(root_wrench_max_component_SI=float(arrays['balance_error_SI'].max()),independent_NE_vs_engine_max_SI=float(arrays['engine_error_SI'].max()),
                            foot_point_max_error_m=float(arrays['foot_error_m'].max()),foot_point_max_speed_m_s=float(arrays['foot_speed_m_s'].max()),
                            grasp_frame_max_error_m=float(arrays['grasp_error_m'].max()),
                            joint_power_vs_energy_rate_max_error_W=power_identity,ascent_work_energy_error_J=float(rate_work-delta),
                            minimum_contact_normal_N=float(arrays['contact_forces_N'][:,:,2].min()),maximum_friction_ratio=float(np.max(np.linalg.norm(arrays['contact_forces_N'][:,:,:2],axis=2)/np.maximum(arrays['contact_forces_N'][:,:,2],1e-10)))),
                contact_force_cap_used=False,strength_cap_used=False,root_reserve_applied=False)
    checks=report['checks'];gates=dict(root_balance=checks['root_wrench_max_component_SI']<1e-4,engine_crosscheck=checks['independent_NE_vs_engine_max_SI']<1e-4,
         planted_feet=checks['foot_point_max_error_m']<.001,contact_normal=checks['minimum_contact_normal_N']>-.001,friction=checks['maximum_friction_ratio']<=.8,
         work_energy=abs(checks['ascent_work_energy_error_J'])<1.,power_identity=power_identity<1.,grasp=checks['grasp_frame_max_error_m']<.0001)
    report['gates']=gates;report['all_mechanics_gates_passed']=all(gates.values())
    (folder/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if not all(gates.values()):raise RuntimeError(('Squat checks failed',gates))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bar-mass',type=float,default=100);ap.add_argument('--hz',type=int,default=250)
    ap.add_argument('--descent',type=float,default=1.25);ap.add_argument('--ascent',type=float);ap.add_argument('--hold',type=float,default=.15)
    ap.add_argument('--mass-kg',type=float,default=88.);ap.add_argument('--height-m',type=float,default=1.8);ap.add_argument('--label')
    ap.add_argument('--bottom-knee-deg',type=float,help='Bottom knee flexion within the existing high-bar path (above 6.876, at most 117.456 degrees).')
    a=ap.parse_args()
    build_trial(a.bar_mass,a.hz,a.descent,a.ascent,a.hold,a.mass_kg,a.height_m,a.label,a.bottom_knee_deg)
