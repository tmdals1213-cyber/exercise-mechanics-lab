"""Replay a measured CMJ on the existing full-body muscle skeleton.

Bone-frame rotation registration uses the two models' zero poses. A continuous
joint-limited fit transfers motion, not forces or muscle recruitment. Measured
forces remain unscaled. Floating-root residuals are reported, never concealed.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco,opensim as osim
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.signal import butter,sosfiltfilt
from scipy.spatial.transform import Rotation,RotationSpline
from jump_model import ArticulatedJumpRig
from parametric_body import Subject
from rigid_baseline_profile import HELPERS
from mechanics_engine_adapter import sample

ROOT=Path(__file__).resolve().parents[1]
C=np.array([[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]])

class MeasuredJumpRig(ArticulatedJumpRig):
    def __init__(self,mass_kg=88.,height_m=1.8):
        super().__init__(subject=Subject(height_m=height_m))
        factor=mass_kg/self.m.body_mass.sum()
        self.m.body_mass[1:]*=factor;self.m.body_inertia[1:]*=factor
        if height_m!=1.8:
            # Near-degenerate native inertia tensors can trigger MuJoCo's
            # spherical balance-inertia fallback after similarity scaling.
            # Use the compiled 180cm reference consistently for all new heights.
            reference=MeasuredJumpRig(mass_kg=mass_kg,height_m=1.8)
            physical=[i for i in range(1,self.m.nbody)if self.m.body(i).name not in HELPERS]
            self.m.body_inertia[physical]=reference.m.body_inertia[physical]*(height_m/1.8)**2
            self.m.body_iquat[physical]=reference.m.body_iquat[physical]
            self.profile['inertia_similarity_reference_height_m']=1.8
        mujoco.mj_setConst(self.m,self.d);self.b.pose(self.b.q0)
        self.profile['subject_mass_kg']=float(self.m.body_mass.sum())
        if height_m!=1.8:self.profile['subject_height_m']=height_m
        self.profile['measured_replay_mass_policy']='Reported participant mass; reference segment fractions retained.'

def matrix(M):return np.array([[M.get(i,j) for j in range(3)] for i in range(3)])

def verify_existing(trial,destination=None):
    dest=Path(destination) if destination else ROOT/'results/measured_jump_analysis'/trial;setup=json.loads((dest/'setup.json').read_text());motion=dest/'motion.npz';a=np.load(motion);ref=np.load(ROOT/'results/measured_cmj'/trial/'mechanics_replay.npz');ts=a['times']+setup['source_start_s'];rep=next(x for x in json.loads((ROOT/'results/measured_cmj/summary.json').read_text())['trials'] if x['trial']==trial);active=(ts>=rep['movement_onset_s'])&(ts<rep['takeoff_s']-.03);rms=np.sqrt(np.mean(a['root_force_residual_N'][active]**2,axis=0));cross=float(a['crosscheck_error_SI'].max());assert cross<1e-5
    digest=hashlib.sha256(motion.read_bytes()).hexdigest();rig=MeasuredJumpRig();rig.b.pose(rig.b.q0);model=osim.Model(str(ROOT/'results/measured_cmj/reference_model/scaled_model.osim'));state=model.initSystem()
    for coord in model.getCoordinateSet():coord.setValue(state,0.,False)
    model.realizePosition(state);fits={};ids=np.array([np.argmin(abs(ref['times']-t)) for t in ts]);snames=list(ref['body_names']);nnames=setup['body_names']
    for n,s in [('chest_r','torso')]+[(n+'_'+side,n+'_'+side) for side in ['r','l'] for n in ['femur','tibia','calcn','humerus','ulna','radius']]:
        zero=matrix(model.getBodySet().get(s).getTransformInGround(state).R().asMat33());native=rig.d.xmat[rig.m.body(n).id].reshape(3,3);wanted=C@ref['body_rotations'][ids,snames.index(s)]@zero.T@C.T@native;actual=a['body_xmat'][:,nnames.index(n)].reshape(-1,3,3);err=Rotation.from_matrix(np.swapaxes(wanted,1,2)@actual).magnitude();fits[n]=float(np.rad2deg(np.sqrt(np.mean(err**2))))
    verification=dict(status='measured_retarget_diagnostic',motion_sha256=digest,gates=dict(independent_newton_euler=cross<1e-5,source_measured_forces=True,proper_world_rotation=bool(abs(np.linalg.det(C)-1)<1e-12)),checks=dict(independent_NE_max_error_SI=cross,active_vertical_force_RMS_N=float(rms[2]),active_vertical_force_RMS_bodyweights=float(rms[2]/(88*9.80665)),active_force_RMS_world_N=rms.tolist(),root_moment_RMS_Nm=np.sqrt(np.mean(a['root_moment_residual_Nm'][active]**2,axis=0)).tolist()),force_residual_applied_as_external_support=False,ten_percent_reference_agreement=bool(rms[2]/(88*9.80665)<=.10),fit_orientation_RMS_deg=fits,muscle_force_validated=False)
    (dest/'verification.json').write_text(json.dumps(verification,indent=2));(dest/'motion_quality.json').write_text(json.dumps(dict(status='passed',motion_sha256=digest,source='Measured upright CMJ, bilateral foot plates; no synthetic flight rotation',measured_replay=True),indent=2));print('RETARGET_COMPLETE',json.dumps(verification),flush=True)

    if setup.get('landing_s') is not None:
        quality=json.loads((dest/'motion_quality.json').read_text())
        quality.update(status='kinematic_review_only',force_analysis_allowed=False,landing_force_validated=False)
        (dest/'motion_quality.json').write_text(json.dumps(quality,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trial',default='s10_CMJ_t1');ap.add_argument('--hz',type=int,default=125);ap.add_argument('--check-existing',action='store_true');ap.add_argument('--include-landing',action='store_true');ap.add_argument('--reuse-full-cycle',action='store_true');args=ap.parse_args()
    if args.reuse_full_cycle and not args.include_landing:raise ValueError('--reuse-full-cycle requires --include-landing')
    if args.check_existing:
        verify_existing(args.trial,ROOT/'results/measured_jump_analysis'/(args.trial+'_full_cycle') if args.include_landing else None);return
    src=ROOT/'results/measured_cmj';dest=ROOT/'results/measured_jump_analysis'/args.trial;dest.mkdir(parents=True,exist_ok=True)
    cached_path=dest/'motion.npz';cached=np.load(cached_path) if args.include_landing and cached_path.exists() else None
    if args.include_landing:dest=ROOT/'results/measured_jump_analysis'/(args.trial+'_full_cycle');dest.mkdir(parents=True,exist_ok=True)
    full_cached=np.load(dest/'motion.npz') if args.reuse_full_cycle else None
    z=np.load(src/args.trial/'mechanics_replay.npz');raw=np.load(src/f'{args.trial}_measured.npz');rep=next(x for x in json.loads((src/'summary.json').read_text())['trials'] if x['trial']==args.trial)
    rig=MeasuredJumpRig();b,m,d=rig.b,rig.m,rig.d;b.pose(b.q0)
    model=osim.Model(str(src/'reference_model/scaled_model.osim'));state=model.initSystem()
    for coord in model.getCoordinateSet():coord.setValue(state,0.,False)
    model.realizePosition(state)
    pairs=[('torso','torso',1.),('chest_r','torso',.5)]
    for side in ['r','l']:
        pairs += [(n+'_'+side,n+'_'+side,1.) for n in ['femur','tibia','calcn','humerus','ulna','radius']]
    snames=list(z['body_names']);native={n:d.xmat[m.body(n).id].reshape(3,3).copy() for n,_,_ in pairs+ [('pelvis','pelvis',1.)]}
    align={}
    for n,s,_ in pairs+[('pelvis','pelvis',1.)]:
        ref=matrix(model.getBodySet().get(s).getTransformInGround(state).R().asMat33())
        align[n]=ref.T@C.T@native[n]
    first=rep['movement_onset_s']-.35;last=rep['landing_s']-.06
    if args.include_landing:last=min(float(z['times'][-1])-.05,rep['landing_s']+1.1)
    ids=np.where((z['times']>=first)&(z['times']<=last))[0][::max(1,int(round(250/args.hz)))];ts=z['times'][ids];root_index=snames.index('pelvis');rootR=C@z['body_rotations'][ids,root_index]@align['pelvis']@native['pelvis'].T;rootP=z['body_origins_m'][ids,root_index]@C.T
    angles=[];fits=[];seed=np.array([b.q0[i] for i in rig.ids]);previous=seed.copy()
    lo=np.array([m.jnt_range[j,0] for j in b.independent]);hi=np.array([m.jnt_range[j,1] for j in b.independent]);hi[[b.ind_names.index('shoulder_elv_'+s) for s in ['r','l']]]=1.86
    # Rotation-frame residuals have no length mismatch to spuriously stretch a
    # rigid skeleton. The native frame registration is fixed at the zero pose.
    for count,index in enumerate(ids):
        if full_cached is not None:
            relative=ts[count]-ts[0];near=int(np.argmin(abs(full_cached['times']-relative)));assert abs(full_cached['times'][near]-relative)<1e-6
            value=full_cached['qpos'][near,rig.ids].copy()
            if cached is not None and relative<=cached['times'][-1]:
                old=int(np.argmin(abs(cached['times']-relative)));assert abs(cached['times'][old]-relative)<1e-6
                blend=np.clip((relative-(cached['times'][-1]-.12))/.12,0.,1.);blend=blend**3*(10-15*blend+6*blend**2)
                value=(1-blend)*cached['qpos'][old,rig.ids]+blend*value
            angles.append(value);continue
        target={n:C@z['body_rotations'][index,snames.index(s)]@align[n] for n,s,_ in pairs}
        q=b.q0.copy();q[:3]=rootP[count];q[3:7]=np.roll(Rotation.from_matrix(rootR[count]).as_quat(),1)
        def pose(x):
            qq=q.copy();qq[rig.ids]=x;qq=b.complete(qq);d.qpos[:]=qq;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);return qq
        def residual(x):
            pose(x);out=[]
            for n,_,weight in pairs:out.extend(weight*Rotation.from_matrix(target[n].T@d.xmat[m.body(n).id].reshape(3,3)).as_rotvec())
            out.extend(.0005*(x-previous));return np.array(out)
        cached_index=None
        if cached is not None:
            near=int(np.argmin(abs(cached['times']-(ts[count]-ts[0]))))
            if abs(cached['times'][near]-(ts[count]-ts[0]))<1e-6:cached_index=near
        if cached_index is None:fit=least_squares(residual,np.clip(previous,lo+1e-7,hi-1e-7),bounds=(lo+1e-8,hi-1e-8),max_nfev=65,ftol=1e-9,xtol=1e-9,gtol=1e-8)
        else:
            from types import SimpleNamespace
            fit=SimpleNamespace(x=cached['qpos'][cached_index,rig.ids])
        previous=fit.x.copy();angles.append(previous);pose(previous);err={n:float(np.linalg.norm(Rotation.from_matrix(target[n].T@d.xmat[m.body(n).id].reshape(3,3)).as_rotvec())) for n,_,_ in pairs};fits.append(err)
        if count%20==0:print('RETARGET',count,len(ids),'max_deg',round(np.rad2deg(max(err.values())),2),flush=True)
    angles=np.array(angles);fs=1/np.median(np.diff(ts))
    if full_cached is None:
        raw_angles=angles.copy();angles=sosfiltfilt(butter(3,15,fs=fs,output='sos'),angles,axis=0)
        if cached is not None:
            # Cached coordinates were already filtered. Preserve their accepted
            # history and blend only the last 120 ms into the new extension.
            blend=np.clip((ts-ts[0]-(cached['times'][-1]-.12))/.12,0.,1.);blend=blend**3*(10-15*blend+6*blend**2)
            angles=(1-blend[:,None])*raw_angles+blend[:,None]*angles
    spl=CubicSpline(ts,angles);pos=CubicSpline(ts,rootP);rot=RotationSpline(ts,Rotation.from_matrix(rootR))
    def configuration(t):
        q=b.q0.copy();q[:3]=pos(t);q[3:7]=np.roll(rot(t).as_quat(),1);q[rig.ids]=spl(t);return b.complete(q)
    # Only a fixed quiet-pose translation aligns the two anatomical floor
    # references. This changes neither acceleration nor measured force size.
    q=configuration(ts[0]);b.pose(q);native_sole=np.mean([d.site_xpos[m.site('subject_sole_reference_'+s).id,2] for s in ['r','l']]);floor_shift=np.array([0.,0.,-native_sole])
    keys=['times','qpos','qvel','qacc','u','udot','com','velocity','force_world_N','body_xpos','body_xmat','contact_forces_N','contact_positions_m','contact_free_moments_Nm','root_force_residual_N','root_moment_residual_Nm','crosscheck_error_SI']
    arrays={key:[] for key in keys};h=1e-4
    for count,(tt,index) in enumerate(zip(ts,ids)):
        q=configuration(tt);q[:3]+=floor_shift
        def velocity(t):
            v=np.zeros(m.nv);mujoco.mj_differentiatePos(m,v,2*h,configuration(t-h),configuration(t+h));return v
        vf=velocity(tt);af=(velocity(tt+h)-velocity(tt-h))/(2*h);u=np.r_[vf[:6],vf[rig.dofs]];udot=np.r_[af[:6],af[rig.dofs]];B,curve=rig.coordinates(q,u);v=B@u;a=B@udot+curve
        F=raw['applied_ground_force_N'][index]@C.T;P=np.nan_to_num(raw['COP_m'][index])@C.T;T=raw['applied_free_moment_Nm'][index]@C.T
        contacts=[(m.body('calcn_'+str(s)).id,P[j],F[j],T[j]) for j,s in enumerate(raw['plate_sides'])];mech=sample(rig,q,v,a,contacts);own=mech['body_wrenches'].sum(0);external_F=F.sum(0);external_M=np.cross(P,F).sum(0)+T.sum(0);com=m.body_mass@d.xipos/m.body_mass.sum();resF=own[:3]-external_F;resM=own[3:]-external_M-np.cross(com,resF)
        vals=dict(times=tt-ts[0],qpos=q,qvel=v,qacc=a,u=u,udot=udot,com=com,velocity=m.body_mass@mech['velocity']/m.body_mass.sum(),force_world_N=external_F,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),contact_forces_N=F,contact_positions_m=P,contact_free_moments_Nm=T,root_force_residual_N=resF,root_moment_residual_Nm=resM,crosscheck_error_SI=mech['crosscheck_max'])
        for key,value in vals.items():arrays[key].append(value)
    arrays={k:np.array(v) for k,v in arrays.items()};motion=dest/'motion.npz';np.savez_compressed(motion,**arrays);np.savez_compressed(dest/'rotation_retarget.npz',**arrays);digest=hashlib.sha256(motion.read_bytes()).hexdigest();active=(ts>=rep['movement_onset_s'])&(ts<rep['takeoff_s']-.03);rms=np.sqrt(np.mean(arrays['root_force_residual_N'][active]**2,axis=0));cross=float(arrays['crosscheck_error_SI'].max());assert cross<1e-5
    setup=dict(profile=rig.profile,rig_type='measured_cmj_standard',physics_hz=fs,coordinate_names=b.coordinate_names,body_names=[m.body(i).name for i in range(m.nbody)],mass_kg=88.,height_m=1.8,trial=args.trial,source_start_s=float(ts[0]),takeoff_s=rep['takeoff_s']-ts[0],countermovement_end_s=float(arrays['times'][np.argmin(arrays['com'][:,2])]),heel_rise_start_s=rep['takeoff_s']-ts[0]-.12,scope='Measured CMJ: quiet lead-in, countermovement, propulsion and flight; excludes landing',ground_reaction_and_COP_are_inputs=True,human_motion_validated=False,anatomical_frame_registration='Two-model zero-pose rotations; joint-limited continuous fit',floor_translation_m=floor_shift.tolist(),root_residuals_are_disclosed_not_applied=True)
    (dest/'setup.json').write_text(json.dumps(setup,indent=2));(dest/'contact_definition.json').write_text(json.dumps(dict(points=[dict(body_id=m.body('calcn_'+str(s)).id,body='calcn_'+str(s),local_m=[0,0,0]) for s in raw['plate_sides']],measured_COP=True,free_moment_included=True),indent=2))
    if args.include_landing:
        pre=np.flatnonzero(arrays['times']<setup['takeoff_s']);setup['countermovement_end_s']=float(arrays['times'][pre[np.argmin(arrays['com'][pre,2])]])
        setup['scope']='Measured CMJ including landing and recovery; kinematic retarget with disclosed dynamic residuals.'
        setup['landing_s']=rep['landing_s']-ts[0]
        setup['landing_force_validated']=False
        (dest/'setup.json').write_text(json.dumps(setup,indent=2))
    verify_existing(args.trial,dest)

if __name__=='__main__':main()
