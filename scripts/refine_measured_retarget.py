"""Kinematic COM/foot registration, without fitting any force amplitudes.

The source model's estimated COM is a kinematic retarget target, not an
independent COM measurement. Whole-body force agreement is consequently
inherited from that reference; it is not a new validation of mass fractions.
"""
import json,shutil,argparse
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from scipy.signal import butter,sosfiltfilt
from retarget_measured_jump import ROOT,C,MeasuredJumpRig,verify_existing
from mechanics_engine_adapter import sample

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['com_only','com_and_feet'],default='com_only');ap.add_argument('--folder');args=ap.parse_args()
    trial='s10_CMJ_t1';dest=ROOT/'results/measured_jump_analysis'/trial;setup=json.loads((dest/'setup.json').read_text());backup=dest/'rotation_retarget.npz'
    if args.folder:
        dest=ROOT/args.folder;setup=json.loads((dest/'setup.json').read_text());backup=dest/'rotation_retarget.npz'
    if not backup.exists():shutil.copyfile(dest/'motion.npz',backup)
    original=np.load(backup);ref=np.load(ROOT/'results/measured_cmj'/trial/'mechanics_replay.npz');raw=np.load(ROOT/'results/measured_cmj'/f'{trial}_measured.npz');ts=original['times'];absolute=ts+setup['source_start_s'];source_ids=np.array([np.argmin(abs(ref['times']-tt)) for tt in absolute]);snames=list(ref['body_names']);rig=MeasuredJumpRig();b,m,d=rig.b,rig.m,rig.d;mass=m.body_mass.sum();rootid=int(m.jnt_bodyid[0]);com_target=ref['COM_m'][source_ids]@C.T;com_spline=CubicSpline(ts,com_target);rootR=Rotation.from_quat(np.roll(original['qpos'][:,3:7],-1,axis=1));rot=RotationSpline(ts,rootR)
    chosen=[i for i,n in enumerate(b.ind_names) if n in ['flex_extension','lat_bending','axial_rotation'] or n.startswith(('hip_','knee_','ankle_','subtalar_'))];qadr=np.array(rig.ids)[chosen];dof=np.array(rig.dofs)[chosen];jids=np.array(b.independent)[chosen];lo=m.jnt_range[jids,0]+1e-7;hi=m.jnt_range[jids,1]-1e-7
    b.pose(original['qpos'][0]);feet=[];desired=[]
    for j,side in enumerate(raw['plate_sides']):
        n='calcn_'+str(side);bid=m.body(n).id;si=snames.index(n);p=np.nan_to_num(raw['COP_m'][source_ids[0],j])@C.T;local=d.xmat[bid].reshape(3,3).T@(p-d.xpos[bid]);source_local=ref['body_rotations'][source_ids[0],si].T@(np.nan_to_num(raw['COP_m'][source_ids[0],j])-ref['body_origins_m'][source_ids[0],si]);path=(ref['body_origins_m'][source_ids,si]+np.einsum('tij,j->ti',ref['body_rotations'][source_ids,si],source_local))@C.T;feet.append((bid,local));desired.append(path)
    desired=np.stack(desired,axis=1);angles=[];fit_errors=[]
    if args.mode=='com_only':
        # Preserve the accepted bone rotations and the quiet-pose anatomy.
        # Transfer the reference COM displacement rather than forcing a new
        # absolute COM height onto a differently proportioned generic body.
        com_target += original['com'][0]-com_target[0]
        com_spline=CubicSpline(ts,com_target)
    for k,tt in enumerate(ts):
        if args.mode=='com_only':angles.append(original['qpos'][k,rig.ids]);continue
        seed=original['qpos'][k].copy();prior=seed[qadr].copy()
        def pose(x):
            q=seed.copy();q[qadr]=x;q[:3]=0;q=b.complete(q);d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);q[:3]=com_target[k]-m.body_mass@d.xipos/mass;d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);return q
        def residual(x):
            pose(x);err=np.concatenate([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local-desired[k,j] for j,(bid,local) in enumerate(feet)]);return np.r_[1000*err,.2*(x-prior)]
        def jacobian(x):
            q=pose(x);B,_=rig.coordinates(q,np.zeros(rig.n));columns=B[:,6+np.array(chosen)];d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);Jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,Jc,rootid);rows=[]
            for bid,local in feet:
                p=d.xpos[bid]+d.xmat[bid].reshape(3,3)@local;J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,p,bid);rows.append(1000*(J-Jc)@columns)
            return np.vstack([*rows,.2*np.eye(len(chosen))])
        fit=least_squares(residual,np.clip(prior,lo+1e-7,hi-1e-7),jac=jacobian,bounds=(lo,hi),max_nfev=55,ftol=1e-9,xtol=1e-9,gtol=1e-8);q=pose(fit.x);angles.append(q[rig.ids]);fit_errors.append(np.linalg.norm(residual(fit.x)[:6].reshape(2,3),axis=1)/1000)
        if k%25==0:print('COM_FOOT_RETARGET',k,len(ts),'foot_mm',np.round(np.array(fit_errors[-1])*1000,3).tolist(),flush=True)
    angles=np.array(angles)
    if args.mode!='com_only':angles=sosfiltfilt(butter(3,15,fs=125,output='sos'),angles,axis=0)
    spline=CubicSpline(ts,angles)
    def configuration(t):
        q=b.q0.copy();q[:3]=0;q[3:7]=np.roll(rot(t).as_quat(),1);q[rig.ids]=spline(t);q=b.complete(q);d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);q[:3]=com_spline(t)-m.body_mass@d.xipos/mass;return q
    keys=['times','qpos','qvel','qacc','u','udot','com','velocity','force_world_N','body_xpos','body_xmat','contact_forces_N','contact_positions_m','contact_free_moments_Nm','root_force_residual_N','root_moment_residual_Nm','crosscheck_error_SI','retarget_foot_point_error_m'];arrays={key:[] for key in keys};h=1e-4
    def velocity(t):
        v=np.zeros(m.nv);mujoco.mj_differentiatePos(m,v,2*h,configuration(t-h),configuration(t+h));return v
    for k,(tt,index) in enumerate(zip(ts,source_ids)):
        q=configuration(tt);vf=velocity(tt);af=(velocity(tt+h)-velocity(tt-h))/(2*h);u=np.r_[vf[:6],vf[rig.dofs]];udot=np.r_[af[:6],af[rig.dofs]];B,curve=rig.coordinates(q,u);v=B@u;a=B@udot+curve;F=raw['applied_ground_force_N'][index]@C.T;P=np.nan_to_num(raw['COP_m'][index])@C.T;T=raw['applied_free_moment_Nm'][index]@C.T;contacts=[(m.body('calcn_'+str(s)).id,P[j],F[j],T[j]) for j,s in enumerate(raw['plate_sides'])];mech=sample(rig,q,v,a,contacts);own=mech['body_wrenches'].sum(0);extF=F.sum(0);extM=np.cross(P,F).sum(0)+T.sum(0);com=m.body_mass@d.xipos/mass;resF=own[:3]-extF;resM=own[3:]-extM-np.cross(com,resF);errors=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local-desired[k,j] for j,(bid,local) in enumerate(feet)])
        vals=dict(times=tt,qpos=q,qvel=v,qacc=a,u=u,udot=udot,com=com,velocity=m.body_mass@mech['velocity']/mass,force_world_N=extF,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),contact_forces_N=F,contact_positions_m=P,contact_free_moments_Nm=T,root_force_residual_N=resF,root_moment_residual_Nm=resM,crosscheck_error_SI=mech['crosscheck_max'],retarget_foot_point_error_m=errors)
        for key,value in vals.items():arrays[key].append(value)
    arrays={k:np.array(v) for k,v in arrays.items()};np.savez_compressed(dest/'motion.npz',**arrays);setup.update(kinematic_retarget_targets='Reference-model estimated COM displacement, registered in the quiet pose. Bone-rotation fit is preserved; foot material-point mismatch is reported.' if args.mode=='com_only' else 'Reference-model estimated COM trajectory and registered material foot points.',COM_trajectory_independently_measured=False,force_agreement_inherited_from_reference_COM=True,floor_translation_m=[0,0,0]);pre=np.flatnonzero(ts<setup['takeoff_s']);setup['countermovement_end_s']=float(ts[pre[np.argmin(arrays['com'][pre,2])]]);(dest/'setup.json').write_text(json.dumps(setup,indent=2));(dest/'kinematic_retarget.json').write_text(json.dumps(dict(mode=args.mode,COM_target_RMS_error_m=float(np.sqrt(np.mean((arrays['com']-com_target)**2))),foot_point_RMS_error_m=float(np.sqrt(np.mean(arrays['retarget_foot_point_error_m']**2))),foot_point_max_error_m=float(np.linalg.norm(arrays['retarget_foot_point_error_m'],axis=2).max()),force_amplitude_fit=False,COM_source='Estimated from measured-marker reference model; not an independently measured trajectory.'),indent=2));verify_existing(trial,dest)

if __name__=='__main__':main()
