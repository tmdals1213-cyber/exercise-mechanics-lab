"""Prescribe vertical task, solve full-body reactions and coordination together.

Horizontal motion and whole-body angular momentum remain outputs. The simple
arm task is declared; no target video pixels, hidden root support or force cap.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from centroidal_jump import s5,Polynomial
from build_centroidal_support_task import Task,MeasuredJumpRig
from dynamic_ground_path import solve
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='hard_com_v1');ap.add_argument('--hz',type=int,default=500);ap.add_argument('--vz',type=float,default=2.5);ap.add_argument('--vy',type=float,default=-.45);ap.add_argument('--arm-scale',type=float,default=.5);ap.add_argument('--crouch-com-forward',type=float,default=.09);ap.add_argument('--prepare-horizontal-hard',action='store_true');ap.add_argument('--sagittal-balance-weight',type=float,default=25.)
    ap.add_argument('--push-duration',type=float,default=.28);ap.add_argument('--takeoff-rise',type=float,default=.06);ap.add_argument('--hard-angular-balance',action='store_true');ap.add_argument('--crouch-hip',type=float,default=1.5);ap.add_argument('--crouch-knee',type=float,default=1.3);ap.add_argument('--high-flex',action='store_true');ap.add_argument('--trunk-gain',type=float,default=100.);ap.add_argument('--trunk-feedforward',action='store_true');ap.add_argument('--elbow-end',type=float,default=.35);ap.add_argument('--minimum-launch-ratio',type=float,default=.9);ap.add_argument('--free-lumbar',action='store_true');ap.add_argument('--launch-pitch-momentum',type=float);ap.add_argument('--prescribe-hip',action='store_true');ap.add_argument('--free-horizontal',action='store_true');ap.add_argument('--momentum-weight',type=float,default=1e5);ap.add_argument('--com-weight',type=float,default=3000.);ap.add_argument('--end-lean',type=float,default=0.);args=ap.parse_args()
    folder=ROOT/'results/physical_jump_task'/args.label;folder.mkdir(parents=True,exist_ok=True)
    r=MeasuredJumpRig(87,1.78);b,m,d=r.b,r.m,r.d
    if args.high_flex:
        from high_flex_profile import apply
        (folder/'high_flexion_profile.json').write_text(json.dumps(apply(r),indent=2))
    low,_=r.planted_pose(args.crouch_hip,args.crouch_knee,args.crouch_com_forward,arms=(np.pi/2,.1,.2),trunk=0.);b.pose(low);low[2]-=np.mean([d.site_xpos[m.site('subject_sole_reference_'+s).id,2] for s in ['r','l']]);b.pose(low);C_low=d.subtree_com[1].copy()
    initial,_=r.planted_pose(.1,.15,.07,arms=(np.pi/2,.12,.15),trunk=0.);b.pose(initial);initial[2]-=np.mean([d.site_xpos[m.site('subject_sole_reference_'+s).id,2] for s in ['r','l']]);b.pose(initial);actual_drop=float(d.subtree_com[1,2]-C_low[2])
    task=Task(r,args.hz,args.vz,args.vy,drop_m=actual_drop,plan_momentum=False,push_duration=args.push_duration,takeoff_rise=args.takeoff_rise)
    task.yprep=Polynomial(task.C0[1],0,0,C_low[1],0,0,task.bottom-.2)
    end=task.start.copy()
    for side in ['r','l']:
        for n,v in [('hip_flexion',.2),('knee_angle',.3),('ankle_angle',-.3),('mtp_angle',-.3),('shoulder_elv',1.2),('elbow_flexion',args.elbow_end)]:end[m.jnt_qposadr[b.joints[n+'_'+side]]]=v
    R0=Rotation.from_quat(np.roll(task.start[3:7],-1));Rl=Rotation.from_quat(np.roll(low[3:7],-1));Re=R0*Rotation.from_rotvec([args.end_lean,0.,0.])
    def reference(t):
        if t<=.2:return task.start.copy()
        if t<=task.bottom:
            u=float(s5((t-.2)/(task.bottom-.2)));q=task.start+u*(low-task.start);R=R0*Rotation.from_rotvec(u*(R0.inv()*Rl).as_rotvec())
        else:
            u=float(s5((t-task.bottom)/(task.takeoff-task.bottom)));q=low+u*(end-low);R=Rl*Rotation.from_rotvec(u*(Rl.inv()*Re).as_rotvec())
        q[3:7]=np.roll(R.as_quat(),1);return b.complete(q)
    arm_names=[n for n in b.ind_names if n.startswith(('shoulder_elv_','elbow_flexion_'))]
    sampled=np.array([reference(t) for t in task.grid]);qs=CubicSpline(task.grid,sampled,axis=0)
    hip_paths={}
    def arm_motion(t):
        result={}
        for n in arm_names:
            adr=m.jnt_qposadr[b.joints[n]];start=task.start[adr]
            result[n]=(float(start+args.arm_scale*(qs(t)[adr]-start)),float(args.arm_scale*qs(t,1)[adr]),float(args.arm_scale*qs(t,2)[adr]))
        start_hip=task.bottom+.025
        if args.prescribe_hip and t>=start_hip:
            qa=[m.jnt_qposadr[b.joints['hip_flexion_'+s]] for s in ['r','l']];va=[m.jnt_dofadr[b.joints['hip_flexion_'+s]] for s in ['r','l']]
            if 'mean' not in hip_paths:hip_paths['mean']=Polynomial(float(np.mean(d.qpos[qa])),float(np.mean(d.qvel[va])),0.,.35,0.,0.,task.takeoff-start_hip)
            poly=hip_paths['mean'];tau=min(t-start_hip,task.takeoff-start_hip);result['mean_hip_flexion']=tuple(float(poly.value(tau,k)[0]) for k in range(3))
        return result
    movable=('hip_','knee_','ankle_','subtalar_','mtp_angle_','shoulder_elv_','elbow_flexion_')
    if args.free_lumbar:
        movable+=('flex_extension',)
        lumbar=b.joints['flex_extension'];m.jnt_range[lumbar]=[max(float(m.jnt_range[lumbar,0]),-.35),min(float(m.jnt_range[lumbar,1]),.35)]
    locked={n:float(task.start[r.ids[i]]) for i,n in enumerate(b.ind_names) if not n.startswith(movable)}
    for i,n in enumerate(b.ind_names):
        if n.startswith('hip_rotation_') and not args.high_flex:
            m.jnt_range[b.independent[i],0]=max(m.jnt_range[b.independent[i],0],task.start[r.ids[i]]-.22)
            m.jnt_range[b.independent[i],1]=min(m.jnt_range[b.independent[i],1],task.start[r.ids[i]]+.22)
    joint_weights={n:50. if n.startswith(('hip_flexion_','knee_angle_')) else 20. for n in b.ind_names if n.startswith(('hip_','knee_','ankle_','subtalar_','mtp_angle_'))}
    profile=dict(locked_joint_targets=locked,prescribed_joint_motion=arm_motion,posture_weight_by_joint=joint_weights,hard_com_tracking_axes=[0,1,2] if args.prepare_horizontal_hard else [2],hard_com_axes_after_start=[0,1,2],horizontal_impulse_goal_m_s=args.vy,minimum_launch_speed_m_s=.9*args.vz,adaptive_contacts=True,reactivate_closed_contacts=True,rigid_contact_frames=True,symmetric_vertical_reactions=False,ground_momentum_axis_weights=[args.sagittal_balance_weight,25.,25.],release_unloaded_heels=True,search_forefoot_contact_modes=True,com_tracking_weight=args.com_weight,ground_trunk_weight=1000.)
    if args.free_horizontal:
        profile['hard_com_axes_after_start']=[2]
        profile.pop('horizontal_impulse_goal_m_s')
    profile['soft_ground_momentum_target']=not args.hard_angular_balance
    profile['minimum_launch_speed_m_s']=args.minimum_launch_ratio*args.vz
    if args.high_flex:profile.update(release_zero_load_points=False,joint_range_barrier_rate=100.)
    profile.update(track_trunk_feedforward=args.trunk_feedforward,trunk_tracking_gain=args.trunk_gain)
    if args.launch_pitch_momentum is not None:profile.update(launch_angular_momentum=[args.launch_pitch_momentum,0.,0.],hard_momentum_start_s=task.bottom+.04)
    solve(r,reference,lambda t:task.com(t),lambda t:task.com(t,1),lambda t:task.com(t,2),task.points,task.bottom,task.takeoff,task.tip,task.takeoff,folder,hz=args.hz,task_profile=profile,balance_weight=args.momentum_weight)
    metadata=dict(parameters=vars(args),geometrically_derived_crouch_drop_m=actual_drop,vertical_COM_is_task_constraint=True,horizontal_preparation_is_soft=not args.prepare_horizontal_hard,horizontal_propulsion='Forward impulse replanned from actual bottom state within friction cone.',angular_momentum_is_output=True,arm_task='single controlled sagittal swing',force_clipping=False,strength_ceiling=False,root_actuator=False,force_analysis_allowed=False,human_motion_validated=False)
    (folder/'task_definition.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main()




