"""Contact-driven common-push-off box-jump models, excluding impact.

Box height changes flight targets, not the shared ground controller. No
external support or box force acts during flight. All results are conditional
model scenarios; height alone is never converted to output or squat mass.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from build_contact_jump import ContactJump
from centroidal_jump import Polynomial,s5
from box_flight_curve import BoxFlightCurve
from retarget_measured_jump import MeasuredJumpRig
from mechanics_engine_adapter import sample
from newton_euler_reference import accumulate,at_point

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'


class BoxJump(ContactJump):
    def __init__(self,rig,hz,height):
        super().__init__(rig,hz);self.box_height=height
        self.bottom=.85;self.toe=1.07;self.takeoff=1.25
        self.requested_launch_vz_m_s=3.2;self.crouch_drop_m=.28;self.flight_point_weight=1000000.
        self.soft_ground_momentum_target=True
        self.ground_momentum_weight=10000.
        self.adaptive_toe_release=True;self.released_toe_points=set();self.toe_release_events=[]
        self.flight_orientation_weight=100000.;self.flight_knee_half_width_m=.12
        self.flight_pelvis_orientation_weight=1000.
        self.posture_weight_by_joint={'flex_extension':100.}
        self.flight_posture_weight_by_joint={'hip_flexion_r':20.,'hip_flexion_l':20.,'mtp_angle_r':100.,'mtp_angle_l':100.}
        self.air[self.m.jnt_qposadr[self.b.joints['flex_extension']]]=0.
        for side in ['r','l']:
            for name,value in [('hip_flexion',1.9),('knee_angle',2.05),('mtp_angle',0.)]:
                self.air[self.m.jnt_qposadr[self.b.joints[name+'_'+side]]]=value
        self.box_rear_y=-.18;self.box_front_y=-.83
        self.forward_T=self.toe-.25
        toe_y=float(np.mean([p[2][1] for p in self.p['points'] if p[3]]))
        # Forward propulsion needs the COM to advance over the supporting
        # forefoot. Holding it behind the COP creates a backward flight spin.
        self.forward=Polynomial(self.c0[1],0,0,toe_y-.10,-.35,0,self.forward_T)
        for side in ['r','l']:
            j=self.b.joints['shoulder_elv_'+side];self.m.jnt_range[j,1]=min(self.m.jnt_range[j,1],1.86)

    def horizontal_target(self,t):
        ss=np.clip(t-.25,0,self.forward_T)
        cy,vy,ay=[float(self.forward.value(ss,k)[0]) for k in range(3)]
        if t>self.toe:cy+=vy*(t-self.toe)
        return cy,vy,ay

    def balanced_horizontal_acceleration(self,t,vertical_acceleration):
        # A control goal, not an applied force or prescribed actual COP.
        # Choose COM acceleration so the desired support resultant passes
        # through the COM, while the QP still derives feasible contact forces.
        rear_support=.14;toe_support=min(p[2][1] for p in self.p['points'])
        nominal_cop=rear_support+(toe_support-rear_support)*s5((t-self.bottom)/(self.toe-self.bottom))
        h=float(self.d.subtree_com[1,2]-self.p['ground'])
        return (vertical_acceleration+9.80665)*(float(self.d.subtree_com[1,1])-nominal_cop)/h

    def clearance_profile(self,y):
        # Collision is audited against the actual box, rather than treating
        # an anticipatory ramp in empty air as an impassable surface.
        return self.p['ground'],0.,0.

    def configure_flight(self,q,u,qd,qdd):
        vz=float(self.d.subtree_linvel[1,2]);C=float(self.d.subtree_com[1,2]);landing_above_box=getattr(self,'landing_above_box_m',.60)
        print('BOX_LAUNCH_STATE',self.actual_takeoff,vz,C,flush=True)
        print('BOX_LAUNCH_ANGLES',[(n,float(q[self.m.jnt_qposadr[self.b.joints[n+'_r']]])) for n in ['hip_flexion','knee_angle','ankle_angle','mtp_angle']],flush=True)
        self.desired_approach_COM_height=self.box_height+landing_above_box
        disc=vz*vz-2*9.80665*(self.desired_approach_COM_height-C)
        if disc<=0:raise RuntimeError(('Insufficient launch for approach configuration',self.box_height,vz,C))
        self.flight_T=float((vz+np.sqrt(disc))/9.80665);self.air_transition_T=max(.2,self.flight_T)
        if getattr(self,'range_based_box_placement',False):
            end_y=float(self.d.subtree_com[1,1]+self.d.subtree_linvel[1,1]*self.flight_T)
            initial_toe_y=min(p[2][1] for p in self.p['points'])
            self.box_rear_y=min(initial_toe_y-.04,end_y+.09);self.box_front_y=self.box_rear_y-.65
        endpoint=self.air.copy();endpoint[:7]=q[:7]
        self.air_poly=Polynomial(q,qd,0,endpoint,0,0,self.air_transition_T)
        self.clearance_height=0.;self.flight_feet=[]
        self.flight_knees=[]
        for side in ['r','l']:
            jid=self.b.joints['knee_angle_'+side];bid=self.m.jnt_bodyid[jid]
            local=self.d.xmat[bid].reshape(3,3).T@(self.d.xanchor[jid]-self.d.xpos[bid])
            J,p=self.b.jacobian(bid,local)
            relative=p-self.d.subtree_com[1]
            relative_velocity=J@self.d.qvel-self.d.subtree_linvel[1]
            end=np.array([-.12 if side=='r' else .12,-.20,-.15])
            self.flight_knees.append((bid,local,Polynomial(relative,relative_velocity,0,end,0,0,self.flight_T*.75)))
        for side,foot in self.p['feet'].items():
            bid=foot['bid'];local=foot['local']
            J,p=self.b.jacobian(bid,local);end=foot['p'].copy();end[0]=self.c0[0]+(-.17 if side=='r' else .17)
            end[1]=self.d.subtree_com[1,1]+self.d.subtree_linvel[1,1]*self.flight_T+.03;end[2]+=self.box_height+.03
            self.flight_feet.append((bid,local,BoxFlightCurve(p,J@self.d.qvel,end,self.flight_T,self.box_rear_y),foot['R'].copy()))


def build(height_cm,hz,audit_existing=False,reuse_ground=False,controller_type=BoxJump,output_suffix='model'):
    rig=MeasuredJumpRig();motion=controller_type(rig,hz,height_cm/100);folder=OUT/f'box_{height_cm}_{output_suffix}';folder.mkdir(parents=True,exist_ok=True)
    try:
        if audit_existing or reuse_ground:
            data=dict(np.load((OUT/f'box_60_{output_suffix}' if reuse_ground else folder)/'motion_raw.npz'))
            k=np.flatnonzero(data['mode']==2)[0];motion.actual_takeoff=float(data['times'][k])
            if abs(float(np.median(np.diff(data['times'][:k])))-1/hz)>1e-8:
                raise RuntimeError('Cached ground sampling rate differs; run a fresh full trial.')
            rig.set_state(data['qpos'][k],data['u'][k]);qd=np.zeros(rig.m.nq);qdd=qd.copy()
            qd[rig.ids]=data['qvel'][k,rig.dofs];qdd[rig.ids]=data['qacc'][k,rig.dofs]
            motion.configure_flight(data['qpos'][k],data['u'][k],qd,qdd)
            if reuse_ground:
                motion.resume_data=data;data=motion.run()
        else:data=motion.run()
    except RuntimeError as e:
        if hasattr(motion,'trace'):np.savez_compressed(folder/'partial_rejected_motion.npz',**{k:np.asarray(v) for k,v in motion.trace.items()})
        (folder/'failure.json').write_text(json.dumps(dict(status='failed_not_released',reason=str(e)),indent=2));raise
    np.savez_compressed(folder/'motion_raw.npz',**data)
    m,d,b=rig.m,rig.d,rig.b;names=[stem+'_'+side for stem in ['hip_flexion','knee_angle','ankle_angle'] for side in ['r','l']]
    saved={k:[] for k in ['root_force_error_N','root_moment_error_Nm','net_joint_moments_Nm','net_joint_powers_W','net_joint_reactions_N','engine_error_SI','box_foot_gap_m','thorax_rotation_deg','heel_position_m','heel_target_error_m','heel_orientation_error_deg']}
    b.pose(b.q0);R0=d.xmat[m.body('torso').id].reshape(3,3).copy()
    # Audit every fourth sample independently; output/contacts retain full Hz.
    indices=np.unique(np.r_[np.arange(0,len(data['times']),4),len(data['times'])-1]);audit_times=[]
    for n,k in enumerate(indices):
        contacts=[(bid,data['contact_positions_m'][k,j],data['contact_forces_N'][k,j],np.zeros(3)) for j,(bid,_,_,_) in enumerate(motion.p['points'])]
        z=sample(rig,data['qpos'][k],data['qvel'][k],data['qacc'][k],contacts);B,_=rig.coordinates(data['qpos'][k],data['u'][k]);reduced=B.T@z['generalized'];mom=[];powers=[];forces=[]
        for name in names:
            jid=b.joints[name];col=6+b.independent.index(jid);value=reduced[col];mom.append(value);powers.append(value*data['qvel'][k,m.jnt_dofadr[jid]])
            forces.append(at_point(z['subtree_wrenches'][m.jnt_bodyid[jid]],d.xanchor[jid])[:3])
        residual=z['net_wrenches'].sum(0);C=data['com'][k];moment=residual[3:]-np.cross(C,residual[:3])
        points=data['contact_positions_m'][k];inside=(points[:,1]<=motion.box_rear_y)&(points[:,1]>=motion.box_front_y)
        gap=float(np.min(points[inside,2]-motion.box_height)) if inside.any() else float('nan')
        heels=[];heel_errors=[];angle_errors=[]
        ss=max(0,float(data['times'][k]-motion.actual_takeoff));ratio=np.clip(ss/motion.flight_T,0,1)
        for bid,local,poly,Rfoot in motion.flight_feet:
            p=b.jacobian(bid,local)[1];target=poly.value(ss);target[2]+=motion.clearance_height*16*ratio**2*(1-ratio)**2
            heels.append(p);heel_errors.append(np.linalg.norm(p-target))
            angle_errors.append(np.rad2deg(Rotation.from_matrix(Rfoot@d.xmat[bid].reshape(3,3).T).magnitude()))
        values=dict(root_force_error_N=residual[:3],root_moment_error_Nm=moment,net_joint_moments_Nm=mom,net_joint_powers_W=powers,net_joint_reactions_N=forces,engine_error_SI=z['crosscheck_max'],box_foot_gap_m=gap,thorax_rotation_deg=float(np.rad2deg(Rotation.from_matrix(d.xmat[m.body('torso').id].reshape(3,3)@R0.T).magnitude())),heel_position_m=heels,heel_target_error_m=heel_errors,heel_orientation_error_deg=angle_errors)
        for key,value in values.items():saved[key].append(value)
        audit_times.append(data['times'][k])
        if n%100==0:print('BOX_AUDIT',height_cm,n,len(indices),flush=True)
    saved={k:np.asarray(v) for k,v in saved.items()};saved['audit_times']=np.asarray(audit_times)
    power=data['force_world_N'][:,2]*data['velocity'][:,2]
    before_to=np.flatnonzero(data['times']<=motion.actual_takeoff)
    actual_bottom=float(data['times'][before_to[np.argmin(data['com'][before_to,2])]])
    phase=(data['times']>=actual_bottom)&(data['times']<=motion.actual_takeoff)
    ts=data['times'][phase];pp=power[phase];flight=data['times']>=motion.actual_takeoff
    foot_inside=(data['contact_positions_m'][:,:,1]<=motion.box_rear_y)&(data['contact_positions_m'][:,:,1]>=motion.box_front_y)
    gaps=np.where(foot_inside,data['contact_positions_m'][:,:,2]-motion.box_height,np.inf)
    min_gap=float(gaps.min()) if foot_inside.any() else None;duration=float(data['times'][-1]-motion.actual_takeoff)
    launch=np.flatnonzero(flight)[0];ft=data['times'][flight]-data['times'][launch]
    ballistic=data['com'][launch]+ft[:,None]*data['velocity'][launch]+ft[:,None]**2*np.array([0,0,-9.80665/2])
    E=88*(9.80665*data['com'][:,2]+.5*data['velocity'][:,2]**2)
    work=float(np.trapezoid(pp,ts));dE=float(E[np.flatnonzero(phase)[-1]]-E[np.flatnonzero(phase)[0]])
    lastfeet=data['contact_positions_m'][-1]
    checks=dict(root_force_max_N=float(abs(saved['root_force_error_N']).max()),root_moment_max_Nm=float(abs(saved['root_moment_error_Nm']).max()),independent_NE_vs_engine_max_SI=float(saved['engine_error_SI'].max()),
                flight_ground_force_max_N=float(np.linalg.norm(data['force_world_N'][flight],axis=1).max()),minimum_foot_box_gap_m=min_gap,
                max_thorax_rotation_from_neutral_deg=float(saved['thorax_rotation_deg'].max()),flight_duration_saved_s=duration,flight_duration_requested_s=motion.flight_T)
    checks['approach_COM_y_m']=float(data['com'][-1,1])
    checks.update(final_heel_tracking_error_m=float(saved['heel_target_error_m'][-1].max()),
                  final_heel_orientation_error_deg=float(saved['heel_orientation_error_deg'][-1].max()),
                  final_heel_y_m=saved['heel_position_m'][-1,:,1].tolist(),
                  final_all_foot_points_y_m_bounds=[float(lastfeet[:,1].min()),float(lastfeet[:,1].max())],
                  final_all_foot_points_z_m_bounds=[float(lastfeet[:,2].min()),float(lastfeet[:,2].max())],
                  final_right_to_left_heel_separation_m=float(saved['heel_position_m'][-1,1,0]-saved['heel_position_m'][-1,0,0]),
                  ballistic_COM_max_error_m=float(abs(data['com'][flight]-ballistic).max()),
                  flight_angular_momentum_change_Nms=float(np.linalg.norm(data['angular_momentum'][flight]-data['angular_momentum'][launch],axis=1).max()),
                  vertical_COM_work_J=work,vertical_COM_energy_change_J=dE,vertical_work_energy_relative_error=abs(work-dE)/max(1,abs(dE)))
    gates=dict(root_balance=max(checks['root_force_max_N'],checks['root_moment_max_Nm'])<1e-4,engine_agreement=checks['independent_NE_vs_engine_max_SI']<1e-5,
               no_flight_support=checks['flight_ground_force_max_N']<1e-6,box_clearance=min_gap is not None and min_gap>-.002,complete_approach=duration>=motion.flight_T-2/hz,
               upright_motion=checks['max_thorax_rotation_from_neutral_deg']<50,
               body_reaches_box=motion.box_front_y<checks['approach_COM_y_m']<motion.box_rear_y,
               feet_reach_top=lastfeet[:,1].max()<motion.box_rear_y-.01 and lastfeet[:,1].min()>motion.box_front_y+.01 and lastfeet[:,2].min()>motion.box_height-.002 and lastfeet[:,2].max()<motion.box_height+.15,
               foot_tracking=checks['final_heel_tracking_error_m']<.04 and checks['final_heel_orientation_error_deg']<20,
               feet_not_crossed=checks['final_right_to_left_heel_separation_m']>.1,
               ballistic_flight=checks['ballistic_COM_max_error_m']<.005 and checks['flight_angular_momentum_change_Nms']<.05,
               vertical_work_energy=checks['vertical_work_energy_relative_error']<.01)
    gates={key:bool(value) for key,value in gates.items()}
    data.update(COM_power_W=power,GRF_N=data['force_world_N'],system_COM_m=data['com'],system_velocity_m_s=data['velocity'],**saved)
    np.savez_compressed(folder/'motion.npz',**data);b.pose(b.q0);np.savez_compressed(folder/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
    report=dict(status='checked_common_push_off_box_model' if all(gates.values()) else 'failed_not_released',movement='box_jump_model',box_height_m=height_cm/100,body_mass_kg=88,height_m=1.8,physics_hz=hz,scale=b.report['global_normalization_scale'],
                body_names=[m.body(i).name for i in range(m.nbody)],coordinates=names,
                phase_times_s=dict(start_descent=getattr(motion,'start_descent',.25),bottom=actual_bottom,start_ascent=actual_bottom,end_ascent=motion.actual_takeoff,end=float(data['times'][-1])),
                box_geometry=dict(x_bounds_m=[-.4,.4],y_bounds_m=[motion.box_front_y,motion.box_rear_y],top_height_m=height_cm/100),
                peak_COM_vertical_power_W=float(pp.max()),mean_ascent_COM_vertical_power_W=float(np.trapezoid(pp,ts)/(motion.actual_takeoff-actual_bottom)),
                peak_GRF_N=float(data['force_world_N'][:,2].max()),takeoff_velocity_m_s=data['velocity'][np.argmin(abs(data['times']-motion.actual_takeoff))].tolist(),
                prescribed_timing_s=dict(ascent=motion.actual_takeoff-actual_bottom),controller_bottom_target_s=motion.bottom,common_ground_controller_for_all_boxes=True,
                toe_release_events=motion.toe_release_events,ground_angular_momentum_control='soft posture stabilization' if motion.soft_ground_momentum_target else 'zero centroidal angular momentum; delivered by solved contacts',
                controller_profile=getattr(motion,'profile','synthetic posture goals'),takeoff_release=getattr(motion,'flight_release_diagnostic',{'mode':'zero normal force'}),
                input_requested_launch_velocity_m_s=motion.requested_launch_vz_m_s,height_is_not_used_to_set_takeoff_power=True,human_motion_validated=False,muscle_forces_computed=False,
                landing_included=False,scope='Ground push-off and free-flight box approach. Ends before impact; box forces are zero.',checks=checks,gates=gates,all_mechanics_gates_passed=all(gates.values()))
    (folder/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    if not all(gates.values()):
        (folder/'failure.json').write_text(json.dumps(dict(status='failed_not_released',reason='Continuous motion/clearance gates failed',failed_checks=[key for key,value in gates.items() if not value]),indent=2))
        raise RuntimeError(('Box model not released',gates))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--height-cm',type=int,choices=[40,60,80],default=60);ap.add_argument('--hz',type=int,default=600);ap.add_argument('--audit-existing',action='store_true');ap.add_argument('--reuse-ground',action='store_true');a=ap.parse_args();build(a.height_cm,a.hz,a.audit_existing,a.reuse_ground)
