"""Merge the accepted support branch; audit actual derivatives and continuity."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from retarget_measured_jump import MeasuredJumpRig
from high_flex_profile import apply
from build_contact_jump import contact_geometry
from box_geometry_audit import segments,minimum_clearance
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--flight',default='results/athletic_box_jump/yun_160_flat_feet_v2');ap.add_argument('--support',default='supported_hold_v1');ap.add_argument('--label',default='yun_160_supported_cycle_v1');a=ap.parse_args();flight=ROOT/a.flight;support=flight/a.support;sr=json.loads((support/'landing_audit.json').read_text());fr=json.loads((flight/'resolved_audit.json').read_text())
    if not sr['completed_supported_landing']:raise RuntimeError('Support branch is rejected')
    # NPZ indexing decompresses the whole member on each access. Cache members
    # once: retaining row views from repeated loads otherwise grows as N^2.
    ground=dict(np.load(ROOT/fr['source_ground']/'dynamic_ground_integrated.npz'));air=dict(np.load(flight/'flight_motion.npz'));land=dict(np.load(support/'supported_motion.npz'));out=ROOT/'results/athletic_box_jump'/a.label;out.mkdir(exist_ok=True);r=MeasuredJumpRig(87,1.78);profile=apply(r);b,m,d=r.b,r.m,r.d;r.contact_stand_override=ground['qpos'][0];points=contact_geometry(r)['points'];to=fr['takeoff_s'];contact=sr['contact_time_s'];n=r.n
    sequence=[(0,i) for i,t in enumerate(ground['times']) if t<to-1e-8]+[(1,i) for i,t in enumerate(air['times']) if t<contact-1e-8]+[(2,i) for i in range(len(land['times']))]
    sources=[ground,air,land];splines=[None,CubicSpline(air['times'],air['u'],axis=0),CubicSpline(land['times'],land['u'],axis=0)]
    # The absorption/balance controller changes at 0.10 s. Velocity is
    # continuous but acceleration can jump: one global C2 spline invents an
    # acceleration overshoot. Use independent one-sided fits at this known
    # event, without changing or excluding any saved velocity samples.
    split=int(round(.10/float(land['times'][1]-land['times'][0])))
    support_left=CubicSpline(land['times'][:split+1],land['u'][:split+1],axis=0)
    support_right=CubicSpline(land['times'][split:],land['u'][split:],axis=0)
    keys=['times','qpos','qvel','qacc','u','body_xpos','body_xmat','com','velocity','angular_momentum','contact_positions_m','collision_segments_m','force_world_N','contact_forces_N','mode'];arr={k:[] for k in keys};residuals={0:[],1:[],2:[]};minclear=np.inf;ROM=0.;closed=0.;peak_torque=0.;front=fr['box_front_y_m']
    for frame_index,(phase,i) in enumerate(sequence):
        source=sources[phase];q=source['qpos'][i];u=source['u'][i];t=float(source['times'][i]);B,curve=r.set_state(q,u);ud=ground['udot'][i] if phase==0 else splines[phase](t,1);v=B@u;acc=B@ud+curve;forces=np.zeros((len(points),3)) if phase==1 else source['contact_forces_N'][i];external=np.zeros(m.nv);positions=[]
        if phase==2:ud=(support_left if i<split else support_right)(t,1);acc=B@ud+curve
        for (bid,local,*_),force in zip(points,forces):J,p=b.jacobian(bid,local);external+=J.T@force;positions.append(p)
        M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);net=M@acc+d.qfrc_bias-d.qfrc_passive-external;reduced=B.T@net;peak_torque=max(peak_torque,float(abs(reduced[6:]).max()))
        # Numerical differentiation near mode switches is excluded, separately
        # from exact solve residuals and the explicit continuity checks below.
        if phase==0 or (phase==1 and to+.012<t<contact-.012) or (phase==2 and contact+.012<t<land['times'][-1]-.012):residuals[phase].append(net[:6].copy())
        mujoco.mj_subtreeVel(m,d);caps,rad=segments(m,d);minclear=min(minclear,minimum_clearance(caps,rad,front,1.6));ROM=max(ROM,float(np.maximum(m.jnt_range[b.independent,0]-q[r.ids],q[r.ids]-m.jnt_range[b.independent,1]).max()));qc=b.complete(q);closed=max(closed,float(abs(qc-q).max()));r.set_state(q,u);mujoco.mj_subtreeVel(m,d)
        vals=dict(times=t,qpos=q,qvel=v,qacc=acc,u=u,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),contact_positions_m=np.array(positions),collision_segments_m=caps,force_world_N=forces.sum(0),contact_forces_N=forces,mode=phase)
        for key,value in vals.items():arr[key].append(value)
        if frame_index%500==0:print('FINALIZE_BOX',frame_index,len(sequence),round(t,3),flush=True)
    data={key:np.array(value) for key,value in arr.items()};data['collision_radii_m']=rad;np.savez_compressed(out/'motion.npz',**data);b.pose(b.q0);np.savez_compressed(out/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
    ki=int(abs(ground['times']-to).argmin());ci=sr['contact_index'];continuity=dict(takeoff_qpos_max=float(abs(ground['qpos'][ki]-air['qpos'][0]).max()),takeoff_reduced_velocity_max=float(abs(ground['u'][ki]-air['u'][0]).max()),contact_qpos_max=float(abs(air['qpos'][ci]-land['qpos'][0]).max()),contact_reduced_velocity_max=float(abs(air['u'][ci]-land['u'][0]).max()))
    last=data['times']>data['times'][-1]-.30;P=data['contact_positions_m'];F=data['contact_forces_N'];cop=np.einsum('nij,ni->nj',P,F[:,:,2])/np.maximum(F[:,:,2].sum(1),1e-10)[:,None];feet_top=bool((P[-1,:,1]<front).all() and (P[-1,:,1]>front-.8).all() and abs(P[-1,:,2]-1.6).max()<.005)
    late_angular_speed=float(np.linalg.norm(data['u'][last,3:6],axis=1).max());late_joint_speed=float(abs(data['u'][last,6:]).max());res={str(phase):dict(force_max_N=float(abs(np.array(rows)[:,:3]).max()),moment_max_Nm=float(abs(np.array(rows)[:,3:]).max())) for phase,rows in residuals.items()};airP=P[data['mode']==1];inside=(airP[:,:,1]<front)&(airP[:,:,1]>front-.8)&(abs(airP[:,:,0]+.025)<.5);penetration=float(np.maximum(1.6-airP[:,:,2],0)[inside].max(initial=0.));floorpenetration=max(0.,-float(airP[:10,:,2].min()))
    stable=bool(feet_top and sr['final_COM_speed_m_s']<.10 and late_angular_speed<.20 and late_joint_speed<.5)
    passed=bool(stable and minclear>=0 and penetration<.003 and floorpenetration<.002 and ROM<.002 and max(continuity.values())<1e-10)
    dynamic_pass=all(v['force_max_N']<.10*87*9.80665 and v['moment_max_Nm']<.10*87*9.80665*1.78 for v in res.values())
    report=dict(status='supported_160cm_motion_candidate' if passed else 'rejected_full_cycle_audit',full_cycle_motion_pass=passed,subject_height_m=1.78,subject_mass_kg=87.,box_height_m=1.6,box_front_y_m=front,takeoff_s=to,contact_s=contact,end_s=float(data['times'][-1]),takeoff_velocity_m_s=fr['takeoff_velocity_m_s'],takeoff_angular_momentum_kg_m2_s=fr['takeoff_H_kg_m2_s'],takeoff_state_from_saved_ground_step=True,deliberate_contact_release_before_last_source_step=bool(to<ground['times'][-1]-.001),release_has_no_pose_or_velocity_reset=True,continuity=continuity,flight_COM_ballistic_error_m=fr['ballistic_COM_error_m'],flight_H_error_kg_m2_s=fr['angular_momentum_error_kg_m2_s'],independent_root_balance_from_velocity_derivatives=res,all_phase_body_capsule_clearance_m=float(minclear),air_sole_box_penetration_m=penetration,initial_air_floor_penetration_m=floorpenetration,ROM_excess_rad=max(0.,ROM),coupler_pose_closure_error=closed,feet_on_top=feet_top,terminal_COM_speed_m_s=sr['final_COM_speed_m_s'],terminal_angular_speed_max_last_0p3s_rad_s=late_angular_speed,terminal_joint_speed_max_last_0p3s_rad_s=late_joint_speed,terminal_H_norm_kg_m2_s=float(np.linalg.norm(data['angular_momentum'][-1])),terminal_COM_minus_COP_m=(data['com'][-1,:2]-cop[-1,:2]).tolist(),peak_takeoff_GRF_N=float(np.linalg.norm(data['force_world_N'][data['mode']==0],axis=1).max()),peak_landing_GRF_N=sr['peak_landing_GRF_N'],support_audit=sr,root_actuator=False,strength_ceiling=False,force_clipping=False,human_motion_validated=False,athlete_personal_performance_identified=False,muscle_force_analysis_allowed=False,full_mesh_collision_validated=False,high_flexion_profile=profile,source_flight=str(flight.relative_to(ROOT)),source_support=str(support.relative_to(ROOT)))
    report['independent_dynamics_pass']=dynamic_pass
    report['independent_derivative_method']='Saved-velocity cubic interpolation; left/right pieces at the declared contact + 0.10 s balance-controller event; right-sided derivative at the event. No velocity samples changed.'
    report['independent_discrete_contact_momentum_audit']={k:v for k,v in sr.items() if k.startswith('independent_discrete_')}
    if not dynamic_pass:report['status']='motion_candidate_rejected_independent_dynamics'
    report['landing_load_validated']=False
    (out/'cycle_audit.json').write_text(json.dumps(report,indent=2));meta=dict(status=report['status'],movement='box_jump_reference_replay',physical_task=True,force_analysis_allowed=False,landing_simulated=True,landing_included=True,scale=b.report['global_normalization_scale'],body_names=[m.body(i).name for i in range(m.nbody)],phase_times_s=dict(bottom=.85,start_ascent=.85,end_ascent=to,landing=contact,end=float(data['times'][-1])),display_platform_only=True,box_geometry=dict(top_height_m=1.6,x_bounds_m=[-.525,.475],y_bounds_m=[front-.8,front]))
    (out/'summary.json').write_text(json.dumps(meta,indent=2));(out/'README.txt').write_text('160 cm box-jump motion task; nominal 178 cm / 87 kg subject.\nTarget is a Yun-level box height, not measured Yun kinematics or individual force prediction.\nContact: floating-root rigid segments, Coulomb mu=0.8, 3 mm contact tolerance, acceleration stabilization.\nHigh-flexion coordinate ranges have not validated deep-flexion muscle paths.\nMuscle recruitment and individual force accuracy remain unreleased.\nSee cycle_audit.json for checks and explicit limitations.\n');print('FULL_BOX_CYCLE',json.dumps({key:value for key,value in report.items() if key not in ['high_flexion_profile','support_audit']}),flush=True)


if __name__=='__main__':main()
