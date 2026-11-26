"""Package the actual high-box trial, retaining rejected geometry and audit.

Rendering is a view of a task candidate, never evidence of successful landing.
Ground forces remain from the contact solve; flight forces are exactly zero.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from retarget_measured_jump import MeasuredJumpRig
from high_flex_profile import apply
from build_contact_jump import contact_geometry
from box_geometry_audit import segments,minimum_clearance
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder');a=ap.parse_args();folder=ROOT/a.folder
    report=json.loads((folder/'resolved_audit.json').read_text());air=dict(np.load(folder/'flight_motion.npz'));ground=dict(np.load(ROOT/report['source_ground']/'dynamic_ground_integrated.npz'))
    r=MeasuredJumpRig(87,1.78);profile=apply(r);b,m,d=r.b,r.m,r.d;r.contact_stand_override=ground['qpos'][0];points=contact_geometry(r)['points']
    arrays={k:[] for k in ['times','qpos','qvel','qacc','body_xpos','body_xmat','com','velocity','angular_momentum','contact_positions_m','collision_segments_m','contact_forces_N','force_world_N','mode']}
    spline=CubicSpline(air['times'],air['u'],axis=0);root_res=[];torques=[];caps_clear=[]
    work=[(0,i) for i in range(0,int(np.searchsorted(ground['times'],air['times'][0])),4)]+[(1,i) for i in range(len(air['times']))]
    for phase,i in work:
        source=air if phase else ground;q=source['qpos'][i];u=source['u'][i];t=float(source['times'][i]);B,curve=r.set_state(q,u)
        ud=spline(t,1) if phase else ground['udot'][i];acc=B@ud+curve
        # This uses the derivative of saved velocity, not a root acceleration
        # chosen to make the inverse-dynamics residual vanish.
        d.qacc[:]=acc;M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M);net=M@acc+d.qfrc_bias-d.qfrc_passive
        if phase and 5<i<len(air['times'])-5:
            root_res.append(net[:6].copy());torques.append((B.T@net)[6:].copy())
        mujoco.mj_subtreeVel(m,d);caps,radii=segments(m,d);caps_clear.append(minimum_clearance(caps,radii,report['box_front_y_m'],1.6))
        P=np.array([d.xpos[bid]+d.xmat[bid].reshape(3,3)@local for bid,local,*_ in points]);forces=np.zeros((len(points),3)) if phase else ground['contact_forces_N'][i]
        values=dict(times=t,qpos=q,qvel=B@u,qacc=acc,body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy(),com=d.subtree_com[1].copy(),velocity=d.subtree_linvel[1].copy(),angular_momentum=d.subtree_angmom[1].copy(),contact_positions_m=P,collision_segments_m=caps,contact_forces_N=forces,force_world_N=forces.sum(0),mode=2 if phase else 0)
        for k,v in values.items():arrays[k].append(v)
    data={k:np.array(v) for k,v in arrays.items()};data['collision_radii_m']=radii;np.savez_compressed(folder/'motion.npz',**data)
    b.pose(b.q0);np.savez_compressed(folder/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
    residual=np.array(root_res);report['independent_flight_root_force_residual_max_N']=float(abs(residual[:,:3]).max());report['independent_flight_root_moment_residual_max_Nm']=float(abs(residual[:,3:]).max());report['flight_net_joint_torque_max_Nm']=float(abs(np.array(torques)).max());report['ground_and_air_body_capsule_clearance_m']=float(min(caps_clear));report['peak_ground_reaction_N']=float(np.linalg.norm(data['force_world_N'],axis=1).max());report['launch_velocity_is_mechanical_scenario_not_athlete_measurement']=True;report['high_flexion_profile']=profile
    P=data['contact_positions_m'][data['mode']==2];front=report['box_front_y_m'];inside=(P[:,:,1]<front)&(P[:,:,1]>front-.8)&(abs(P[:,:,0]+.025)<.5)
    report['sole_box_penetration_max_m']=float(np.maximum(1.6-P[:,:,2],0)[inside].max(initial=0.))
    if report['ground_and_air_body_capsule_clearance_m']<0 or report['sole_box_penetration_max_m']>.003:report['status']='rejected_resolved_high_box';report['precontact_geometry_pass']=False
    meta=dict(status=report['status'],movement='box_jump_reference_replay',physical_task=True,force_analysis_allowed=False,landing_simulated=False,scale=b.report['global_normalization_scale'],body_names=[m.body(i).name for i in range(m.nbody)],phase_times_s=dict(bottom=.85,start_ascent=.85,end_ascent=report['takeoff_s'],end=float(data['times'][-1])),display_platform_only=True,box_geometry=dict(top_height_m=1.6,x_bounds_m=[-.525,.475],y_bounds_m=[front-.8,front]))
    (folder/'summary.json').write_text(json.dumps(meta,indent=2));(folder/'resolved_audit.json').write_text(json.dumps(report,indent=2));print('PACKAGED_HIGH_BOX',json.dumps({k:v for k,v in report.items() if k!='high_flexion_profile'}),flush=True)


if __name__=='__main__':main()
