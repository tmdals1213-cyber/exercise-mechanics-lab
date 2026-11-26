"""Independent multibody inverse-dynamics / momentum checks for jump references."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from jump_rig import JumpRig,OUT,PHYSICS


def normal_weights(points,cx,cy):
    A=np.vstack([np.ones(len(points)),points[:,0],points[:,1]]);target=np.array([1.,cx,cy]);active=list(range(len(points)));w=np.zeros(len(points))
    for _ in range(len(points)):
        C=A[:,active];x=C.T@np.linalg.lstsq(C@C.T,target,rcond=1e-11)[0]
        if x.min()>-1e-8:w[active]=np.maximum(x,0);return w
        active.pop(int(np.argmin(x)))
    raise RuntimeError('COP outside support')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['vertical','long'],required=True);ap.add_argument('--hz',type=int,default=600);ap.add_argument('--solver',choices=['reference','acceleration'],default='reference');args=ap.parse_args()
    dest=OUT/args.kind/f'{args.solver}_{args.hz}hz';z=dict(np.load(dest/'motion.npz'));setup=json.loads((dest/'setup.json').read_text('utf-8'));definition=json.loads((dest/'contact_definition.json').read_text('utf-8'))
    rig=JumpRig();b=rig.b;m,d=rig.m,rig.d;times=z['times'];u_dot=np.zeros_like(z['u'])
    for mode in np.unique(z['mode']):
        indices=np.where(z['mode']==mode)[0]
        if len(indices)>2:u_dot[indices]=np.gradient(z['u'][indices],times[indices],axis=0,edge_order=2)
    points=definition['points'];tau=[];residual=[];contactF=[];contactP=[];power=[];energy=[];passive_power=[];contact_power=[];contact_velocity=[];ground_gaps=[];momentum_expected=[]
    mass=definition['mass_kg'];root_q=np.zeros(m.nv);virtual_check=0.;cartesian_check=0.;constraint_gap=0.
    for i,(t,q,u) in enumerate(zip(times,z['qpos'],z['u'])):
        B,curve=rig.set_state(q,u);d.qacc[:]=z['qacc'][i] if 'qacc' in z else B@u_dot[i]+curve
        Ps=[];Js=[]
        for p in points:
            J,x=b.jacobian(p['body_id'],np.array(p['local_m']));Ps.append(x);Js.append(J)
        Ps=np.asarray(Ps);forces=np.zeros_like(Ps);mode=int(z['mode'][i]);active=[j for j,p in enumerate(points) if mode!=1 or p['forefoot']]
        force=z['force_world_N'][i]
        if force[2]>1e-8:
            weights=normal_weights(Ps[active],z['com'][i,0],z['COP_y_m'][i]);forces[active]=weights[:,None]*force
        external=np.zeros(m.nv)
        for J,F in zip(Js,forces):external+=J.T@F
        required=np.zeros(m.nv);mujoco.mj_rne(m,d,1,required);net=B.T@(required-d.qfrc_passive-external)
        residual.append(net[:6]);tau.append(net[6:]);contactF.append(forces);contactP.append(Ps);power.append(float(net[6:]@u[6:]));passive_power.append(float(d.qfrc_passive@d.qvel));contact_power.append(sum(float(F@(J@d.qvel)) for F,J in zip(forces,Js)))
        contact_velocity.append(max(np.linalg.norm(Js[j]@d.qvel) for j in active) if mode!=2 else 0.)
        ground_gaps.append(Ps[:,2]-definition['ground_z_m'])
        mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);energy.append(d.energy.sum())
        if i%max(1,args.hz//30)==0:
            total=np.zeros(3)
            for bid in range(1,m.nbody):
                if m.body_mass[bid]<=0:continue
                J=np.zeros((3,m.nv));Jr=np.zeros_like(J);Jdot=np.zeros_like(J);Jrdot=np.zeros_like(J)
                mujoco.mj_jacBodyCom(m,d,J,Jr,bid);mujoco.mj_jacDot(m,d,Jdot,Jrdot,d.xipos[bid],bid)
                total+=m.body_mass[bid]*(J@d.qacc+Jdot@d.qvel-m.opt.gravity)
            cartesian_check=max(cartesian_check,float(abs(total-forces.sum(0)-net[:3]).max()))
        if i%(args.hz//2)==0:print('VERIFY_JUMP',args.kind,round(t,3),'base residual',net[:6].round(3),flush=True)
    residual=np.asarray(residual);contactF=np.asarray(contactF);contactP=np.asarray(contactP);energy=np.asarray(energy);gap=np.asarray(ground_gaps)
    events=[.25,.6,.85,setup['takeoff_s']-.02,setup['takeoff_s'],setup['landing_s'],setup['absorption_end_s'],setup['recovery_end_s']]
    regular=np.ones(len(times),bool);regular[[0,1,-1,-2]]=False
    for event in events:regular&=abs(times-event)>3/args.hz
    # Analytic airborne trajectory and zero external support.
    flight=(times>setup['takeoff_s']+.005)&(times<setup['landing_s']-.005);start=np.argmin(abs(times-setup['takeoff_s']));s=times[flight]-times[start]
    expected=z['com'][start]+s[:,None]*z['velocity'][start]+s[:,None]**2*np.array([0,0,-definition['gravity_m_s2']/2])
    ballistic=float(abs(z['com'][flight]-expected).max());H_drift=float(abs(z['angular_momentum'][flight]-z['angular_momentum'][start]).max())
    impulse=np.trapezoid(z['force_world_N']+mass*np.array([0,0,-definition['gravity_m_s2']]),times,axis=0);momentum=mass*(z['velocity'][-1]-z['velocity'][0]);impulse_error=float(abs(impulse-momentum).max())
    friction=float((np.linalg.norm(contactF[:,:,:2],axis=2)-PHYSICS['ground']['static_friction_coefficient']*contactF[:,:,2]).max());min_normal=float(contactF[:,:,2].min())
    work=float(np.trapezoid(np.array(power)+np.array(passive_power)+np.array(contact_power),times));energy_change=float(energy[-1]-energy[0]);energy_error=abs(work-energy_change)
    rom_violation=0.
    for jid in b.independent:
        if m.jnt_limited[jid]:
            angle=z['qpos'][:,m.jnt_qposadr[jid]];lo,hi=m.jnt_range[jid];rom_violation=max(rom_violation,float(np.maximum(lo-angle,angle-hi).max()))
    active_gap=0.
    for i,mode in enumerate(z['mode']):
        if mode==2:continue
        active=[j for j,p in enumerate(points) if mode!=1 or p['forefoot']]
        active_gap=max(active_gap,float(abs(gap[i,active]).max()))
    results={'status':'checked','kind':args.kind,'physics_hz':args.hz,'mass_kg':mass,'air_drag':False,'root_actuation_applied':False,'contact_force_profiles_are_inputs':True,
        'maximum_base_force_residual_regular_N':float(abs(residual[regular,:3]).max()),'maximum_base_moment_residual_regular_Nm':float(abs(residual[regular,3:]).max()),
        'maximum_base_force_residual_all_N':float(abs(residual[:,:3]).max()),'maximum_base_moment_residual_all_Nm':float(abs(residual[:,3:]).max()),'cartesian_vs_generalized_force_crosscheck_N':cartesian_check,
        'ballistic_COM_max_error_m':ballistic,'flight_angular_momentum_max_drift_kg_m2_s':H_drift,'minimum_flight_sole_clearance_m':float(gap[flight].min()),'maximum_sticking_contact_speed_m_s':float(max(contact_velocity)),
        'net_impulse_momentum_error_Ns':impulse_error,'minimum_normal_force_N':min_normal,'maximum_circular_friction_violation_N':friction,
        'total_energy_change_J':energy_change,'integrated_joint_passive_contact_work_J':work,'energy_work_error_J':energy_error,
        'maximum_active_contact_gap_m':active_gap,'minimum_sole_gap_all_modes_m':float(gap.min()),'maximum_joint_ROM_violation_rad':rom_violation,
        'maximum_independent_coordinate_speed_rad_s':float(abs(z['u'][:,6:]).max()),'maximum_net_joint_moment_Nm':float(abs(np.array(tau)).max()),'maximum_net_joint_power_W':float(abs(np.array(power)).max()),
        'event_derivative_exclusion_s':3/args.hz,'scope':'Multibody numerical consistency, not force-plate/subject validation. Joint moments are net generalized demands, not muscle-inclusive cartilage forces.'}
    tol=PHYSICS['release_tolerances']
    gates={
        'root_force':results['maximum_base_force_residual_all_N']<=tol['base_force_residual_N'],
        'root_moment':results['maximum_base_moment_residual_all_Nm']<=tol['base_moment_residual_Nm'],
        'ballistic_COM':ballistic<=tol['ballistic_COM_error_m'],
        'flight_angular_momentum':H_drift<=tol['flight_angular_momentum_drift_kg_m2_s'],
        'contact_velocity':max(contact_velocity)<=tol['contact_speed_m_s'],
        'contact_registration':active_gap<=tol['contact_gap_m'],
        'nonpenetration':float(gap.min())>=-tol['contact_gap_m'],
        'joint_ROM':rom_violation<=tol['joint_ROM_violation_rad'],
        'impulse':impulse_error<=tol['impulse_error_Ns'],
        'friction_cone':friction<=1e-6,
        'unilateral_normal':min_normal>=-1e-6,
        'energy_work':energy_error<=tol['energy_work_error_J']}
    gates={k:bool(v) for k,v in gates.items()}
    results.update(numerical_gates=gates,numerical_consistency_passed=all(gates.values()),status='numerically_consistent_reference' if all(gates.values()) else 'numerical_checks_failed',production_release=False,human_movement_validated=False,muscle_force_solution_available=False)
    np.savez_compressed(dest/'loads.npz',times=times,net_joint_moment_Nm=tau,unactuated_base_residual=residual,contact_force_world_N=contactF,contact_position_world_m=contactP,joint_power_W=power,passive_power_W=passive_power,contact_power_W=contact_power,mechanical_energy_J=energy,sole_ground_gap_m=gap)
    (dest/'verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print('JUMP_AUDIT',json.dumps(results),flush=True)


if __name__=='__main__':main()
