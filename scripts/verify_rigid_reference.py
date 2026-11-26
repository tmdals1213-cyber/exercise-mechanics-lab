"""Conditional reference checks, with independent Newton-Euler snapshots."""
import argparse,json,hashlib,numpy as np,mujoco
from pathlib import Path
from rigid_baseline_profile import RigidBaselineRig
from mechanics_engine_adapter import sample
from verify_jump_reference import normal_weights
from jump_rig import ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hz',type=int,default=600);ap.add_argument('--folder',type=Path);a=ap.parse_args();dest=a.folder or ROOT/f'results/mechanics_foundation/rigid_reference_{a.hz}hz';z=np.load(dest/'motion.npz');setup=json.loads((dest/'setup.json').read_text('utf-8'));definition=json.loads((dest/'contact_definition.json').read_text('utf-8'))
    from jump_model import create_rig
    rig=create_rig(setup.get('rig_type','rigid_baseline'))
    m,d=rig.m,rig.d
    assert setup['profile']==rig.profile,'Profile mismatch';N=len(z['times']);moments=[];base=[];energy=[];powers=[];contactpowers=[];forces=[];positions=[];speeds=[];gaps=[];snapshots=[];independent_error=0.;cut_error=0.;sample_indices=set(np.linspace(0,N-1,40).round().astype(int))
    for name,expected in setup['source_sha256'].items():assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==expected,'Source changed: '+name
    for i,t in enumerate(z['times']):
        B,_=rig.set_state(z['qpos'][i],z['u'][i]);d.qacc[:]=z['qacc'][i];js=[];ps=[]
        for p in definition['points']:J,x=rig.b.jacobian(p['body_id'],np.array(p['local_m']));js.append(J);ps.append(x)
        ps=np.array(ps);fs=np.zeros_like(ps);active=[j for j,p in enumerate(definition['points']) if z['mode'][i]!=1 or p['forefoot']];F=z['force_world_N'][i]
        if 'contact_forces_N' in z:fs=z['contact_forces_N'][i]
        elif F[2]>1e-8:fs[active]=normal_weights(ps[active],z['com'][i,0],z['COP_y_m'][i])[:,None]*F
        external=sum((J.T@f for J,f in zip(js,fs)),start=np.zeros(m.nv));rne=np.zeros(m.nv);mujoco.mj_rne(m,d,1,rne);net=B.T@(rne-external);base.append(net[:6]);moments.append(net[6:]);powers.append(float(net[6:]@z['u'][i,6:]));contactpowers.append(sum(float(f@(J@d.qvel)) for f,J in zip(fs,js)));forces.append(fs);positions.append(ps);speeds.append(max(np.linalg.norm(js[j]@d.qvel) for j in active) if z['mode'][i]!=2 else 0.);gaps.append(ps[:,2]-definition['ground_z_m']);mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);energy.append(float(d.energy.sum()))
        if i in sample_indices:
            contact=[(p['body_id'],ps[j],fs[j],np.zeros(3)) for j,p in enumerate(definition['points'])];s=sample(rig,z['qpos'][i],z['qvel'][i],z['qacc'][i],contact);independent_error=max(independent_error,s['crosscheck_max']);cut_error=max(cut_error,s['cut_vs_virtual_work_max']);snapshots.append({'time_s':float(t),'root_world_wrench':s['subtree_wrenches'][0].tolist(),'cuts':{k:s['cuts'][k] for k in ['ankle_angle_r','knee_angle_r','hip_flexion_r','shoulder_elv_r']}})
        if i%max(1,a.hz)==0:print('VERIFY_RIGID',a.hz,round(float(t),3),flush=True)
    ts=z['times'];moments=np.array(moments);base=np.array(base);forces=np.array(forces);gaps=np.array(gaps);mass=definition['mass_kg'];impulse=np.trapezoid(forces.sum(1)+[0,0,-mass*9.80665],ts,axis=0);dp=mass*(z['velocity'][-1]-z['velocity'][0]);work=float(np.trapezoid(np.array(powers)+np.array(contactpowers),ts));dE=energy[-1]-energy[0];rom=0.
    if 'integration_dt_s' in z:
        impulse=np.sum((z['integration_force_world_N']+[0,0,-mass*9.80665])*z['integration_dt_s'][:,None],axis=0);work=float(z['integration_power_W']@z['integration_dt_s'])
    for jid in rig.b.independent:
        if m.jnt_limited[jid]:lo,hi=m.jnt_range[jid];qq=z['qpos'][:,m.jnt_qposadr[jid]];rom=max(rom,float(np.maximum(lo-qq,qq-hi).max()))
    shoulder=[i for i,n in enumerate(rig.b.ind_names) if n.startswith(('shoulder_','elv_angle'))];foot=[i for i,n in enumerate(rig.b.ind_names) if n.startswith(('ankle_','knee_','hip_'))];flight=z['mode']==2;activegap=0.
    for i,mode in enumerate(z['mode']):
        if mode==2:continue
        ids=[j for j,p in enumerate(definition['points']) if mode!=1 or p['forefoot']];activegap=max(activegap,float(abs(gaps[i,ids]).max()))
    checks={'root_force_residual_N':float(abs(base[:,:3]).max()),'root_moment_residual_Nm':float(abs(base[:,3:]).max()),'impulse_momentum_error_Ns':float(abs(impulse-dp).max()),'energy_work_error_J':abs(work-dE),'relative_energy_closure_vs_absolute_work':abs(work-dE)/max(1,float(np.trapezoid(abs(np.array(powers))+abs(np.array(contactpowers)),ts))),'contact_velocity_max_m_s':max(speeds),'active_contact_gap_max_m':activegap,'minimum_gap_all_modes_m':float(gaps.min()),'ROM_violation_rad':rom,'independent_NE_crosscheck_max_SI':independent_error,'recursive_cut_crosscheck_max_SI':cut_error,'shoulder_generalized_torque_peak_Nm':float(abs(moments[:,shoulder]).max()),'leg_generalized_torque_peak_Nm':float(abs(moments[:,foot]).max()),'all_joint_torque_peak_Nm':float(abs(moments).max()),'coordinate_speed_peak_rad_s':float(abs(z['u'][:,6:]).max()),'GRF_peak_N':float(forces.sum(1)[:,2].max()),'flight_force_max_N':float(abs(forces[flight]).max())}
    gates={'root_force':checks['root_force_residual_N']<.1,'root_moment':checks['root_moment_residual_Nm']<.1,'impulse':checks['impulse_momentum_error_Ns']<.1,'energy':checks['energy_work_error_J']<2.,'contact_speed':checks['contact_velocity_max_m_s']<.01,'contact_gap':activegap<.002,'nonpenetration':checks['minimum_gap_all_modes_m']>-.002,'joint_ROM':rom<.002,'zero_support_in_flight':checks['flight_force_max_N']<1e-8,'independent_NE':independent_error<1e-6,'recursive_cuts':cut_error<1e-6}
    gates={k:bool(v) for k,v in gates.items()}
    report={'status':'conditional_numerical_reference' if all(gates.values()) else 'numerical_checks_failed','human_motion_validated':False,'muscle_forces_released':False,'ground_reaction_and_COP_are_inputs':setup.get('ground_reaction_and_COP_are_inputs',True),'checks':checks,'gates':gates,'snapshots':snapshots,'profile':rig.profile,'motion_sha256':hashlib.sha256((dest/'motion.npz').read_bytes()).hexdigest()};(dest/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');np.savez_compressed(dest/'joint_loads.npz',times=ts,joint_moments_Nm=moments,base_residual=base,contact_forces_N=forces,contact_positions_m=positions,joint_power_W=powers,contact_power_W=contactpowers,energy_J=energy)
    print(json.dumps({k:report[k] for k in ['status','checks','gates']}),flush=True)
if __name__=='__main__':main()
