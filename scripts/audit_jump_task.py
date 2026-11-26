"""Task-specific motion acceptance and explicit calf moment reconstruction.

Conservation alone cannot distinguish a vertical jump from a somersault.
Muscle force is never accepted/rejected solely by comparing it with total GRF.
"""
import argparse,json,hashlib,numpy as np
from pathlib import Path
from jump_model import create_rig


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);ap.add_argument('--output-dir',type=Path);a=ap.parse_args();p=a.folder;output=a.output_dir or p;output.mkdir(parents=True,exist_ok=True);z=np.load(p/'motion.npz');setup=json.loads((p/'setup.json').read_text());rig=create_rig(setup['rig_type']);m=rig.m;names=setup['body_names'];bid=names.index('torso');up=z['body_xmat'][:,bid].reshape(-1,3,3)[:,:,1];tilt=np.degrees(np.arccos(np.clip(up[:,2],-1,1)));flight=z['mode']==2;to=int(np.flatnonzero(flight)[0]);H=z['angular_momentum'];f=z['contact_forces_N'];used=f[:,:,2]>1.;mu=np.linalg.norm(f[:,:,:2],axis=2)/np.maximum(f[:,:,2],1e-9)
    checks=dict(takeoff_s=float(z['times'][to]),takeoff_angular_momentum_kg_m2_s=float(np.linalg.norm(H[to])),maximum_flight_thorax_tilt_deg=float(tilt[flight].max()),takeoff_horizontal_speed_m_s=float(np.linalg.norm(z['velocity'][to,:2])),takeoff_vertical_speed_m_s=float(z['velocity'][to,2]),COM_rise_after_takeoff_m=float(z['com'][flight,2].max()-z['com'][to,2]),peak_total_vertical_GRF_N=float(f.sum(1)[:,2].max()),peak_GRF_bodyweights=float(f.sum(1)[:,2].max()/(m.body_mass.sum()*9.80665)),max_contact_friction_ratio_above_1N=float(mu[used].max()),minimum_normal_force_N=float(f[:,:,2].min()))
    gates=dict(takeoff_rotation=checks['takeoff_angular_momentum_kg_m2_s']<1.,upright_flight=checks['maximum_flight_thorax_tilt_deg']<20.,vertical_launch=checks['takeoff_horizontal_speed_m_s']<.2,circular_friction=checks['max_contact_friction_ratio_above_1N']<=.800001,compressive_support=checks['minimum_normal_force_N']>-.00001)
    report=dict(status='passed' if all(gates.values()) else 'rejected_for_vertical_jump',motion_sha256=hashlib.sha256((p/'motion.npz').read_bytes()).hexdigest(),checks=checks,gates=gates,definition='Designed acceptance for a moderate upright vertical-jump reference; not experimental validation.',calf_force_ceiling_enforced=False);(output/'motion_quality.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not (p/'tensions.npz').exists():return
    t=np.load(p/'tensions.npz');summary=json.loads((p/'tension_summary.json').read_text());definition=json.loads((p/'contact_definition.json').read_text());muscle_names=summary['names'];audits=[]
    if summary['source_motion_sha256']!=hashlib.sha256((p/'motion.npz').read_bytes()).hexdigest():
        print('CALF_AUDIT_SKIPPED_STALE_FORCE_SOLUTION',flush=True);return
    for side in ['r','l']:
        calf=[i for i,n in enumerate(muscle_names) if n.startswith(('soleus','gasmed','gaslat')) and n.endswith('_'+side)];F=t['force_N'][:,calf].sum(1);k=int(F.argmax());index=int(t['source_indices'][k]);rig.b.pose(z['qpos'][index],z['qvel'][index],z['qacc'][index]);jid=rig.b.joints['ankle_angle_'+side];anchor=rig.d.xanchor[jid].copy();axis=rig.d.xaxis[jid].copy();ids=[i for i,pt in enumerate(definition['points']) if pt['body'].endswith('_'+side)];forces=f[index,ids];positions=z['contact_positions_m'][index,ids];net=forces.sum(0);moment=float(sum(np.cross(pos-anchor,force)@axis for pos,force in zip(positions,forces)));row=summary['coordinate_names'].index('ankle_angle_'+side);R=t['R'][k,row];calf_moment=float(R[calf]@t['force_N'][k,calf]);totalmoment=float(R@t['force_N'][k]);weight=np.maximum(forces[:,2],0);COP=np.average(positions,axis=0,weights=weight) if weight.sum()>0 else positions.mean(0)
        q=z['qpos'][index].copy();eps=1e-6;lengths=[]
        for sign in [-1,1]:
            trial=q.copy();trial[m.jnt_qposadr[jid]]+=sign*eps;rig.b.pose(trial);lengths.append(rig.d.actuator_length[calf].copy())
        fd=-(lengths[1]-lengths[0])/(2*eps)
        audits.append(dict(side=side,time_s=float(t['times'][k]),total_GRF_N=f[index].sum(0).tolist(),same_foot_GRF_N=net.tolist(),ankle_world_m=anchor.tolist(),COP_world_m=COP.tolist(),GRF_ankle_moment_Nm=moment,effective_external_lever_m=abs(moment)/max(np.linalg.norm(net),1e-9),required_net_ankle_moment_Nm=float(t['target'][k,row]),calf_muscle_moment_Nm=calf_moment,other_muscle_moment_Nm=totalmoment-calf_moment,total_calf_elements_tension_N=float(F[k]),force_weighted_calf_moment_arm_m=abs(calf_moment)/max(F[k],1e-9),moment_arm_finite_difference_max_error_m=float(abs(fd-R[calf]).max()),balance_error_Nm=abs(totalmoment-t['target'][k,row]),elements=[dict(name=muscle_names[j],tension_N=float(t['force_N'][k,j]),signed_moment_arm_m=float(R[j]),finite_difference_moment_arm_m=float(fd[jj]),passive_N=float(t['passive'][k,j])) for jj,j in enumerate(calf)]))
    (output/'calf_moment_audit.json').write_text(json.dumps(dict(motion_status=report['status'],definition='Sum of modeled calf muscle-tendon element tensions; not a direct measurement of common Achilles tendon force.',audits=audits,references=['https://pubmed.ncbi.nlm.nih.gov/37787026/','https://pubmed.ncbi.nlm.nih.gov/23915826/']),indent=2));print('CALF_MOMENT_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
