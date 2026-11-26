"""Build conditional tension histories on sampled existing task operators.

The OHP fixture checks transfer to already verified GH operators. The jump
fixture is diagnostic until the current motion and LAT mapping are repaired.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from tension_reference import solve_history,scale_parameters,reaction_cone
from audit_jump_muscles import GROUPS
from muscle_mechanics import subtree,tendon_body_wrenches
ROOT=Path(__file__).resolve().parents[1]

def input_hashes(paths):return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def spatial_force_matrix(body):
    m,d=body.m,body.d;A=np.zeros((m.nbody,3,m.nu));points=d.wrap_xpos.reshape(-1,3);objects=d.wrap_obj.ravel()
    for actuator in range(m.nu):
        tid=int(m.actuator_trnid[actuator,0]);p=int(d.ten_wrapadr[tid]);end=p+int(d.ten_wrapnum[tid]);previous=None
        for j in range(int(m.tendon_adr[tid]),int(m.tendon_adr[tid]+m.tendon_num[tid])):
            kind=int(m.wrap_type[j]);obj=int(m.wrap_objid[j])
            if kind==int(mujoco.mjtWrap.mjWRAP_SITE):count=1;bid=int(m.site_bodyid[obj])
            elif kind in [int(mujoco.mjtWrap.mjWRAP_SPHERE),int(mujoco.mjtWrap.mjWRAP_CYLINDER)]:count=2 if p<end and objects[p]==obj else 0;bid=int(m.geom_bodyid[obj])
            else:raise ValueError('Unsupported spatial muscle path')
            for _ in range(count):
                point=points[p]
                if previous is not None and previous[0]!=bid:
                    other,prior=previous;delta=point-prior;length=np.linalg.norm(delta)
                    if length>1e-12:
                        pull=m.actuator_gear[actuator,0]*delta/length;A[other,:,actuator]+=pull;A[bid,:,actuator]-=pull
                previous=(bid,point);p+=1
        if p!=end:raise RuntimeError('Muscle path indexing')
    return A


def jump_inputs(samples):
    from jump_rig import JumpRig
    folder=ROOT/'results/jump_validation/vertical/acceleration_2400hz';audit=dict(np.load(folder/'muscle_operating_audit.npz'));info=json.loads((folder/'muscle_operating_audit.json').read_text(encoding='utf-8'));motion=dict(np.load(folder/'motion.npz'));rig=JumpRig();b=rig.b;m,d=b.m,b.d
    ix=np.unique(np.linspace(0,len(audit['times'])-1,samples).round().astype(int));metadata=json.loads((ROOT/'results/grip_comparison/grip_65/cycle/physiology_operators.json').read_text(encoding='utf-8'));geometry=metadata['geometry'];parts=[subtree(m,m.body('humerus_'+s).id) for s in ['r','l']];S=[];offset=[];axes=[];virtual_error=0.
    for k in ix:
        i=info['frames'][k]['index'];b.pose(motion['qpos'][i],motion['qvel'][i],motion['qacc'][i]);A=spatial_force_matrix(b);S.append(np.array([A[part].sum(0) for part in parts]));off=[];axis=[]
        for part,g in zip(parts,geometry):
            required=np.zeros(3)
            for bid in part:
                if m.body_mass[bid]<=0:continue
                J=np.zeros((3,m.nv));Jr=np.zeros_like(J);Jdot=np.zeros_like(J);Jrdot=np.zeros_like(J);mujoco.mj_jacBodyCom(m,d,J,Jr,bid);mujoco.mj_jacDot(m,d,Jdot,Jrdot,d.xipos[bid],bid);required+=m.body_mass[bid]*(J@d.qacc+Jdot@d.qvel-m.opt.gravity)
            off.append(required);side=g['side'];sbid=m.body('scapula_'+side).id;hbid=m.body('humerus_'+side).id;centre=d.xpos[sbid]+d.xmat[sbid].reshape(3,3)@(np.array(g['glenoid_centre_local_native_m'])*b.report['global_normalization_scale']);direction=d.xpos[hbid]-centre;axis.append(direction/np.linalg.norm(direction))
        offset.append(off);axes.append(axis)
        probe=np.linspace(.1,1.,m.nu);independent=tendon_body_wrenches(b,probe)[:,:3];virtual_error=max(virtual_error,float(abs(independent-np.einsum('bcn,n->bc',A,probe)).max()))
    if virtual_error>1e-8:raise RuntimeError('Independent muscle spatial assembly mismatch')
    return {'times':audit['times'][ix],'R':audit['moment_arms_m'][ix],'target':audit['net_demand_Nm'][ix],'gain':audit['active_gain_N'][ix],'passive':audit['passive_N'][ix],'reference':m.actuator_gainprm[:,2].copy(),'names':b.names,'S':np.array(S),'offset':np.array(offset),'axes':np.array(axes),'warnings':['Sampled diagnostic motion; current jump includes excessive internal transients.','LAT1/LAT1_l inherited operating-length mapping needs review.','Thoracoscapular force paths are not yet integrated.','GH cone is an approximate directional constraint; lower-joint co-contraction is not identified.'],'source':str(folder.relative_to(ROOT)),'spatial_assembly_error':virtual_error,'input_sha256':input_hashes([folder/'motion.npz',folder/'muscle_operating_audit.npz',folder/'muscle_operating_audit.json',ROOT/'results/grip_comparison/grip_65/cycle/physiology_operators.json'])}


def ohp_inputs(samples):
    folder=ROOT/'results/grip_comparison/grip_65/cycle';o=dict(np.load(folder/'physiology_operators.npz'));motion=dict(np.load(folder/'physiology/fullbody_motion.npz'));meta=json.loads((folder/'physiology/fullbody_metadata.json').read_text(encoding='utf-8'));ix=np.unique(np.linspace(0,len(o['times'])-1,samples).round().astype(int));target=o['target']-np.einsum('tqc,tc->tq',o['C'],motion['contacts_N'].reshape(len(o['times']),-1)/1000)
    return {'times':o['times'][ix],'R':o['R'][ix],'target':target[ix],'gain':o['gain_N'][ix],'passive':o['passive_N'][ix],'reference':np.array(meta['reference_force_N']),'names':meta['muscle_names'],'S':o['shoulder_force_coeff'][ix],'offset':o['shoulder_force_offset'][ix],'axes':o['cone_axis'][ix],'warnings':['Existing OHP operators reused to verify a task-independent tension solver, not a new video.','Fixed ground-contact forces from the verified reference.','Generic GH cone and retained muscle parameters; not personal recruitment.'],'source':str(folder.relative_to(ROOT)),'spatial_assembly_error':None,'input_sha256':input_hashes([folder/'physiology_operators.npz',folder/'physiology/fullbody_motion.npz',folder/'physiology/fullbody_metadata.json'])}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['jump','ohp'],default='jump');ap.add_argument('--samples',type=int,default=25);a=ap.parse_args();cfg=json.loads((ROOT/'scenarios/tension_reference.json').read_text(encoding='utf-8'));data=(jump_inputs if a.task=='jump' else ohp_inputs)(a.samples);rule=cfg['solution_selection'];stability=[reaction_cone(S,off,axis,rule['GH_cone_half_angle_deg'],rule['GH_cone_faces']) for S,off,axis in zip(data['S'],data['offset'],data['axes'])];dest=ROOT/'results/tension_reference'/a.task;dest.mkdir(parents=True,exist_ok=True);reports={};results={}
    for name,profile in cfg['profiles'].items():
        ref,gain,passive,scale=scale_parameters(data['names'],data['reference'],data['gain'],data['passive'],profile,GROUPS);solution=solve_history(data['times'],data['R'],data['target'],gain,passive,ref,stability,rule['temporal_weight'],rule['level_weight']);F=solution['force_N'];reaction=data['offset']-np.einsum('tscn,tn->tsc',data['S'],F);angles=[]
        for axis,jrf in zip(data['axes'],reaction):angles.append(np.rad2deg(np.arccos(np.clip(np.sum(axis*jrf,axis=1)/np.maximum(np.linalg.norm(jrf,axis=1),1e-12),-1,1))))
        checks=solution.pop('checks');checks.update(GH_max_angle_deg=float(np.max(angles)),minimum_GH_axial_compression_N=float(np.min(np.sum(data['axes']*reaction,axis=2))),scaled_elements=[data['names'][i] for i in np.where(scale!=1)[0]],geometry_and_body_mass_unchanged=True)
        if checks['GH_max_angle_deg']>rule['GH_cone_half_angle_deg']+.01:raise RuntimeError('Independent circular GH cone check failed')
        np.savez_compressed(dest/(name+'.npz'),times=data['times'],**solution,reference_force_N=ref,force_parameter_scale=scale)
        reports[name]=checks;results[name]=F;print('TENSION_REFERENCE',a.task,name,'balance',checks['moment_balance_max_error'],'GH angle',checks['GH_max_angle_deg'],flush=True)
    changes={name:float(abs(F-results['standard']).max()) for name,F in results.items()};summary={'status':'conditional_tension_reference_diagnostic' if a.task=='jump' else 'operator_verification_fixture','task':a.task,'samples':len(data['times']),'names':data['names'],'source':data['source'],'input_sha256':data['input_sha256'],'profiles':reports,'largest_force_difference_from_standard_N':changes,'muscle_force_trajectory_publication_ready':False,'strength_ceiling_enforced':False,'primary_output':'muscle-tendon tension, N','selection_rule':rule,'warnings':data['warnings'],'independent_spatial_assembly_error_N':data['spatial_assembly_error'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'solver_sha256':hashlib.sha256((ROOT/'scripts/tension_reference.py').read_bytes()).hexdigest(),'configuration_sha256':hashlib.sha256((ROOT/'scenarios/tension_reference.json').read_bytes()).hexdigest()};(dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('TENSION_REFERENCE_COMPLETE',a.task,changes,flush=True)

if __name__=='__main__':main()
