"""Common-geometry jump propulsion versus squat ascent diagnostic.

One reproducible force sharing solution; not unique neural recruitment. No
strength ceiling. Passive tension is standard-model-dependent. Landing excluded.
"""
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco,opensim as osim
from scipy.optimize import linprog
from rigid_baseline_profile import RigidBaselineRig
from conservative_ellipsoid_paths import ConservativeEllipsoidPaths,force_basis
from build_tension_reference import spatial_force_matrix
from mechanics_engine_adapter import sample,path_speeds
from muscle_mechanics import subtree
from tension_reference import solve_causal_history,solve_history,reaction_cone
from jump_rig import ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['jump','squat'],required=True);ap.add_argument('--fps',type=int,default=30);ap.add_argument('--published-patella',action='store_true');ap.add_argument('--label',default='yun_common_v1');ap.add_argument('--selection',choices=['causal','batch'],default='batch');ap.add_argument('--shoulder-passive',choices=['source','unidentified'],default='source');ap.add_argument('--published-leg-paths',action='store_true');ap.add_argument('--full-cycle',action='store_true');ap.add_argument('--motion-folder');args=ap.parse_args();args.scapula_contact=True
    from pathlib import Path
    from retarget_measured_jump import MeasuredJumpRig
    from high_flex_profile import apply
    from scipy.interpolate import CubicSpline
    from build_contact_jump import contact_geometry
    sourcefolder=ROOT/('results/physical_jump_task/yun_extended_launch_v10' if args.case=='jump' else 'results/power_comparison/yun_sungbin/squat_240/analysis')
    if args.motion_folder:sourcefolder=ROOT/args.motion_folder
    sourcefile=sourcefolder/('motion.npz' if args.motion_folder or args.case=='squat' else 'dynamic_ground_integrated.npz');z=dict(np.load(sourcefile));source_hash=hashlib.sha256(sourcefile.read_bytes()).hexdigest()
    rig=MeasuredJumpRig(87,1.78);apply(rig)
    profile=None
    if args.published_patella:
        from published_patella_profile import apply as apply_patella
        profile=apply_patella(rig)
    leg_profile=None
    if args.published_leg_paths:
        from published_knee_paths_profile import apply as apply_leg_paths
        leg_profile=apply_leg_paths(rig)
    rig.contact_stand_override=z['qpos'][0];b,m,d=rig.b,rig.m,rig.d
    definition=contact_geometry(rig) if args.case=='jump' else json.loads((sourcefolder/'contact_definition.json').read_text())
    if args.case=='jump':definition={'points':[{'body_id':int(item[0]),'local':np.asarray(item[1]).tolist()} for item in definition['points']]}
    # Recompute dependent-coordinate velocities/accelerations using the same
    # geometry profile as the force paths. No source forces are adjusted.
    if 'u' not in z:z['u']=np.c_[z['qvel'][:,:6],z['qvel'][:,rig.dofs]]
    original_count=len(z['times'])
    # Integrated ground states include a final state without a solved udot.
    # Keep only common, solved time samples before any window selection.
    # Never index shortened qvel/qacc with indices from the full time array.
    if 'udot' in z:
        common=min(original_count,len(z['qpos']),len(z['u']),len(z['udot']))
        z={k:(val[:common] if isinstance(val,np.ndarray) and val.ndim and len(val) in [original_count,common] else val) for k,val in z.items()}
        udot=z['udot']
    elif 'qacc' in z:udot=np.c_[z['qacc'][:,:6],z['qacc'][:,rig.dofs]]
    else:udot=CubicSpline(z['times'],z['u'],axis=0)(z['times'],1)
    z['qpos']=np.array([b.complete(q) for q in z['qpos']]);v=[];acc=[]
    for q,u,ud in zip(z['qpos'],z['u'],udot):
        B,curve=rig.set_state(q,u);v.append(B@u);acc.append(B@ud+curve)
    z['qvel']=np.array(v);z['qacc']=np.array(acc)
    if 'contact_positions_m' not in z:
        positions=[]
        for q in z['qpos']:
            b.pose(q);positions.append([d.xpos[p['body_id']]+d.xmat[p['body_id']].reshape(3,3)@p['local'] for p in definition['points']])
        z['contact_positions_m']=np.array(positions)
    assert len(z['qpos'])==len(z['qvel'])==len(z['qacc'])==len(z['times'])
    begin,end=(.85,float(z['times'][-1])) if args.case=='jump' else (1.25,3.25)
    if args.full_cycle:begin,end=float(z['times'][0]),float(z['times'][-1])
    mask=(z['times']>=begin-1e-8)&(z['times']<=end+1e-8);nt=len(z['times']);z={k:(val[mask] if isinstance(val,np.ndarray) and val.ndim and len(val)==nt else val) for k,val in z.items()}
    folder=ROOT/'results/muscle_validation'/args.label/args.case;folder.mkdir(parents=True,exist_ok=True)
    verified=dict(status='conditional_common_geometry_diagnostic',source_motion_sha256=source_hash,source_file=str(sourcefile.relative_to(ROOT)),analysis_window_s=[begin,end],published_patella_profile=profile,published_leg_paths_profile=leg_profile,human_muscle_force_validated=False)
    b,m,d=rig.b,rig.m,rig.d;bridge=ConservativeEllipsoidPaths(rig);names=b.names+bridge.names;muscle_count=len(names);refs=np.r_[m.actuator_gainprm[:,2],bridge.refs]

    from pathlib import Path
    assert hashlib.sha256(Path('D:/Exercise_mechanics/models/foundation/TSM_subject_noWeight.osim').read_bytes()).digest()==hashlib.sha256((ROOT/'models/rmr/TSM_subject_noWeight.osim').read_bytes()).digest(),'Shoulder source variants differ'
    if args.scapula_contact:names+=['scapular_contact_'+str(i) for i in range(4)];refs=np.r_[refs,np.full(4,500.)]
    nm=len(names)
    print('TENSION_CONFIGURATION',args.shoulder_passive,str(__file__),flush=True)
    source=ET.parse(ROOT/'models/rmr/TSM_subject_noWeight.osim');architecture={m.attrib['name']:{k:float(m.findtext(k)) for k in ['optimal_fiber_length','tendon_slack_length','max_isometric_force']} for m in source.findall('.//Millard2012EquilibriumMuscle')};g=b.report['global_normalization_scale']
    original=osim.Model('D:/Exercise_mechanics/models/foundation/TSM_subject_noWeight.osim');original_state=original.initSystem();original.realizePosition(original_state);source_lengths={mus.getName():mus.getLength(original_state) for mus in original.getMuscles()};b.pose(b.q0);neutral_lengths=bridge.operators()[2];registered={}
    for name,L in zip(bridge.names,neutral_lengths):
        ar=architecture[name[:-2]].copy();factor=L/source_lengths[name[:-2]]
        ar['optimal_fiber_length']*=factor;ar['tendon_slack_length']*=factor;ar['path_similarity_scale']=factor;registered[name]=ar
    # LAT1's inherited slack mapping was nonphysical even in the unscaled
    # source model. Use the published upper-lat standard architecture explicitly.
    lr=m.actuator_lengthrange.copy();repairs=[]
    for j,name in enumerate(b.names):
        if name in ['LAT1','LAT1_l']:
            ar=architecture['LatissimusDorsi_S'];l0=ar['optimal_fiber_length']*g;lt=ar['tendon_slack_length']*g;lr[j]=lt+m.actuator_gainprm[j,:2]*l0;repairs.append(dict(element=name,optimal_fiber_length_m=l0,tendon_slack_length_m=lt,source='Seth2019/RMR upper latissimus standard; generic transfer, not personal anatomy'))
    target_times=np.unique(np.r_[np.arange(z['times'][0],z['times'][-1]+1e-8,1/args.fps),z['times'][-1]])
    # Stored decimal times can round just below an exact display target.
    # Select the nearest physical sample rather than consistently advancing it.
    indices=np.unique(np.array([int(np.argmin(abs(z['times']-tt))) for tt in target_times]))
    geometry=json.loads((ROOT/'results/grip_comparison/grip_65/cycle/physiology_operators.json').read_text())['geometry'];parts=[subtree(m,m.body('humerus_'+s).id) for s in ['r','l']]
    arrays={k:[] for k in ['R','target','passive','source_shoulder_passive_N','S','offset','axes','length_m','path_velocity_m_s','unit_wrench_closure','GH_cut_offset_N','root_residual','NE_crosscheck']};maxfd=0.;coordinate_names=None;checks=[]
    for count,index in enumerate(indices):
        q,v,a=z['qpos'][index],z['qvel'][index],z['qacc'][index];b.pose(q,v,a);B,coordinate_names=force_basis(rig,q,z['u'][index]);Rb,Ab,Lb,_=bridge.operators();Rn=b.moments();An=spatial_force_matrix(b);Rall=np.column_stack([Rn,Rb]);Aall=np.concatenate([An,Ab],axis=2)
        if args.scapula_contact:
            C,points,gaps=rig.contact_operators(q);Ac=np.zeros((m.nbody,3,4))
            for j,(bid,cb,p,n,_) in enumerate(points):Ac[bid,:,j]=n;Ac[cb,:,j]=-n
            Rall=np.column_stack([Rall,C]);Aall=np.concatenate([Aall,Ac],axis=2)
        contacts=[(p['body_id'],z['contact_positions_m'][index,j],z['contact_forces_N'][index,j],z['contact_free_moments_Nm'][index,j] if 'contact_free_moments_Nm' in z else np.zeros(3)) for j,p in enumerate(definition['points'])];mechanics=sample(rig,q,v,a,contacts);demand=(B.T@mechanics['generalized'])[6:];R=(B.T@Rall)[6:]
        passive=[]
        for j in range(m.nu):passive.append(max(0.,-mujoco.mju_muscleBias(float(d.actuator_length[j]),np.ascontiguousarray(lr[j]),float(m.actuator_acc0[j]),np.ascontiguousarray(m.actuator_biasprm[j,:9]))))
        for name,L in zip(bridge.names,Lb):
            ar=registered[name];norm=(L-ar['tendon_slack_length'])/ar['optimal_fiber_length']
            # Declared Thelen-style exponential reference spring, zero below
            # optimal length. Not an identified tissue law or strength ceiling.
            passive.append(ar['max_isometric_force']*np.expm1(4*max(norm-1,0)/.6)/np.expm1(4))
        passive=np.array(passive);source_shoulder_passive=passive[m.nu:].copy()
        if args.shoulder_passive=='unidentified':passive[m.nu:]=0
        if args.scapula_contact:passive=np.r_[passive,np.zeros(4)]
        S=np.array([Aall[p].sum(0) for p in parts]);offset=np.array([mechanics['body_wrenches'][p,:3].sum(0) for p in parts]);axes=[]
        for geo in geometry:
            s=geo['side'];sb=m.body('scapula_'+s).id;hb=m.body('humerus_'+s).id;centre=d.xpos[sb]+d.xmat[sb].reshape(3,3)@(np.array(geo['glenoid_centre_local_native_m'])*g);axis=d.xpos[hb]-centre;axes.append(axis/np.linalg.norm(axis))
        values=dict(R=R,target=demand,passive=passive,source_shoulder_passive_N=source_shoulder_passive,S=S,offset=offset,axes=axes,length_m=np.r_[d.actuator_length.copy(),Lb,np.zeros(4) if args.scapula_contact else np.zeros(0)],path_velocity_m_s=-Rall.T@v,unit_wrench_closure=np.max(abs(bridge.unit_wrenches.sum(0))),GH_cut_offset_N=offset,root_residual=mechanics['generalized'][:6],NE_crosscheck=mechanics['crosscheck_max'])
        for k,val in values.items():arrays[k].append(val)
        keep=[i for i,name in enumerate(coordinate_names[6:]) if not name.startswith('mtp_angle_')];mat=R[keep]*refs;scale=np.maximum(np.linalg.norm(mat,axis=1),1);lp=linprog(np.ones(nm),A_eq=mat/scale[:,None],b_eq=(demand-R@passive)[keep]/scale,bounds=(0,None),method='highs')
        checks.append(dict(time_s=float(z['times'][index]),feasible=bool(lp.success),status=lp.message))
        if not lp.success:
            zero=linprog(np.ones(nm),A_eq=mat/scale[:,None],b_eq=demand[keep]/scale,bounds=(0,None),method='highs');native_only=passive.copy();native_only[m.nu:]=0;no_new=linprog(np.ones(nm),A_eq=mat/scale[:,None],b_eq=(demand-R@native_only)[keep]/scale,bounds=(0,None),method='highs');nq=len(keep)
            elastic=linprog(np.r_[np.zeros(nm),np.ones(2*nq)],A_eq=np.c_[mat/scale[:,None],np.eye(nq),-np.eye(nq)],b_eq=(demand-R@passive)[keep]/scale,bounds=(0,None),method='highs')
            residual=(elastic.x[nm:nm+nq]-elastic.x[nm+nq:])*scale if elastic.success else np.zeros(nq)
            detail=dict(frame=count,passive_max_N=float(passive.max()),zero_passive_feasible=bool(zero.success),zero_new_passive_feasible=bool(no_new.success),missing_moments=[(coordinate_names[6+keep[i]],float(v)) for i,v in enumerate(residual) if abs(v)>1e-5],detail=checks[-1]);(folder/'tension_failure.json').write_text(json.dumps(detail,indent=2));raise RuntimeError(('Nonnegative tension infeasible',detail))
        print('JUMP_TENSION_OPERATORS',count+1,len(indices),'time',round(float(z['times'][index]),3),flush=True)
    arrays={k:np.array(v) for k,v in arrays.items()};ts=z['times'][indices];stability=[reaction_cone(S,o,ax,20.,16) for S,o,ax in zip(arrays['S'],arrays['offset'],arrays['axes'])]
    # Intrinsic plantar muscles and plantar fascia are absent. Do not force
    # their MTP support moment into the few extrinsic digital actuators.
    unresolved=[i for i,name in enumerate(coordinate_names[6:]) if name.startswith('mtp_angle_')]
    solved=[i for i in range(arrays['R'].shape[1]) if i not in unresolved]
    selector=solve_history if args.selection=='batch' else solve_causal_history
    solution=selector(ts,arrays['R'][:,solved],arrays['target'][:,solved],np.ones_like(arrays['passive']),arrays['passive'],refs,stability,temporal_weight=.002,level_weight=1.)
    solution['unidentified_joint_support_moment_Nm']=arrays['target']-np.einsum('tqn,tn->tq',arrays['R'],solution['force_N'])
    solution.pop('unbounded_effort_coefficient',None);solution['force_to_generic_reference_ratio']=solution['force_N']/refs
    solution['checks'].pop('maximum_effort_coefficient',None);solution['checks']['maximum_force_to_generic_reference_ratio']=float(solution['force_to_generic_reference_ratio'].max())
    F=solution['force_N'];reaction=arrays['offset']-np.einsum('tscn,tn->tsc',arrays['S'],F);summary=dict(status='conditional_force_balance_solution',human_muscle_force_validated=False,scope='stance through flight; landing not solved',model_muscle_elements=nm,names=names,coordinate_names=coordinate_names[6:],strength_ceiling_enforced=False,activation_or_EMG_prediction=False,active_force_velocity_limit_enforced=False,force_sharing='normalized force magnitude plus causal continuity; GH reaction cone 20 degrees; no prescribed antagonist floor',passive_law='native MuJoCo passive curves, repaired LAT1 architecture; new shoulder elements exponential reference spring k=4, strain=.6',architecture_repairs=repairs,shoulder_girdle_force_coupling='five SC/AC rotations per side balanced independently; original rhythm only prescribes motion',shoulder_wrapping='source ellipsoid length; conservative derivatives in chest frame',checks=solution.pop('checks'),peak_element_N=float(F.max()),peak_element=names[int(np.unravel_index(F.argmax(),F.shape)[1])],max_added_unit_wrench_closure_SI=float(arrays['unit_wrench_closure'].max()),sample_count=len(ts),source_motion_sha256=source_hash,warnings=['Generic attachment registration needs anatomical validation.','Native ideal joints, numerical scapular rhythm, approximate wrapping, and GH cone remain assumptions.','No independent neural co-contraction inference. Zero is permitted by the chosen conditional solution.','Muscle-tendon tension only; not tissue stress, activation, damage, or hypertrophy.','Sampled force solution does not validate unsampled times.'])
    summary['unidentified_joint_support_coordinates']=[coordinate_names[6+i] for i in unresolved];summary['unidentified_support_peak_Nm']={coordinate_names[6+i]:float(abs(solution['unidentified_joint_support_moment_Nm'][:,i]).max()) for i in unresolved};summary['foot_intrinsic_and_fascia_force_solution_complete']=False;summary['warnings'].append('MTP requirements remain explicitly unidentified joint-support moments. Extrinsic toe-muscle force histories are not released as validated tissue loads.')
    summary['model_muscle_elements']=muscle_count;summary['force_unknowns']=nm;summary['motion_verification']=verified
    task_setup=json.loads((folder/'setup.json').read_text()) if (folder/'setup.json').exists() else {}
    summary['movement_kind']=args.case;summary['scope']=task_setup.get('scope',summary['scope'])
    summary['selection_mode']=args.selection
    if args.selection=='batch':summary['force_sharing']='Whole-history integral of normalized force magnitude and derivative squared; GH cone 20 degrees; no strength ceiling or antagonist floor. This is not metabolic energy.'
    if verified.get('status')=='measured_retarget_diagnostic':
        summary['warnings'].append('Measured replay has a disclosed floating-root residual. No residual support is applied; internal-joint moment equations alone select these conditional tensions. These are not a complete exact whole-body equilibrium solution to the measured data.')
        summary['measured_ground_reaction_and_COP']=True
    if verified.get('status')=='prescribed_projected_contact_diagnostic':
        summary['warnings'].extend(verified['limitations'])
        summary['exact_whole_body_equilibrium']=False
        summary['measured_ground_reaction_and_COP']=False
    summary['root_force_residual_max_N']=float(abs(arrays['root_residual'][:,:3]).max());summary['root_moment_residual_max_Nm']=float(abs(arrays['root_residual'][:,3:]).max());summary['NE_crosscheck_max_SI']=float(arrays['NE_crosscheck'].max());summary['scope']='Full motion, including flight and landing where present' if args.full_cycle else ('Propulsion only' if args.case=='jump' else 'Squat ascent only');summary['common_profile']=verified
    np.savez_compressed(folder/'tensions.npz',times=ts,source_indices=indices,reference_N=refs,GH_reaction_N=reaction,**solution,**arrays);(folder/'tension_summary.json').write_text(json.dumps(summary,indent=2));print('JUMP_TENSIONS_COMPLETE',summary['checks'],summary['peak_element'],summary['peak_element_N'],flush=True)

if __name__=='__main__':main()
