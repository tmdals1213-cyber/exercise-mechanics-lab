"""Prepare one anatomy/force data record, including joint cut reactions."""
import json,hashlib,os
import numpy as np
from jump_model import create_rig
from conservative_ellipsoid_paths import ConservativeEllipsoidPaths
from mechanics_engine_adapter import sample
from muscle_mechanics import tendon_body_wrenches
from newton_euler_reference import accumulate,at_point
from tension_groups import REGISTRY,matches
from jump_rig import ROOT


def main():
    source=ROOT/os.environ.get('EXERCISE_JUMP_SOURCE','results/contact_jump_scapula/600hz');out=ROOT/os.environ.get('EXERCISE_JUMP_VISUALS','results/jump_contact_visualization');out.mkdir(parents=True,exist_ok=True);motion=np.load(source/'motion.npz');tension=np.load(source/'tensions.npz');summary=json.loads((source/'tension_summary.json').read_text());setup=json.loads((source/'setup.json').read_text());definition=json.loads((source/'contact_definition.json').read_text());rig=create_rig(setup['rig_type'],setup);b,m,d=rig.b,rig.m,rig.d;paths=ConservativeEllipsoidPaths(rig);names=summary['names'];indices=tension['source_indices'];F=tension['force_N'];muscle_count=m.nu+len(paths.names)
    registry=REGISTRY+[('trapezius','Trapezius',('Trapezius',),'TrapeziusScapula_S'),('serratus','Serratus anterior',('Serratus',),'SerratusAnterior_I'),('rhomboids','Rhomboids',('Rhomboideus',),'Rhomboideus_I'),('levator','Levator scapulae',('Levator',),'LevatorScapulae'),('pectoralis_minor','Pectoralis minor',('PectoralisMinor',),'PectoralisMinor')]
    assert hashlib.sha256((source/'motion.npz').read_bytes()).hexdigest()==summary['source_motion_sha256'],'Force solution uses a different motion'
    groups=[];assigned=[]
    for key,label,prefix,representative in registry:
        for side in ['r','l']:
            ids=[j for j,n in enumerate(names[:muscle_count]) if matches(n,prefix) and ('l' if n.endswith('_l') else 'r')==side]
            if not ids:continue
            released=key not in ['toe_flexors','toe_extensors']
            groups.append(dict(key=key,label=label,side=side,indices=ids,display_id=len(groups)+1 if released else 0,released=released));assigned.extend(ids)
    assert len(set(assigned))==muscle_count and len(assigned)==muscle_count and len(groups)<=63
    joints=[dict(key=key+'_'+side,joint=joint+'_'+side,label=label+' '+side.upper()) for key,joint,label in [('shoulder','shoulder_rot','Shoulder'),('elbow','elbow_flexion','Elbow'),('wrist','flexion','Wrist'),('hip','hip_flexion','Hip'),('knee','knee_angle','Knee'),('ankle','ankle_angle','Ankle')] for side in ['r','l']]
    node_names=['pelvis']+[j['key'] for j in joints];nodes=[];reactions=[];normal_work=[];normal_gaps=[]
    for k,index in enumerate(indices):
        q,v,a=motion['qpos'][index],motion['qvel'][index],motion['qacc'][index];contacts=[(p['body_id'],motion['contact_positions_m'][index,j],motion['contact_forces_N'][index,j],motion['contact_free_moments_Nm'][index,j] if 'contact_free_moments_Nm' in motion else np.zeros(3)) for j,p in enumerate(definition['points'])];mech=sample(rig,q,v,a,contacts)
        native=tendon_body_wrenches(b,F[k,:m.nu]);native[:,3:]+=np.cross(d.xipos,native[:,:3]);paths.operators();muscles=native+np.einsum('bcn,n->bc',paths.unit_wrenches,F[k,m.nu:muscle_count]);C,points,gaps=rig.contact_operators(q)
        for j,(bid,cb,p,n,_) in enumerate(points):
            force=F[k,muscle_count+j]*n
            for body,sign in [(bid,1),(cb,-1)]:muscles[body,:3]+=sign*force;muscles[body,3:]+=sign*np.cross(p,force)
        # Unidentified intrinsic-foot/fascial support is an internal joint couple,
        # not an invented extrinsic muscle force or an external support.
        for side in ['r','l']:
            name='mtp_angle_'+side;jid=b.joints[name];child=m.jnt_bodyid[jid];parent=m.body_parentid[child]
            residual=tension['unidentified_joint_support_moment_Nm'][k,summary['coordinate_names'].index(name)]
            couple=residual*d.xaxis[jid];muscles[child,3:]+=couple;muscles[parent,3:]-=couple
        cuts=accumulate(m.body_parentid,mech['net_wrenches']-muscles);row=[];xyz=[d.xpos[m.body('pelvis').id].copy()]
        for joint in joints:
            jid=b.joints[joint['joint']];bid=m.jnt_bodyid[jid];point=d.xanchor[jid];row.append(at_point(cuts[bid],point));xyz.append(point.copy())
        nodes.append(xyz);reactions.append(row);normal_work.append(float(np.max(abs((C.T@v)*F[k,muscle_count:]))));normal_gaps.append(float(abs(gaps).max()))
        print('CONTACT_VISUAL_DATA',k+1,len(indices),flush=True)
    reactions=np.array(reactions);joint_N=np.linalg.norm(reactions[:,:,:3],axis=2);group_N=np.array([F[:,g['indices']].sum(1) for g in groups]);ts=tension['times'];duration=ts[-1]-ts[0];b.pose(b.q0)
    rest_nodes=np.array([d.xpos[m.body('pelvis').id].copy()]+[d.xanchor[b.joints[j['joint']]].copy() for j in joints])
    np.savez_compressed(out/'geometry.npz',times=ts,body_xpos=motion['body_xpos'][indices],body_xmat=motion['body_xmat'][indices],rest_body_xpos=d.xpos.copy(),rest_body_xmat=d.xmat.copy(),rest_nodes=rest_nodes,nodes=np.array(nodes),contact_sides=np.array([0 if p['body'].endswith('_r') else 1 for p in definition['points']]),contact_forces_N=motion['contact_forces_N'][indices],contact_positions_m=motion['contact_positions_m'][indices])
    np.savez_compressed(out/'joint_reactions.npz',times=ts,joint_wrenches=reactions,joint_force_N=joint_N)
    data=dict(times=ts.tolist(),groups=groups,scale=b.report['global_normalization_scale'],body_names=setup['body_names'],node_names=node_names,names=names[:muscle_count],muscle_count=muscle_count,contact_variable_count=4,group_tension_N=group_N.tolist(),joint_force_N=joint_N.T.tolist(),joints=joints,takeoff_s=setup['takeoff_s'],scope='Countermovement, propulsion and flight; ends before landing',human_force_prediction_validated=False,GRF_N=motion['force_world_N'][indices].tolist(),mean_group_N=(np.trapezoid(group_N,ts,axis=1)/duration).tolist(),max_group_N=group_N.max(1).tolist(),mean_joint_N=(np.trapezoid(joint_N,ts,axis=0)/duration).tolist(),max_joint_N=joint_N.max(0).tolist(),scapular_contact_max_power_W=max(normal_work),scapular_contact_max_surface_residual=max(normal_gaps),force_selection='One conditional solution, not measured recruitment',mechanics=json.loads((source/'verification.json').read_text())['checks'],source_summary='../contact_jump_scapula/600hz/tension_summary.json')
    apex=int(np.argmax(motion['com'][indices,2]));data.update(poster_index=apex,preview_indices=sorted(set([0,int(np.argmin(motion['com'][indices,2])),int(np.argmax(motion['force_world_N'][indices,2])),apex,len(ts)-1])),countermovement_end_s=setup['countermovement_end_s'],heel_rise_start_s=setup['heel_rise_start_s'],motion_quality=json.loads((source/'motion_quality.json').read_text()),unresolved_plantar_support=True)
    data.update(mass_kg=setup.get('mass_kg',90.),height_m=setup.get('height_m',1.8),measured_replay=setup.get('ground_reaction_and_COP_are_inputs',False),scope=setup['scope'],source_summary=os.path.relpath(source/'tension_summary.json',out).replace('\\','/'),motion_verification=json.loads((source/'verification.json').read_text()))
    if data['measured_replay']:data['poster_index']=int(np.argmax(motion['force_world_N'][indices,2]))
    (out/'data.json').write_text(json.dumps(data,separators=(',',':')));(out/'data.js').write_text('window.JUMP_CONTACT_DATA='+json.dumps(data,separators=(',',':'))+';');(out/'CREDITS.txt').write_text((ROOT/'assets/fullbody/CREDITS.md').read_text(encoding='utf-8') if (ROOT/'assets/fullbody/CREDITS.md').exists() else 'Anatomical surface assets: see project assets/fullbody credits. MyoHub/BP3D CC BY 4.0; Z-Anatomy supplemental surfaces CC BY-SA. Shoulder attachments: Seth et al. 2019, doi:10.3389/fnbot.2019.00090; RMR solver v1.1 source and LICENSE_data in models/rmr.',encoding='utf-8')
    print('VISUAL_DATA_COMPLETE',muscle_count,'muscles',len(groups),'groups','contact power',max(normal_work),flush=True)

if __name__=='__main__':main()
