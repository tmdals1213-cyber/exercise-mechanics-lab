"""Independent exported-value, path-endpoint and pose registration checks."""
import json,hashlib
from pathlib import Path
import numpy as np
from jump_rig import JumpRig
from tension_reference import scale_parameters,reaction_cone
from audit_jump_muscles import GROUPS
from tension_groups import REGISTRY,matches
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/tension_visualization'

def main():
    data=json.loads((OUT/'data.json').read_text(encoding='utf-8'));ops=dict(np.load(OUT/'operators.npz'));geometry=dict(np.load(OUT/'geometry.npz'));source=dict(np.load(ROOT/'results/jump_validation/vertical/acceleration_2400hz/motion.npz'));cfg=json.loads((ROOT/'scenarios/tension_reference.json').read_text(encoding='utf-8'));rig=JumpRig();b=rig.b;m,d=b.m,b.d
    for name,h in data['input_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    assert data['solver_sha256']==hashlib.sha256((ROOT/'scripts/tension_reference.py').read_bytes()).hexdigest();assert data['builder_sha256']==hashlib.sha256((ROOT/'scripts/build_tension_visualization.py').read_bytes()).hexdigest()
    assigned=[i for g in data['groups'] for i in g['indices']];assert sorted(assigned)==list(range(m.nu))
    for key,label,prefixes,preferred in REGISTRY:
        actual=sorted(i for g in data['groups'] if g['key']==key for i in g['indices']);assert actual==[i for i,n in enumerate(b.names) if matches(n,prefixes)]
    required=['gastrocnemius','soleus','tibialis_anterior','tibialis_posterior','fibularis','toe_flexors','toe_extensors','abdominals','spinal_extensors','quadratus_lumborum']
    assert all(sum(g['key']==key for g in data['groups'])==2 for key in required)
    assert 'Foot intrinsics' in data['unmodeled_groups'] and data['coverage']['foot_intrinsic_actuators']==0
    definition=json.loads((ROOT/'results/jump_validation/vertical/acceleration_2400hz/contact_definition.json').read_text('utf-8'));source_loads=np.load(ROOT/'results/jump_validation/vertical/acceleration_2400hz/loads.npz');contact_error=0.
    for side in ['r','l']:
        ids=[j for j,p in enumerate(definition['points']) if p['body']=='calcn_'+side];expected=source_loads['contact_force_world_N'][geometry['source_indices']][:,ids].sum(axis=1);contact_error=max(contact_error,float(abs(expected-np.array(data['ground_reaction'][side])).max()))
    assert contact_error<.000051
    coverage={'status':'passed',**data['coverage'],'group_side_records':len(data['groups']),'required_lower_leg_and_core_groups':required,'ground_reaction_export_error_N':contact_error,'contact_force_is_not_plantar_muscle_tension':True,'foot_intrinsics_not_fabricated':True,'latissimus_review_preserved':True,'display_coverage_is_not_biological_validation':True};(OUT/'coverage_checks.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
    assert np.array_equal(geometry['body_xpos'],source['body_xpos'][geometry['source_indices']]);assert np.array_equal(geometry['body_xmat'],source['body_xmat'][geometry['source_indices']])
    moment_error=0.;group_error=0.;cone_error=0.;angle_max=0.
    for profile,settings in cfg['profiles'].items():
        z=dict(np.load(OUT/(profile+'.npz')));ref,gain,passive,scale=scale_parameters(b.names,ops['reference'],ops['gain'],ops['passive'],settings,GROUPS);F=z['force_N'];assert np.all(z['active_force_N']>=0);assert np.allclose(F,z['active_force_N']+passive);assert np.allclose(z['active_force_N'],gain*z['unbounded_effort_coefficient'],atol=1e-7)
        moment_error=max(moment_error,float(abs(np.einsum('tqn,tn->tq',ops['R'],F)-ops['target']).max()))
        for k,(S,o,a) in enumerate(zip(ops['S'],ops['offset'],ops['axes'])):
            C,rhs=reaction_cone(S,o,a);cone_error=max(cone_error,float((C@F[k]-rhs).max()));reaction=o-S@F[k];angles=np.rad2deg(np.arccos(np.clip(np.sum(a*reaction,axis=1)/np.maximum(np.linalg.norm(reaction,axis=1),1e-12),-1,1)));angle_max=max(angle_max,float(angles.max()))
        for j,g in enumerate(data['groups']):
            if g['status']!='computed':
                assert g['display_id']==0 and all(data['profiles'][profile][key][j] is None for key in ['total','active','passive']);continue
            for key,raw in [('total',F),('active',z['active_force_N']),('passive',passive)]:group_error=max(group_error,float(abs(raw[:,g['indices']].sum(1)-np.array(data['profiles'][profile][key][j])).max()))
    node_error=0.;endpoint_error=0.
    for frame in [0,27,37,len(data['times'])-1]:
        i=int(geometry['source_indices'][frame]);b.pose(source['qpos'][i]);
        for side in ['r','l']:
            for label,joint in [('shoulder','elv_angle'),('elbow','elbow_flexion'),('wrist','flexion'),('hip','hip_flexion'),('knee','knee_angle'),('ankle','ankle_angle')]:
                j=data['node_names'].index(label+'_'+side);node_error=max(node_error,float(abs(geometry['nodes'][frame,j]-d.xanchor[b.joints[joint+'_'+side]]).max()))
        for j,g in enumerate(data['groups']):
            aid=b.names.index(g['representative']);tid=int(m.actuator_trnid[aid,0]);start=int(d.ten_wrapadr[tid]);count=int(d.ten_wrapnum[tid]);points=d.wrap_xpos.reshape(-1,3)[start:start+count];endpoint_error=max(endpoint_error,float(abs(geometry['paths'][frame,j,0]-points[0]).max()),float(abs(geometry['paths'][frame,j,-1]-points[-1]).max()))
    if max(moment_error,cone_error)>.01 or group_error>.000051 or max(node_error,endpoint_error)>1e-8 or angle_max>20.01:raise RuntimeError('Visualization value/geometry registration failed')
    result={'status':'passed','frames':len(data['times']),'profiles':len(cfg['profiles']),'group_side_regions':len(data['groups']),'moment_balance_max_error_Nm':moment_error,'GH_max_angle_deg':angle_max,'GH_linear_cone_max_violation_N':max(0.,cone_error),'exported_group_sum_max_rounding_error_N':group_error,'proxy_joint_anchor_max_error_m':node_error,'representative_path_endpoint_max_error_m':endpoint_error,'body_poses_identical_to_analysis':True,'review_regions_are_null_not_zero':True,'source_hashes_matched':True,'strength_ceiling_enforced':False,'selection_mode':'causal magnitude and previous-tension continuity','scope':'Data mapping and mechanical consistency of a development reference; existing motion/path model limitations remain.'};(OUT/'value_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
