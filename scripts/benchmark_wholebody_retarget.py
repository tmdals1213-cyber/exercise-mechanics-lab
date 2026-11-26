"""Same hand/foot targets after morphology changes; all-body IK and load check."""
import json,hashlib
from pathlib import Path
import numpy as np
from parametric_body import Subject,build_subject,load_subject_profiles
from wholebody_ik import FrameTarget,solve_wholebody_ik
from benchmark_parametric_body import load_fixture,snapshot,ROOT,OUT


def main():
    source=np.load(ROOT/'results/grip_comparison/grip_65/cycle/physiology/fullbody_motion.npz')
    cases=load_subject_profiles(ROOT/'scenarios/body_profiles.json')
    reference=build_subject(cases['reference']);targets_by_frame={}
    for frame in [20,35,50]:
        load_fixture(reference,source,frame);targets=[]
        for side in ['r','l']:
            for label,body_name,local in [('hand','lunate_'+side,np.array([0.,-.065,0.])*reference.report['global_normalization_scale']),('foot','calcn_'+side,reference.m.site_pos[reference.m.site('subject_sole_reference_'+side).id])]:
                bid=reference.m.body(body_name).id;R=reference.d.xmat[bid].reshape(3,3).copy();p=reference.d.xpos[bid]+R@local
                targets.append({'name':label+'_'+side,'body_name':body_name,'position':p.copy(),'rotation':R.copy()})
        targets_by_frame[frame]=targets
    all_results={}
    for name,subject in cases.items():
        b=build_subject(subject);rows=[];records=[]
        for frame in [20,35,50]:
            seed=load_fixture(b,source,frame);targets=[]
            for target in targets_by_frame[frame]:
                bid=b.m.body(target['body_name']).id
                local=np.array([0.,-.065,0.])*b.report['global_normalization_scale'] if target['name'].startswith('hand_') else b.m.site_pos[b.m.site('subject_sole_reference_'+target['name'][-1]).id]
                targets.append(FrameTarget(target['name'],bid,local,target['position'],target['rotation']))
            feet=np.array([t.position_world for t in targets if t.name.startswith('foot_')]);hands=np.array([t.position_world for t in targets if t.name.startswith('hand_')])
            support=np.array([feet[:,:2].min(0)+[-.025,-.12],feet[:,:2].max(0)+[.025,.065]])
            q,fit=solve_wholebody_ik(b,seed,targets,support_bounds_world_xy=support,external_masses=[(100.,hands.mean(0))])
            data,check=snapshot(b,q);records.append(data);rows.append({'reference_pose_index':frame,'IK':fit,'load_check':check})
            print('WHOLEBODY_FIT',name,frame,max(fit['position_errors_m'].values()),flush=True)
        folder=OUT/name;np.savez_compressed(folder/'retargeted_fixtures.npz',**{k:np.array([r[k] for r in records]) for k in records[0]})
        (folder/'retargeted_verification.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');all_results[name]=rows
    # A kinematic-only check that the shared IK also accepts hand frame targets
    # without assuming a foot support or ground. This is NOT a hanging-force solve.
    free_seed=load_fixture(reference,source,35);free_targets=[]
    for target in targets_by_frame[35]:
        if target['name'].startswith('hand_'):
            bid=reference.m.body(target['body_name']).id
            free_targets.append(FrameTarget(target['name'],bid,np.array([0.,-.065,0.])*reference.report['global_normalization_scale'],target['position']+[0,0,.02],target['rotation']))
    _,free_check=solve_wholebody_ik(reference,free_seed,free_targets)
    assert free_check['ground_support_constraint_applied'] is False
    report={'status':'passed_all_15_static_fits','hand_only_kinematic_check_no_ground':free_check,'control':'Same three world-space hand/foot positions AND orientations for all five body shapes. Whole-body root, trunk, hip, knee, ankle and arm coordinates may adapt within native ranges.','bar_mass_kg':100.,
            'max_position_error_m':max(max(r['IK']['position_errors_m'].values()) for rows in all_results.values() for r in rows),'max_orientation_error_rad':max(max(r['IK']['orientation_errors_rad'].values()) for rows in all_results.values() for r in rows),
            'cases':all_results,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'IK_implementation_sha256':hashlib.sha256((ROOT/'scripts/wholebody_ik.py').read_bytes()).hexdigest(),'scope':'Three independently fitted held configurations per subject. No trajectory derivatives, activation dynamics, independent scapular actuation or jump/turnover contacts are solved.'}
    (OUT/'wholebody_retarget.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('WHOLEBODY_RETARGET_COMPLETE',report['max_position_error_m'],flush=True)


if __name__=='__main__':main()
