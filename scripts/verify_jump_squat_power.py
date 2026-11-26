"""Mechanical, grasp-invariance, convergence and encoded-video checks."""
from pathlib import Path
import hashlib,json,platform
from datetime import datetime,timezone
import numpy as np
import imageio_ffmpeg
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'


def verify_component_provenance():
    reference=json.loads((OUT/'muscle_reference_review.json').read_text())
    assert reference['review']==json.loads((ROOT/'scenarios/muscle_force_literature.json').read_text())
    assert not reference['changes_to_force_solution'] and not reference['reference_used_as_force_cap']
    rows=[]
    for key in ['cmj','squat_60','squat_100','squat_140']:
        folder=OUT/('cmj_analysis' if key=='cmj' else key);source=folder if key=='cmj' else folder/'analysis';visual=source/'visualization'
        audit=json.loads((folder/'joint_components_audit.json').read_text())
        assert audit['status']=='decomposition_checks_passed' and not audit['force_solution_changed']
        for name,field in [('joint_reactions.npz','source_joint_reactions_sha256'),('geometry.npz','source_geometry_sha256')]:
            assert hashlib.sha256((visual/name).read_bytes()).hexdigest()==audit[field]
        ref_hash=reference['CMJ_force_file_sha256'] if key=='cmj' else reference['force_file_sha256'][key]
        assert hashlib.sha256((source/'tensions.npz').read_bytes()).hexdigest()==ref_hash
        assert audit['checks']['known_compression_distraction_shear_cases'] and audit['checks']['rotation_invariance']
        rows.append(dict(case=key,status=audit['status'],checks=audit['checks']))
    return rows


def main():
    report=json.loads((OUT/'comparison_results.json').read_text())
    normalization=json.loads((OUT/'force_normalization_audit.json').read_text())
    assert not normalization['force_values_changed'] and not normalization['physiological_capacity_validated']
    conditional=[]
    for key in ['cmj','squat_60','squat_100','squat_140']:
        folder=OUT/('cmj_analysis' if key=='cmj' else key)
        source=folder if key=='cmj' else folder/'analysis'
        audit=json.loads((folder/'tension_audit.json').read_text())
        assert audit['status']=='passed' and all(audit['gates'].values())
        assert not audit['muscle_tension_is_independently_validated'] and not audit['strength_cap_used']
        force_hash=hashlib.sha256((source/'tensions.npz').read_bytes()).hexdigest()
        assert force_hash==audit['force_file_sha256']
        assert hashlib.sha256((source/'motion.npz').read_bytes()).hexdigest()==audit['source_motion_sha256']
        normalized=next(r for r in normalization['cases'] if r['case']==key)
        assert normalized['force_file_sha256']==force_hash
        if key!='cmj':
            mapping=json.loads((folder/'tension_render_mapping.json').read_text())
            assert {'latissimus','trapezius'} <= {r['group'] for r in mapping['meshes']}
            assert all(n>0 for n in mapping['joint_face_counts'])
            assert mapping['values_are_conditional'] and not mapping['muscle_activation_or_stress']
        conditional.append(dict(case=key,status=audit['status'],force_file_sha256=force_hash,
                                moment_balance_max_error_Nm=audit['moment_balance_max_error_Nm']))
    grip=json.loads((ROOT/'models/grips/high_bar_28mm.json').read_text())
    fixture=np.load(OUT/'grip_fixture/motion.npz');skin=np.load(OUT/'grip_fixture/skin_bind.npz')
    active=np.flatnonzero(skin['weights'].sum(0)>1e-8)
    def relative(d):
        R=np.swapaxes(d['bar_rotation'],-1,-2)
        p=np.einsum('tij,tbj->tbi',R,d['body_xpos'][:,active]-d['bar_position_m'][:,None])
        r=np.einsum('tij,tbjk->tbik',R,d['body_xmat'][:,active].reshape(-1,len(active),3,3))
        return p,r
    p0,r0=relative(fixture);grasp=[]
    for row in report['squats']:
        assert row['all_mechanics_gates_passed'];folder=OUT/f"squat_{row['bar_mass_kg']}";d=np.load(folder/'motion.npz')
        assert not row['root_reserve_applied'] and not row['contact_force_cap_used'] and not row['strength_cap_used']
        p,r=relative(d);pe=float(abs(p-p0).max());re=float(abs(r-r0).max())
        assert pe<1e-8 and re<1e-8
        assert max(d['grasp_error_m'])<1e-4
        assert np.allclose(d['COM_power_W'],d['GRF_N'][:,2]*d['system_velocity_m_s'][:,2],rtol=0,atol=1e-7)
        grasp.append(dict(bar_mass_kg=row['bar_mass_kg'],all_skin_influence_bones_relative_to_bar_max_translation_error_m=pe,
                          max_rotation_matrix_error=re,all_frames_grasp_error_m=float(max(d['grasp_error_m']))))
        visual=json.loads((folder/'display_grip.json').read_text())
        assert visual['native_finger_surfaces_replaced'] and not visual['hand_forces_recomputed'] and not visual['inertias_changed']
        assert all(f['shaft_clearance_lower_bound_m']>0 for f in visual['fingers'])
        assert all(f['wrap_angle_deg']>=180 for f in visual['fingers'] if f['digit']!=1)
    assert all(a['inside_axis_samples']==0 and a['clearance_lower_bound_with_sampling_m']>0 for a in grip['surface_audit'])
    coarse=np.load(OUT/'squat_100/motion_125hz.npz');fine=np.load(OUT/'squat_100/motion.npz');conv={}
    for key in ['COM_power_W','GRF_N','net_joint_moments_Nm','net_joint_powers_W']:
        c=coarse[key];f=fine[key];shape=c.shape;flat_f=f.reshape(len(f),-1);flat_c=c.reshape(len(c),-1)
        interp=np.stack([np.interp(coarse['times'],fine['times'],flat_f[:,j]) for j in range(flat_f.shape[1])],axis=1)
        rms=float(np.sqrt(np.mean((flat_c-interp)**2))/max(1.,np.sqrt(np.mean(interp**2))))
        peak=float(abs(np.max(abs(c))-np.max(abs(f)))/max(1.,np.max(abs(f))))
        assert max(rms,peak)<.02
        conv[key]=dict(relative_curve_RMS_difference=rms,relative_peak_difference=peak)
    media=[]
    for key,row in report['media'].items():
        assert row['credits']['frames']==90 and row['credits']['duration_s']==3
        assert row['source_motion_video_frames']+row['credits']['frames']==row['frames']
        assert not row['credits']['motion_statistics_include_credits']
        stream=imageio_ffmpeg.read_frames(str(OUT/row['file']),pix_fmt='rgb24');meta=next(stream)
        assert tuple(meta['size'])==(1080,1920) and meta['fps']==30
        count=0;footer=0
        for raw in stream:
            a=np.frombuffer(raw,dtype=np.uint8).reshape(1920,1080,3);footer=max(footer,int(np.max(abs(a[1536:].astype(int)-[13,23,38]))));count+=1
        assert footer<=3 and count==row['frames']
        credit=np.asarray(Image.open(OUT/row['credits']['file']).convert('RGB'))
        credit_error=float(np.mean(abs(a.astype(float)-credit.astype(float))))
        assert credit_error<5.,('Encoded credit differs',key,credit_error)
        media.append(dict(condition=key,frames=count,footer_max_channel_error=footer,display_speed=row['display_speed'],
                          end_credit_frames=row['credits']['frames'],last_credit_frame_mean_channel_error=credit_error))
    result=dict(status='passed',mechanically_checked_squats=3,human_movement_accuracy_bound_established=False,
                source_joint_ranges_expanded=False,native_finger_surfaces_replaced_for_display=True,display_hand_geometry_is_not_a_hand_force_model=True,
                static_grasp_surface_audit=grip['surface_audit'],grasp_invariance_over_full_motions=grasp,
                convergence_125_to_250_Hz=conv,convergence_gate_is_engineering_choice=.02,media=media,
                box_dynamic_comparison_completed=False,squat_muscle_tension_validated=False,
                conditional_muscle_forces_computed=True,conditional_tension_checks=conditional,
                normalization_changes_forces=False,reference_ratio_is_activation=False,
                joint_component_checks=verify_component_provenance(),literature_screening_is_biological_validation=False)
    development=report.get('box_development_review')
    if development:
        assert not development['released_in_comparison']
        result['box_development_trial']=development
    (OUT/'comparison_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    spec_path=ROOT/'scenarios/jump_box_squat_power.json';spec=json.loads(spec_path.read_text())
    for case in spec['conditions']:
        if case['movement']=='high_bar_back_squat':
            row=next(r for r in report['squats'] if r['bar_mass_kg']==case['bar_mass_kg'])
            case.update(status='mechanically_checked_prescribed_motion',power_result_W=row['peak_COM_vertical_power_W'],ascent_duration_s=row['prescribed_timing_s']['ascent'],human_motion_validated=False)
        elif case['id']=='box_60' and report.get('box_development_review',{}):
            development=report['box_development_review']
            if development['failed_checks']:
                case.update(status='development_trial_rejected',power_result_W=None,failed_checks=development['failed_checks'],landing_included=False)
    spec['status']='CMJ_reference_and_prescribed_squats_ready_box_dynamics_pending'
    spec['timing']['source']='Measured CMJ reference; explicit per-load squat model tempos, not predicted personal speeds.'
    spec_path.write_text(json.dumps(spec,indent=2),encoding='utf-8');(OUT/'comparison_setup.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--components-only',action='store_true');args=parser.parse_args()
    if args.components_only:
        result=json.loads((OUT/'comparison_verification.json').read_text())
        result.update(joint_component_checks=verify_component_provenance(),literature_screening_is_biological_validation=False)
        (OUT/'comparison_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('COMPONENT_AND_REFERENCE_PROVENANCE_PASSED')
    else:main()
