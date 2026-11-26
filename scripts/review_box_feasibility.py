"""Record rejected low-box trials; never promote their power or muscle forces."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'


def main():
    folder=OUT/'box_40_low_trial';trials=[]
    for filename in ['summary_slow_rejected.json','summary_short_rejected.json','summary.json']:
        path=folder/filename
        if path.exists():
            s=json.loads(path.read_text())
            trials.append(dict(source=str(path.relative_to(ROOT)).replace('\\','/'),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status=s['status'],
                failed_checks=[n for n,v in s['gates'].items() if not v],checks=s['checks'],
                release=s.get('takeoff_release'),controller=s.get('controller_profile')))
    latest=trials[-1]
    assert latest['status']=='failed_not_released'
    report=dict(status='quick_feasibility_did_not_pass',review_date='2026-11-20',
        box_height_m=.4,accepted_box_motion_count=0,box_muscle_comparison_ready=False,
        box_landing_solved=False,invalid_trial_outputs_in_comparison=False,
        corrected_liftoff_logic='Allow vanishing-normal-force liftoff from either flat-foot or toe support; preserve flight mode if liftoff precedes nominal heel-rise time; require separating foot motion; no velocity reset.',
        remaining='Current acceleration-level controller does not simultaneously satisfy box clearance, foot placement/orientation and whole-body posture. Coordinated trajectory development is required.',
        muscle_force_comparison=dict(status='deferred_until_valid_motion',
            planned_boxes_cm=[40,60,80],squat_bar_masses_kg=[60,100,140],
            metrics=['peak muscle-group tension N','mean tension over propulsion/ascent N','tension-time integral N s','generic force-reference ratio'],
            force_ratio_is_activation=False,box_height_is_not_COM_rise_or_power=True,
            separate_phases=['ground propulsion','free flight','landing, once solved']),
        trials=trials,
        implementation_sha256={n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest()
            for n in ['try_low_box_jump.py','build_contact_jump.py','build_box_jump_power.py']})
    (OUT/'box_feasibility_review.json').write_text(json.dumps(report,indent=2))
    specpath=ROOT/'scenarios/jump_box_squat_power.json';spec=json.loads(specpath.read_text())
    case=next(c for c in spec['conditions'] if c['id']=='box_40')
    case.update(status='development_trial_rejected',power_result_W=None,
        failed_checks=latest['failed_checks'],landing_included=False,
        development_review='results/power_comparison/box_feasibility_review.json')
    spec['muscle_comparison']=report['muscle_force_comparison']
    specpath.write_text(json.dumps(spec,indent=2));(OUT/'comparison_setup.json').write_text(json.dumps(spec,indent=2))
    print('LOW_BOX_REVIEW_RECORDED',latest['failed_checks'])


if __name__=='__main__':main()
