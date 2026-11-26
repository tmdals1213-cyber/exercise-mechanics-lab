"""Publish honest development status, separate from accepted social releases."""
from pathlib import Path
import json,html
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'deliverables/jump_motion_development';out.mkdir(parents=True,exist_ok=True)
    quiet=json.loads((ROOT/'results/physical_jump_task/native_quiet_support/verification.json').read_text())
    reference=json.loads((ROOT/'results/measured_jump_analysis/s10_CMJ_t1_full_cycle/full_cycle_review.json').read_text())
    flight=json.loads((ROOT/'results/physical_jump_task/floor_jump_v5/flight_verification.json').read_text())
    attempts=[]
    for folder in sorted((ROOT/'results/physical_jump_task').glob('native_floor_landing_v*'),key=lambda p:int(p.name.rsplit('v',1)[1])):
        p=folder/'landing_status.json'
        if p.exists():attempts.append(dict(case=folder.name,**json.loads(p.read_text())))
    report=dict(status='full_cycle_reference_available_generated_landing_incomplete',reviewed_on='2026-11-23',
                scope='Complete measured-motion retarget and native landing-controller development; no landing muscle-force release.',
                quiet_support=quiet,measured_reference=reference,generated_flight=flight,native_landing_attempts=attempts,
                generated_landing_accepted=False,box_jump_accepted=False,muscle_force_analysis_allowed=False,
                next_required='Match measured contact timing and sole roll; verify realized COP, friction and momentum balance before extending generated jumps to box heights.',
                references=[dict(title='Toe pressure and dynamic standing balance',url='https://pubmed.ncbi.nlm.nih.gov/8919397/'),
                            dict(title='Ankle torque, pressure-centre shift and angular momentum',url='https://pubmed.ncbi.nlm.nih.gov/24524989/'),
                            dict(title='MuJoCo contact dynamics',url='https://mujoco.readthedocs.io/en/stable/computation.html'),
                            dict(title='Measured CMJ dataset',url='https://doi.org/10.6084/m9.figshare.28890545')])
    (out/'development_status.json').write_text(json.dumps(report,indent=2))
    rows=''.join('<tr><td>'+html.escape(a['case'])+'</td><td>Rejected</td><td>'+html.escape(a.get('failure') or 'Support/velocity gate failed')+'</td></tr>' for a in attempts)
    links=' · '.join(f'<a href="{html.escape(s["url"])}">{html.escape(s["title"])}</a>' for s in report['references'])
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Jump motion development</title>
<style>body{{max-width:1000px;margin:30px auto;padding:24px;background:#0d1726;color:#edf3fb;font:17px/1.6 system-ui}}a{{color:#8ce2d5}}video{{max-height:78vh;max-width:100%;display:block;margin:20px auto}}section{{padding:18px;background:#152338;margin:18px 0;border-radius:12px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #384a63;text-align:left}}small{{color:#a6bcd1}}</style>
<h1>Jump motion development</h1><p>Measured full-cycle motion review. Generated landing remains unaccepted.</p><video src="jump_motion_review.mp4" controls loop playsinline></video>
<section><h2>Foot support</h2><p>Heel, forefoot and toe contacts; independent ankle, subtalar and MTP coordinates. All toes share one MTP hinge per foot. Desired COP is restricted to the sole polygon. Actual COP is recorded from the contact solver separately. Foot-centre distance alone is not a stability test: COM velocity, available contact region, unilateral pressure, friction and angular momentum matter.</p>
<p>Quiet support: mean vertical ground force {quiet['mean_GRF_last_second_N'][2]:.2f} N versus weight {quiet['expected_weight_N']:.2f} N. Final COM speed {sum(x*x for x in quiet['final_COM_velocity_m_s'])**.5:.3f} m/s. This is a quiet-support check only.</p></section>
<section><h2>What is and is not verified</h2><p>The video retargets measured countermovement-jump motion through landing and recovery onto the existing anatomical model. It does not show a successfully generated or athlete-specific box jump.</p>
<p>Active-stance vertical residual: {reference['active_stance_vertical_RMS_bodyweights']*100:.2f}% of body weight. The COM displacement is registered to the reference model, so force agreement is inherited from that kinematic reference; it is not independent anatomical validation. Landing residuals are reported separately in <a href="development_status.json">the audit</a>.</p>
<p>The generated flight passes floating-root balance and airborne momentum checks up to first surface contact. Later free-flight samples bracket contact and are not valid supported motion. Native landing attempts are rejected when support recovery fails. No hidden root forces, force ceilings or muscle-force release.</p></section>
<details><summary>Rejected native landing attempts</summary><table><tr><th>Case</th><th>Status</th><th>Reason</th></tr>{rows}</table></details>
<p><small>{links}</small></p></html>'''
    (out/'index.html').write_text(text,encoding='utf-8')
    state=ROOT/'results/reference_box_jump/continuation_state.json'
    state.write_text(json.dumps(dict(status=report['status'],active_reference='results/measured_jump_analysis/s10_CMJ_t1_full_cycle',active_generated_flight='results/physical_jump_task/floor_jump_v5',active_contact_experiment='results/physical_jump_task/'+attempts[-1]['case'],generated_landing_accepted=False,box_jump_accepted=False,next_required=report['next_required'],report='deliverables/jump_motion_development/index.html'),indent=2))
    print('JUMP_DEVELOPMENT_REPORT',out,flush=True)


if __name__=='__main__':main()
