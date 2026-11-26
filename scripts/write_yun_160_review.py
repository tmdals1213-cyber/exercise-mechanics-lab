"""Local review entry point and explicit continuation state; no web publishing."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]


def main():
    case=ROOT/'results/athletic_box_jump/yun_160_supported_cycle_v2';out=ROOT/'deliverables/yun_160_box_jump_review';audit=json.loads((case/'cycle_audit.json').read_text());movie=out/'box_jump_motion_replay.mp4'
    if not movie.exists():raise RuntimeError('Render and compose the new 160 cm movie first')
    coarse=json.loads((ROOT/'results/athletic_box_jump/yun_160_flat_feet_v2/supported_delayed_balance_v1/landing_audit.json').read_text());fine=audit['support_audit'];convergence=dict(coarse_hz=500,refined_hz=1000,coarse_peak_landing_GRF_N=coarse['peak_landing_GRF_N'],refined_peak_landing_GRF_N=fine['peak_landing_GRF_N'],peak_relative_change=abs(fine['peak_landing_GRF_N']/coarse['peak_landing_GRF_N']-1),contact_time_difference_s=abs(fine['contact_time_s']-coarse['contact_time_s']),caveat='Time-step sensitivity and balance are numerical checks, not experimental or individual muscle-force accuracy.')
    (out/'time_step_check.json').write_text(json.dumps(convergence,indent=2))
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>160 cm box jump</title>
<style>body{{max-width:940px;margin:24px auto;padding:20px;background:#0d1726;color:#edf3fb;font:17px/1.6 system-ui}}a{{color:#8ce2d5}}video{{display:block;max-width:100%;height:78vh;margin:24px auto}}details{{background:#152338;padding:20px;border-radius:12px}}small{{color:#a6bcd1}}li{{margin:.6em 0}}</style>
<h1>160 cm box jump</h1><p>178 cm / 87 kg · 1.60 m box · motion study</p><video src="box_jump_motion_replay.mp4" poster="poster.png" controls loop playsinline preload="metadata"></video>
<p>Takeoff → tuck → box contact → standing hold.</p><p><small>A specified motion task at a Yun-level box height. Athlete-specific kinematics and muscle loads have not been validated.</small></p>
<details><summary>Model and checks</summary><ul>
<li>Rigid segments with articulated ankles, subtalar and MTP joints; scapular contact geometry. Gravity 9.80665 m/s²; no air drag; Coulomb friction μ = 0.8.</li>
<li>No floating-root motor, pose reset, velocity reset, muscle-strength ceiling or force clipping.</li>
<li>Landing reaction peak: {fine['peak_landing_GRF_N']/1000:.2f} kN. Changing the landing step from 2 ms to 1 ms changed the peak by {100*convergence['peak_relative_change']:.2f}%.</li>
<li>Independent landing momentum check: maximum force-equivalent residual {fine['independent_discrete_momentum_force_equivalent_max_N']:.2f} N; RMS {fine['independent_discrete_momentum_force_equivalent_RMS_N']:.3f} N.</li>
<li>Terminal support force approaches body weight. No body-capsule/box intersection; numerical sole penetration below 0.4 mm. Full skin/mesh collision is a separate, unreleased check.</li>
<li>MyoSim full-body skeleton with the project scapular/MTP rig. High-flexion ranges use the Catelli and Rajagopal–Lai–Uhlrich models. This does not import or validate their muscle paths; native deep-knee coupler extrapolation still requires review.</li>
<li>Joint control supplies the task torques. Muscle recruitment, individual force accuracy and a 240 kg squat comparison remain unreleased. Mechanical balance checks do not establish 10% human-muscle accuracy.</li></ul>
<p><a href="../../results/athletic_box_jump/yun_160_supported_cycle_v2/cycle_audit.json">Full motion audit</a> · <a href="time_step_check.json">Step-size check</a> · <a href="review_manifest.json">Video manifest</a> · <a href="../yun_sungbin_study/squat_240_preview.mp4">Existing 240 kg squat motion</a></p></details></html>'''
    (out/'index.html').write_text(html,encoding='utf-8')
    old=ROOT/'deliverables/jump_motion_development';archive=old/'index_measured_cmj_archive.html'
    if (old/'index.html').exists() and not archive.exists():archive.write_bytes((old/'index.html').read_bytes())
    (old/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../yun_160_box_jump_review/index.html"><title>160 cm box jump</title><a href="../yun_160_box_jump_review/index.html">Open the new 160 cm box-jump review</a></html>',encoding='utf-8')
    study=ROOT/'deliverables/yun_sungbin_study';old_study=study/'index_before_160cm_motion.html'
    if (study/'index.html').exists() and not old_study.exists():old_study.write_bytes((study/'index.html').read_bytes())
    (study/'index.html').write_text((old/'index.html').read_text(encoding='utf-8'),encoding='utf-8')
    state=dict(status='160cm_generated_full_cycle_motion_review_available',active_target='Yun-level 160 cm box jump; not ordinary CMJ replay',active_generated_motion=str(case.relative_to(ROOT)),active_contact_experiment='results/athletic_box_jump/yun_160_refined_flight_v1/supported_balance_v1',generated_landing_accepted_for_motion=True,box_jump_full_cycle_motion_completed=bool(audit['full_cycle_motion_pass']),mechanical_root_balance_pass=bool(audit['independent_dynamics_pass']),box_jump_muscle_load_validation_complete=False,athlete_personal_measurement_reproduced=False,active_video=str(movie.relative_to(ROOT)),report=str((out/'index.html').relative_to(ROOT)),next_required=['Review the actual 160 cm full-cycle video; do not restart ordinary ground-jump reconstruction.','Validate deep-flexion joint geometry and muscle paths before releasing muscle-force recruitment.','Use the accepted contact/COM/angular-momentum history for the subsequent 240 kg squat comparison.'])
    (ROOT/'results/reference_box_jump/continuation_state.json').write_text(json.dumps(state,indent=2));(out/'continuation_state.json').write_text(json.dumps(state,indent=2))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [case/'motion.npz',movie,ROOT/'scripts/plan_athletic_box_velocity.py',ROOT/'scripts/support_athletic_box_landing.py',ROOT/'scripts/finalize_athletic_box_cycle.py',ROOT/'scripts/render_squat_power_blender.py']};(out/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
    (ROOT/'open_yun_box_jump.cmd').write_text('@echo off\r\nstart "" "%~dp0deliverables\\yun_160_box_jump_review\\index.html"\r\n',encoding='ascii')
    print('YUN_BOX_REVIEW_READY',json.dumps(state),flush=True)


if __name__=='__main__':main()
