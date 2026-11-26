"""Keep mechanical consistency separate from task success and human validity."""
from pathlib import Path
import json,hashlib
import numpy as np
from audit_jump_coordination import audit
ROOT=Path(__file__).resolve().parents[1]
def main():
 base=ROOT/'results/physical_jump_task';rows=[]
 for name,target in [('physical_jump_task_v3',2.5),('physical_jump_task_v4',3.2),('physical_jump_task_v5',2.5),('physical_jump_task_v6',2.5)]:
  if not (base/name/'verification.json').exists():continue
  folder=base/name;r=json.loads((folder/'verification.json').read_text());z=np.load(folder/'motion.npz');actual=float(r['takeoff_velocity_m_s'][2]);error=abs(actual-target)/target;air=z['mode']==2;F=z['root_residual'][air,:3];M=z['root_residual'][air,3:]-np.cross(z['com'][air],F)
  coordination=audit(folder)
  r.update(numerical_mechanics_passed=all(r['gates'].values()),human_motion_validated=False,ready_for_comparative_analysis=False,motion_sha256=hashlib.sha256((folder/'motion.npz').read_bytes()).hexdigest())
  r['task_tracking']=dict(requested_launch_vz_m_s=target,actual_launch_vz_m_s=actual,relative_launch_velocity_error=error,within_ten_percent=error<=.1,requested_target_is_a_soft_control_goal=True)
  r['checks'].update(flight_force_residual_max_N=float(np.linalg.norm(F,axis=1).max()),flight_moment_about_COM_residual_max_Nm=float(np.linalg.norm(M,axis=1).max()))
  r['coordination_screen']=coordination
  quality_status='rejected_foot_crossing' if not coordination['foot_order_preserved'] else 'failed_mechanics_checks' if not all(r['gates'].values()) else 'mechanics_pass_task_incomplete'
  (folder/'verification.json').write_text(json.dumps(r,indent=2));(folder/'motion_quality.json').write_text(json.dumps(dict(status=quality_status,numerical_mechanics_passed=all(r['gates'].values()),task_target_met=error<=.1,foot_order_preserved=coordination['foot_order_preserved'],human_motion_validated=False,force_analysis_allowed=False),indent=2))
  rows.append(dict(case=name,mechanics_passed=all(r['gates'].values()),requested_launch_vz_m_s=target,actual_launch_vz_m_s=actual,target_error_percent=100*error,peak_GRF_N=r['checks']['peak_GRF_N'],peak_GRF_bodyweights=r['checks']['peak_GRF_N']/(87*9.80665),COM_rise_above_takeoff_m=actual**2/(2*9.80665),maximum_full_sole_height_m=r['maximum_full_sole_height_m'],flight_pelvis_rotation_deg=r['checks']['flight_pelvis_rotation_max_deg'],landing_simulated=False))
  rows[-1].update(foot_order_preserved=coordination['foot_order_preserved'],minimum_shank_centreline_distance_m=coordination['minimum_shank_centreline_distance_m'])
 out=ROOT/'deliverables/physical_jump_task_review';out.mkdir(exist_ok=True)
 report=dict(status='development_tests_not_comparative_release',priority='Physical mechanics and realistic coordination; reference videos are advisory only.',scope='Rigid-body mechanics and conditional tension analysis. No physiological activation, fatigue, metabolism or injury model.',cases=rows,
  conclusions=['Numerical rigid-body/contact/flight checks and human coordination are independent acceptance criteria.','Earlier candidates crossed the feet and almost intersected the shank centrelines despite passing equation checks. These candidates are rejected.','Requested launch velocity is a task goal, not an achieved result or an estimate of athlete ability.','The task controller requires trajectory-level takeoff constraints; further local gain tuning is not sufficient evidence of task completion.','Landing impact and supported settling have not been simulated.'],
  next_required=['Whole-trajectory task formulation with an explicit terminal takeoff state and contact feasibility.','Task-appropriate limb ordering, surface clearance and natural coordination independently of equation checks.','Platform clearance, impact and settling.','Only then comparative joint/muscle loads.'],
  force_clipping=False,strength_ceiling=False,hidden_root_support=False,physiology_in_scope=False)
 (out/'development_status.json').write_text(json.dumps(report,indent=2))
 cells=''.join('<tr>'+''.join(f'<td>{v}</td>'for v in [r['case'],f'{r["actual_launch_vz_m_s"]:.2f} / {r["requested_launch_vz_m_s"]:.2f}',f'{r["target_error_percent"]:.1f}%',f'{r["peak_GRF_N"]:.0f}',f'{r["COM_rise_above_takeoff_m"]*100:.1f}',f'{r["maximum_full_sole_height_m"]*100:.1f}', 'Preserved' if r['foot_order_preserved'] else 'REJECTED'])+'</tr>'for r in rows)
 (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
 <meta name="viewport" content="width=device-width,initial-scale=1"><title>Physics-first jump development</title>
 <style>body{font:16px/1.6 system-ui;max-width:1150px;margin:30px auto;padding:20px;background:#101923;color:#ecf1f5}a{color:#6ce0d4}td,th{padding:10px;text-align:left;border-bottom:1px solid #40505f}table{border-collapse:collapse;font-size:14px}</style>
 <h1>Physics-first motion development</h1>
 <p>Reference videos inform movement style. Contact and rigid-body mechanics constrain admissible motion. Realistic coordination must pass a separate review.</p>
 <p><b>Development only.</b> Earlier previews are rejected because the feet cross. Passing Newton–Euler and momentum checks alone does not establish a valid human jump.</p>
 <table><tr><th>Case</th><th>Actual / requested launch (m/s)</th><th>Task error</th><th>Peak GRF (N)</th><th>Flight COM rise (cm)</th><th>Max sole height (cm)</th><th>Foot ordering</th></tr>'''+cells+'''</table>
 <p>COM rise above takeoff and foot clearance are different quantities. Foot ordering and shank centreline checks are geometric screens, not full mesh collision validation.</p>
 <p>The remaining work requires terminal takeoff constraints, plausible coordination, surface clearance, platform impact and supported settling. No result on this page is a validated high-box-jump or athlete-performance estimate.</p>
 <p><a href="development_status.json">Status and remaining work</a> | <a href="../../scenarios/physical_motion_tasks.json">Model scope</a> | <a href="../../docs/modeling_references.html">Reference library</a></p></html>''',encoding='utf-8')
 manifest_path=out/'review_manifest.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text());manifest.update(status='rejected_foot_crossing',accepted_as_final_motion=False,publication_ready=False,reason='The v5 preview crosses the feet and nearly intersects shank centrelines; numerical balance does not validate coordination.');manifest_path.write_text(json.dumps(manifest,indent=2))
 old=ROOT/'deliverables/yun_reference_motion_review/review_manifest.json';record=json.loads(old.read_text());record.update(status='rejected_unnatural_forward_and_arm_motion',accepted_as_reference=False,user_feedback='Forced-looking forward motion and excessive arm motion. Reference similarity must be secondary to physical realism.');old.write_text(json.dumps(record,indent=2))
 state_path=ROOT/'results/reference_box_jump/continuation_state.json';state=json.loads(state_path.read_text());state.update(status='physics_first_task_generation_incomplete',physiology_in_scope=False,force_analysis_allowed=False,reference_role='Movement style only; do not fit exact video pixels at the expense of reality.',active_candidate_folder='results/physical_jump_task/physical_jump_task_v5',
  latest_completed=['Rejected unnatural reference replay.','Measured camera motion and excluded identity-switch observations; retained only as reference diagnostics.','Built a task-defined contact/flight generator with simple sagittal arm swing, unilateral contacts and friction.','Separated equation consistency, terminal-task success and geometric coordination screens. Earlier crossing-leg candidates are rejected.'],
  pending=report['next_required'],next_solver_requirement='Replace local controller-only task generation with a whole-trajectory terminal-state constrained formulation. Do not promote low-hop tests to the requested box-jump scenario.')
 state['active_candidate_folder']='results/physical_jump_task/'+rows[-1]['case']
 state_path.write_text(json.dumps(state,indent=2));protocol_path=ROOT/'scenarios/box_jump_reference_first.json';protocol=json.loads(protocol_path.read_text());protocol.update(status='superseded_by_physics_first_task_generation',priority_protocol='scenarios/physical_motion_tasks.json',reference_video_role='advisory',force_analysis_allowed=False);protocol_path.write_text(json.dumps(protocol,indent=2))
 domain=dict(observed_native_knee_flexion_limit_deg=120.,current_task_tuck_target_deg=float(np.degrees(2.0)),references=[dict(url='https://pmc.ncbi.nlm.nih.gov/articles/PMC3591185/',use='Healthy dynamic deep-flexion kinematics, not a validation of our model.'),dict(url='https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090607/Musculoskeletal%2BModelssrc%3Dbreadcrumbs-parent',use='Model-domain limitations at high knee flexion.')],note='A ROM setting alone cannot validate knee geometry or muscle paths in deep flexion. Audit the model domain before taller box/deep-squat force claims.')
 (ROOT/'docs/deep_flexion_model_domain.json').write_text(json.dumps(domain,indent=2));print(json.dumps(dict(cases=len(rows),all_mechanics_passed=all(row['mechanics_passed'] for row in rows),any_target_met=any(row['target_error_percent']<=10 for row in rows),comparison_release=False,status_file=str(out/'development_status.json'))))
if __name__=='__main__':main()
