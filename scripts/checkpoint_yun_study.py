"""Save reviewable results without promoting rejected box-jump trials."""
from pathlib import Path
from datetime import datetime
import argparse,json,hashlib

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/power_comparison/yun_sungbin';OUT=ROOT/'deliverables/yun_sungbin_study'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--usage-used-percent',type=float);a=p.parse_args();OUT.mkdir(exist_ok=True)
 valid=[];rejected=[]
 for folder in sorted([*BASE.glob('forward_*'),*BASE.glob('native_*')]):
  audit=folder/'verification.json'
  if not audit.exists():
   rejected.append(dict(case=folder.name,status='trajectory_incomplete'));continue
  v=read(audit);passed=all(v['gates'].values())and sha(folder/'motion.npz')==v['motion_sha256']
  if not passed:
   rejected.append(dict(case=folder.name,status='rejected',failed_gates=[k for k,b in v['gates'].items()if not b]));continue
  c=v['checks'];placement=read(folder/'box_placement.json')if(folder/'box_placement.json').exists()else{}
  valid.append(dict(case=folder.name,motion_sha256=v['motion_sha256'],gates_passed=sum(v['gates'].values()),COM_rise_m=c['airborne_COM_rise_m'],takeoff_velocity_m_s=c['actual_takeoff_velocity_m_s'],lowest_sole_peak_height_m=c['max_minimum_sole_height_m'],peak_GRF_N=c['peak_GRF_N'],peak_propulsive_COM_vertical_power_W=c['peak_propulsive_COM_vertical_power_W'],stance_force_RMS_BW=c['stance_force_RMS_BW'],stance_moment_max_BW_height=c['stance_moment_max_BW_height'],flight_root_max_SI=c['flight_root_max_SI'],complete_footprint_box=placement.get('best_complete_footprint_placement'),bilateral_forefoot_box=placement.get('best_bilateral_forefoot_placement'),source_motion_is_measured_human=False))
 squat=read(BASE/'squat_240/summary.json');ts=read(BASE/'squat_240/analysis/tension_summary.json')
 assert ts['source_motion_sha256']==sha(BASE/'squat_240/analysis/motion.npz')
 squat['muscle_forces_computed']=True;squat['human_muscle_force_validated']=False
 (BASE/'squat_240/summary.json').write_text(json.dumps(squat,indent=2))
 scenario=read(ROOT/'scenarios/yun_sungbin_squat_box_jump.json')
 scenario['updated_at']=datetime.now().astimezone().isoformat();scenario['status']='squat_rendered_forward_jump_validated_160cm_box_pending'
 scenario['subject']['morphology_applied_to_model']=True
 scenario['subject']['morphology_policy']='Native 178 cm rigid geometry; segment masses normalized to 87 kg; nonhelper inertias follow validated uniform height similarity. Individual anatomy is unmeasured.'
 scenario['squat']['status']='trajectory_conditional_tensions_and_portrait_preview_complete'
 scenario['squat']['result_folder']='results/power_comparison/yun_sungbin/squat_240'
 scenario['box_jump']['status']='no_validated_160cm_bilateral_box_arrival_or_landing'
 scenario['revised_box_development_method']['implementation_status']='Contact-acceleration QP ground dynamics, unilateral support modes and conservative centroidal flight implemented. Generic forward-jump trials pass 14 audits; complete box arrival and landing remain unresolved.'
 scenario['execution_note']='240 kg squat and generic forward-jump diagnostics calculated. Reported 160 cm remains an unverified comparison target; no athlete muscle-force result or successful 160 cm jump is claimed.'
 (ROOT/'scenarios/yun_sungbin_squat_box_jump.json').write_text(json.dumps(scenario,indent=2))
 report=dict(updated_at=scenario['updated_at'],status='partial_review_160cm_comparison_pending',subject=dict(height_m=1.78,mass_kg=87.,individual_anatomy_measured=False),squat=dict(all_mechanics_gates_passed=squat['all_mechanics_gates_passed'],bar_mass_kg=240,descent_s=1,ascent_s=2,peak_combined_system_COM_vertical_power_W=squat['peak_COM_vertical_power_W'],bar_travel_m=squat['bar_travel_m'],conditional_tensions_computed=True,human_muscle_force_validated=False),validated_forward_jump_trials=valid,rejected_or_incomplete_trials=rejected,box_160cm=dict(bilateral_arrival_verified=False,landing_simulated=False,athlete_maximum_capability_predicted=False),assumptions=['Rigid segments, standard generic muscle architecture and mass fractions.','Gravity and unilateral Coulomb-friction contact; air drag omitted.','Desired COM and posture curves are motion-design objectives. Realized takeoff speed is an output and can differ from the nominal target.','No muscle strength ceiling, force clipping or external floating-root support.','One muscle tension history selected by magnitude and continuity regularization; neural recruitment, fatigue and tissue damage are not predicted.','COM pseudo-power and segmental energy rate remain distinct metrics.','Numerical equilibrium checks do not establish individual human muscle-force accuracy.'],next_steps=['Constrain the box-jump posture family and timing with an actual motion reference; keep its status distinct from athlete-specific reconstruction.','Improve coordinated foot approach and bilateral top-surface arrival under conserved angular momentum.','Validate 160 cm precontact geometry, then impact, deceleration and stable box support.','Only then release the requested 240 kg squat / 160 cm box-jump muscle comparison.'],usage=dict(requested_minimum_remaining_percent=75,last_observed_used_percent=a.usage_used_percent,last_observed_remaining_percent=None if a.usage_used_percent is None else 100-a.usage_used_percent),sources=[dict(label='2018 athlete profile',url='https://www.yna.co.kr/view/AKR20180216041800007'),dict(label='Contemporary reported 160 cm box jump',url='https://www.donga.com/news/Sports/article/all/20180215/88688614/2'),dict(label='CMJ reference dataset',url='https://doi.org/10.6084/m9.figshare.28890545'),dict(label='CMJ dataset paper',url='https://doi.org/10.1038/s41597-025-05934-5'),dict(label='Shoulder model source',url='https://doi.org/10.3389/fnbot.2019.00090')])
 report['human_motion_plausibility_review_pending']=True
 report['force_review_note']='Fast synthetic joint transitions can create large conditional muscle tensions. The experimental higher-jump forces are review results, not validated athlete force estimates.'
 import numpy as np
 force_case=BASE/'forward_v480_coordinated/analysis'
 if (force_case/'tensions.npz').exists():
  f=np.load(force_case/'tensions.npz');summary=read(force_case/'tension_summary.json');count=summary['model_muscle_elements'];i,j=np.unravel_index(np.argmax(f['force_N'][:,:count]),f['force_N'][:,:count].shape)
  report['higher_jump_force_diagnostic']=dict(case='forward_v480_coordinated',peak_element=summary['names'][j],time_s=float(f['times'][i]),total_N=float(f['force_N'][i,j]),active_N=float(f['active_force_N'][i,j]),passive_N=float(f['passive_force_N'][i,j]),status='human_motion_and_force_plausibility_review_pending')
 report['code_sha256']={name:sha(ROOT/'scripts'/name)for name in ['build_forward_jump_posthoc.py','dynamic_ground_path.py','native_jump_ground_path.py','momentum_ground_path.py','build_contact_jump.py','retarget_measured_jump.py','build_loaded_squat_power.py','prepare_power_case_analysis.py','build_contact_jump_tensions.py','prepare_contact_jump_visuals.py','render_squat_power_blender.py','case_absolute_render.py','build_yun_squat_preview.py','place_box_on_forward_jump.py']}
 (OUT/'study_progress.json').write_text(json.dumps(report,indent=2));(BASE/'progress.json').write_text(json.dumps(report,indent=2))
 jump_file='forward_jump_53cm_validation.mp4'if(OUT/'forward_jump_53cm_validation.mp4').exists()else'forward_jump_validation.mp4'
 html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Biomechanics study review</title><style>body{font:16px system-ui;background:#0d1726;color:#edf3fb;max-width:1040px;margin:40px auto;padding:0 24px}h1{font-size:30px}p{max-width:800px;line-height:1.6;color:#b5c8da}.grid{display:flex;gap:28px;flex-wrap:wrap}video{width:320px;max-width:100%;background:#0d1726}a{color:#8dcfdb}.card{max-width:440px}</style><h1>Biomechanics study review</h1><p>178 cm / 87 kg generic model. Prescribed task mechanics and conditional muscle tension estimates.</p><div class="grid"><div class="card"><h2>240 kg squat</h2><p>1 s descent / 2 s ascent. Three replays with synchronized muscle traces.</p><video controls preload="metadata" src="squat_240_preview.mp4"></video></div><div class="card"><h2>Forward-jump validation</h2><p>Ground propulsion and conservative free flight. Landing is not simulated.</p><video controls preload="metadata" src="JUMP_FILE"></video></div></div><h2>160 cm box comparison remains pending</h2><p>The reported box height has not been substituted for COM rise. No successful bilateral 160 cm arrival, landing, or athlete-specific force estimate has been established. This review is not the requested completed athlete comparison.</p><p>Numerical equilibrium, contact and flight checks are reported separately from human muscle-force accuracy.</p><p><a href="study_progress.json">Calculation status and assumptions</a></p><p>MyoSim / MuJoCo / Seth et al. 2019 / Calisti et al. 2025.</p></html>'''.replace('JUMP_FILE',jump_file)
 (OUT/'index.html').write_text(html,encoding='utf-8');print('STUDY_CHECKPOINT',json.dumps(dict(validated_trials=len(valid),rejected_or_incomplete=len(rejected),status=report['status'])),flush=True)
if __name__=='__main__':main()
