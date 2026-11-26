"""Primary-source catalogue and the mechanical comparison contract.

No new packages, external services or physiology models are installed.
"""
from pathlib import Path
import json,html

ROOT=Path(__file__).resolve().parents[1]
DATE='2026-11-23'
ITEMS=[
 dict(id='rtmpose_tracking',title='RTMPose / RTMLib',year=None,kind='Open-source 2D pose estimation library',
      links=dict(code='https://github.com/Tau-J/rtmlib',pose_api='https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/pose_estimation/rtmpose.py',body_with_feet='https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/body_with_feet.py'),
      input='Video frames and athlete bounding boxes; Halpe26 includes toe and heel keypoints.',output='2D keypoints and confidence scores; no measured depth or forces.',
      reuse='Independent tracking check for the crouch/tuck phases where MediaPipe selected background legs. Installed only in the isolated reference-tracking environment.',
      limits='Manual athlete boxes do not eliminate occlusion or keypoint errors. Estimated keypoints require frame review, and cannot establish calibrated 3D dynamics.'),
 dict(id='opencap',title='OpenCap: Human movement dynamics from smartphone videos',year=2023,kind='Open-source pipeline + peer-reviewed paper',
      links=dict(paper='https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462',kinematics='https://github.com/opencap-org/opencap-core',dynamics='https://github.com/opencap-org/opencap-processing',example='https://github.com/opencap-org/opencap-processing/blob/main/example_kinetics.py'),
      input='Two or more synchronized, calibrated camera views; subject dimensions; a musculoskeletal model.',output='3D kinematics, modeled contacts, joint mechanics and conditional muscle-force estimates.',
      reuse='Separate video kinematics from dynamic simulation; inspect the kinetics example and contact/trajectory-tracking workflow.',
      limits='Validation applies to the reported tasks and conditions. It does not validate our single-view box jump or 240 kg squat. Its physiological outputs are outside our present scope.'),
 dict(id='moco',title='OpenSim Moco: Musculoskeletal optimal control',year=2020,kind='Open-source solver + peer-reviewed paper',
      links=dict(paper='https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008493',code='https://github.com/opensim-org/opensim-core',reproduction='https://github.com/stanfordnmbl/mocopaper',example='https://github.com/opensim-org/opensim-core/blob/main/Bindings/Python/examples/Moco/exampleSquatToStand/exampleSquatToStand_answers.py',tutorial='https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089435/Moco%3A%2BPredict%2Ba%2BSquat-to-stand'),
      input='Rigid-body model, muscle paths, external contacts, task goals and optionally a reference trajectory.',output='Prescribed, tracked or predicted motions and muscle-driven solutions, depending on the formulation.',
      reuse='The squat-to-stand example is a concrete reference for movement definition, dynamic solution and comparing a model modification.',
      limits='The teaching example is simplified, not a validated full-body loaded high-/low-bar/front squat. We would borrow the mechanics formulation without adding physiological activation or fatigue predictions.'),
 dict(id='pose2sim',title='Pose2Sim: An open-source Python package for multiview markerless kinematics',year=2022,kind='Open-source tracking pipeline + software paper',
      links=dict(paper='https://joss.theoj.org/papers/10.21105/joss.04362',code='https://github.com/perfanalytics/pose2sim'),
      input='Calibrated multi-view video and 2D body observations.',output='3D marker trajectories and OpenSim kinematics.',
      reuse='Camera calibration, triangulation, tracking QC and musculoskeletal IK before any force calculation.',
      limits='A kinematics workflow does not by itself identify contact forces or muscle tensions. Single-view depth ambiguity remains in our current source video.'),
 dict(id='addbiomechanics',title='AddBiomechanics: Automating model scaling, inverse kinematics, and inverse dynamics from human motion data through sequential optimization',year=2023,kind='Open-source fitting pipeline + peer-reviewed paper',
      links=dict(paper='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0295152',code='https://github.com/keenon/AddBiomechanics',reproduction='https://github.com/stanfordnmbl/addbiomechanics-paper',docs='https://nimblephysics.org/docs/working-with-addbiomechanics-data.html'),
      input='Optical marker trajectories; corresponding measured ground-reaction forces for the dynamics fit.',output='Scaled geometry, marker registration, kinematics, adjusted inertial properties and inverse dynamics.',
      reuse='Quantify root residuals and fit motion/inertia to measured forces rather than treating force balance alone as human-motion validation.',
      limits='This is not a direct video-to-individual-muscle-force tool. Our video reference has no measured force plates, so its measured-force validation cannot be transferred.'),
 dict(id='myoconverter',title='MyoConverter',year=None,kind='Open-source model conversion and validation tools',
      links=dict(code='https://github.com/MyoHub/myoconverter',kinematics='https://myoconverter.readthedocs.io/en/latest/step2.html',kinetics='https://myoconverter.readthedocs.io/en/latest/step3.html',limitations='https://myoconverter.readthedocs.io/en/latest/limitations.html'),
      input='OpenSim musculoskeletal models.',output='MuJoCo model conversion and optimization of muscle path/force properties.',
      reuse='Compare muscle lengths, moment arms and force behavior across engines before trusting tension magnitudes.',
      limits='Conversion has documented limitations for contacts, moving/conditional path points and some actuators. Successful conversion does not validate every muscle path or exercise.'),
 dict(id='porsa_jump',title='Direct Methods for Predicting Movement Biomechanics Based Upon Optimal Control Theory with Implementation in OpenSim',year=2016,kind='Peer-reviewed predictive jumping study',
      links=dict(paper='https://pubmed.ncbi.nlm.nih.gov/26715209/',doi='https://doi.org/10.1007/s10439-015-1538-6'),
      input='Eight-segment, 48-muscle model; maximum-height jumping task and optimal-control assumptions.',output='Predicted jumping kinematics and ground reactions compared with experimental trajectories.',
      reuse='A jumping-specific reference for dynamic consistency and time-history validation, not just matching peak values.',
      limits='It addresses maximum-height jumping, not our complete box approach, tuck and supported landing. Code availability was not established in this collection.'),
 dict(id='sky_higher',title='Sky Higher: Dynamic Optimization of Maximum Jump Height',year=2013,kind='Official OpenSim-hosted teaching project',
      links=dict(tutorial='https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53087695/Sky+Higher+Dynamic+Optimization+of+Maximum+Jump+Height'),
      input='Simplified lower-body muscle model, foot-ground spheres and optional shoulder-torque arms.',output='Optimized takeoff and ballistic COM height.',
      reuse='Keep post-takeoff COM motion ballistic while limb rearrangement changes foot clearance. Useful separation for box jumps.',
      limits='An educational simplified model, not a quantitative reference for our full-body individual muscle forces or a 160 cm box height.'),
 dict(id='larsen_2024',title='A Biomechanical Comparison Between a High and Low Barbell Placement on Net Joint Moments, Kinematics, Muscle Forces, and Muscle-Specific Moments in 3 Repetition Maximum Back Squats',year=2024,kind='Peer-reviewed measured-motion + muscle-estimation comparison',
      links=dict(paper='https://pubmed.ncbi.nlm.nih.gov/38900172/',doi='https://doi.org/10.1519/JSC.0000000000004790'),
      input='Twelve trained men; standardized stance; high- and low-bar 3RM trials; motion capture and force plates.',output='Inverse-dynamic joint moments and statically optimized individual muscle forces.',
      reuse='Closest empirical comparison to the proposed high-/low-bar muscle-tension study.',
      limits='No significant lower-limb force/moment differences were found under this protocol. Do not force our model to produce a large style effect. Model-estimated muscle forces are not directly measured tensions. Abstract reviewed; numerical full-text tables not yet extracted.'),
 dict(id='bryanton_2012',title='Effect of squat depth and barbell load on relative muscular effort in squatting',year=2012,kind='Peer-reviewed depth/load mechanics study',
      links=dict(paper='https://pubmed.ncbi.nlm.nih.gov/22797000/',doi='https://doi.org/10.1519/JSC.0b013e31826791a7'),
      input='Ten strength-trained women; 50-90% 1RM; squat joint moments and angle-matched maximum voluntary torques.',output='Joint-level relative muscular effort as net moment divided by angle-matched maximum torque.',
      reuse='Separate depth and load in the comparison design and match the mechanical quantity being compared.',
      limits='RME is a joint torque ratio, not individual muscle activation or direct tendon tension. Its strength-normalized metric is outside our present output scope. Abstract reviewed.'),
 dict(id='gullett_2009',title='A biomechanical comparison of back and front squats in healthy trained individuals',year=2009,kind='Peer-reviewed front/back squat comparison',
      links=dict(paper='https://pubmed.ncbi.nlm.nih.gov/19002072/',doi='https://doi.org/10.1519/JSC.0b013e31818546bb'),
      input='Fifteen trained participants; motion, force and surface EMG during front/back squats.',output='Net knee forces and moments, with EMG as a separate measurement.',
      reuse='External bar placement, knee compression/shear and posture checks for the future front-squat branch.',
      limits='Do not substitute its EMG for muscle tension or generalize a relative-load protocol to an equal-100 kg comparison without matching conditions. Abstract reviewed.')
]

def main():
 out=ROOT/'docs';out.mkdir(exist_ok=True)
 catalog=dict(reviewed_on=DATE,primary_sources_only=True,collection_type='Annotated source links and reuse notes, not downloaded full-text papers',
              project_scope='Motion definition -> contact and inverse dynamics -> conditional muscle tensions -> within-muscle comparison. No physiological activation, EMG, fatigue or injury predictions.',items=ITEMS)
 (out/'modeling_references.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
 esc=html.escape;cards=[]
 for item in ITEMS:
  links=' | '.join(f'<a href="{esc(v,quote=True)}">{esc(k)}</a>'for k,v in item['links'].items())
  fields=''.join(f'<p><b>{label}:</b> {esc(item[key])}</p>'for label,key in [('Inputs','input'),('Outputs','output'),('Use here','reuse'),('Limits','limits')])
  cards.append(f'<article><small>{esc(item["kind"])} / {item["year"]or"current docs"}</small><h2>{esc(item["title"])}</h2><nav>{links}</nav>{fields}</article>')
 body=''.join(cards)
 (out/'modeling_references.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Movement mechanics reference library</title>
 <style>body{{font:16px/1.6 system-ui;background:#101923;color:#e9eff4;max-width:1050px;margin:40px auto;padding:0 24px}}a{{color:#6ce0d4}}h1{{line-height:1.2}}h2{{font-size:21px;line-height:1.35}}article{{border-top:1px solid #40505f;padding:24px 0}}small{{color:#9aafc1}}p{{margin:9px 0}}nav{{font-size:14px}}</style>
 <h1>Movement -> mechanics -> muscle tension</h1><p>{esc(catalog['project_scope'])}</p><small>Primary sources reviewed {DATE}. Annotated links; full-text numerical extraction remains pending where stated.</small>{body}</html>''',encoding='utf-8')
 contract=dict(updated_on=DATE,scope='mechanical_only',physiological_activation_in_scope=False,
   excluded_outputs=['neural_activation','EMG_prediction','fatigue','metabolism','microdamage','strength_failure'],
   stages=[dict(id='motion',inputs=['rigid_segment_geometry','joint_constraints','time_resolved_reference_or_prescription','bar_attachment_and_grip'],outputs=['q','qdot','qddot'],gates=['source_visual_trajectory_review','bone_lengths','joint_limits','continuous_motion','hand_bar_contact']),
           dict(id='contact_and_inverse_dynamics',inputs=['segment_masses_and_inertias','gravity','q_qdot_qddot','contacts_and_friction','external_loads'],outputs=['body_and_combined_COM','COP','contact_wrenches','joint_moments','net_joint_reactions'],gates=['Newton_Euler_balance','contact_support_and_friction','ballistic_flight','power_and_work_energy','time_step_sensitivity']),
           dict(id='muscle_tension',inputs=['joint_mechanics','muscle_paths_and_moment_arms','passive_tension_law','explicit_force_sharing_and_stability_assumptions'],outputs=['muscle_tension_N','conditional_muscle_induced_joint_reactions'],gates=['muscle_moment_balance','path_and_moment_arm_reference_checks','sharing_sensitivity']),
           dict(id='comparison',inputs=['phase_matched_conditions'],outputs=['peak_N','mean_N','time_integral_N_s','within_muscle_reference_ratio','force_per_bodyweight'],gates=['same_muscle_definition','same_phase_window','same_normalization','declared_load_depth_cadence_and_body'])],
   inference_limit='Motion and external loads identify net joint mechanics, not a unique muscle force vector. Tension estimates remain conditional on path, passive and force-sharing/stability assumptions even with no physiological activation model.',
   normalization=dict(preferred='F_case/F_reference for the same muscle and metric',alternative='F/(body_mass*g)',not_activation=True,sum_of_all_muscle_forces_not_external_output=True),
   current_squat=dict(implemented=['high_bar','variable_bottom_knee_flexion_within_existing_path','body_plus_bar_COM','full_body_and_bar_inertia','calculated_COP_and_GRF'],
       prescribed='Coupled hip/knee path; root/torso solve fixed feet and horizontal system COM at support centre. This is a balance strategy assumption, not a free balance prediction.',
       pending_variants=['low_bar_attachment_and_grip_plus_referenced_posture_family','front_rack_attachment_and_grip_plus_referenced_posture_family'],
       comparison_controls=['same_absolute_bar_mass_first','same_height_mass_segment_ratios','same_footwear_stance','explicit_depth_and_timing','do_not_presume_large_style_effect']),
   jump=dict(current='sequence_level_image_reprojection_fit',force_analysis_allowed=False,pending=['native_3D_visual_review','metric_geometry_and_physical_timing','contact_events','COM_ballistic_flight_and_momentum_checks']))
 scenario=ROOT/'scenarios/motion_mechanics_comparison.json';scenario.parent.mkdir(exist_ok=True)
 scenario.write_text(json.dumps(contract,indent=2),encoding='utf-8')
 state=ROOT/'results/reference_box_jump/continuation_state.json'
 if state.exists():
  record=json.loads(state.read_text());record.update(date=DATE,status='active_reference_trajectory_review',usage_floor_cancelled_by_user=True,physiology_in_scope=False)
  record.pop('remaining_usage_percent_last_observed',None)
  record['completed']=['Corrected camera projection and front/rear initialization rerun.','Native-to-source reprojection and trajectory QC images generated.']
  record['pending']=['Review native 3D poses across countermovement, takeoff, tuck and box support.','Confirm metric geometry, physical timing and contact transitions.','Check flight COM and whole-body momentum before enabling forces.']
  state.write_text(json.dumps(record,indent=2),encoding='utf-8')
 print(json.dumps(dict(references=len(ITEMS),catalog=str(out/'modeling_references.html'),contract=str(scenario),scope='mechanical_only')))

if __name__=='__main__':main()
