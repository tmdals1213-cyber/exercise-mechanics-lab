"""Prepare the four-variant experiment specification, not simulated results."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/squat_comparison';OUT.mkdir(parents=True,exist_ok=True)
variants=[
 {'id':'high_bar','label':'HIGH-BAR SQUAT','bar_support_region':'upper trapezius / upper back','arm_rack':'back rack; hands stabilize bar'},
 {'id':'low_bar','label':'LOW-BAR SQUAT','bar_support_region':'posterior deltoid shelf on the upper back','arm_rack':'low back rack; hands stabilize bar'},
 {'id':'front','label':'FRONT SQUAT','bar_support_region':'anterior deltoid and clavicular shelf','arm_rack':'front rack; elbows forward and raised'},
 {'id':'zercher','label':'ZERCHER SQUAT','bar_support_region':'bilateral elbow creases / proximal forearm contact','arm_rack':'bar cradled by flexed elbows; hands together'}]
common={'subject':{'sex':'male','height_m':1.8,'mass_kg':90.},'bar_mass_kg':100.,'bar_diameter_m':.028,
 'stance':{'ankle_centre_width_m':.36,'toe_out_deg':15.,'heel_elevation_m':0.,'status':'candidate; verify foot geometry and feasible joint ranges'},
 'depth':{'criterion':'hip-to-knee segment horizontal at bottom','target_angle_deg':0.,'tolerance_deg':1.,'competition_depth_claim':False},
 'timing':{'descent_s':2.,'bottom_hold_s':.3,'ascent_s':2.,'top_hold_s':.3},
 'strength_ceiling_enforced':False,'head_pose':'neutral relative to trunk; clearance to be checked',
 'joint_classification':'hip flexion is independent; anatomical knee couplers retained',
 'matching':['anthropometry','bar mass','stance and footwear','depth definition','phase timing','camera and force scales'],
 'solved_not_prescribed':['trunk inclination','hip/knee/ankle strategy','actual bar-surface contact points','support-force distribution']}
presentation={'language':'en','canvas':[1080,1920],'fps':30,'bottom_reserved_px':384,
 'title':'SQUAT / BAR POSITION','question':'How does bar position change a squat?',
 'main_view':'front oblique; full body','comparison_view':'matched side views for bar position and trunk inclination',
 'intro_format':'one full cycle per variant, followed by one synchronized four-way cycle',
 'intro_duration_target_s':25.,'detailed_muscle_analysis':'follow-up episode',
 'muscle_color_policy':'computed tension only after validated contact/trajectory dynamics; no manually assigned dominance colors',
 'first_metrics':['bar support position and path','trunk inclination','phase-matched posture comparison'],
 'later_metrics':['hip/knee/ankle net joint moments','active/passive muscle tension','joint transmitted resultants','anthropometry sensitivity']}
requirements=['Map external loading to the actual support region; front and back racks must not load only the hands.',
 'Fit the shaft radius to the displayed surfaces, including posterior racks and elbow creases.',
 'Verify flat-foot support, friction and zero root reserve across the entire cycle.',
 'Let posture follow geometry, balance and stated motion criteria; do not assign a fixed muscle-dominance ranking.',
 'Validate each upper-body rack and arm range before interpreting muscle forces.',
 'Do not infer local skin pressure, pain or injury risk from elbow contact resultants.']
sources=[{'url':'https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.604177/full',
 'use':'Matched load, stance and depth matter; high/low bar kinematic differences were small in this particular experiment. EMG is not used as force data.'},
 {'url':'https://pubmed.ncbi.nlm.nih.gov/33161870/','use':'Load-placement comparison; participants adjusted posture, and most lower-limb moments were similar across conditions. Not a Zercher validation.'},
 {'url':'https://www.nsca.com/education/articles/kinetic-select/zercher-squat-field-alternative-for-front-squat/','use':'Technique reference; not experimental validation of the proposed Zercher force model.'}]
spec={'schema_version':1,'status':'specification_only_not_simulated','common':common,'variants':variants,
 'presentation':presentation,'required_checks':requirements,'sources':sources}
for variant in variants:
 config={'movement':'squat','status':'specification_only_not_simulated','common':common,'variant':variant,'presentation':presentation}
 (ROOT/f'scenarios/squat_{variant["id"]}.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
(ROOT/'scenarios/squat_comparison.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
(OUT/'comparison_setup.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
print('FOUR_SQUAT_COMPARISON_SPECIFICATIONS_PREPARED',flush=True)
