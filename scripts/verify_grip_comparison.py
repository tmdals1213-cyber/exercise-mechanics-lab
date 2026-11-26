"""Validate matched experiment, force/moment accounting and final portrait output."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/grip_comparison'
CYCLES='--full-cycle' in sys.argv;PHYSIOLOGY='--physiology' in sys.argv or CYCLES
layout=json.loads((OUT/'layout.json').read_text('utf-8'))
configs=[json.loads((ROOT/f'scenarios/ohp_grip_{g}.json').read_text('utf-8')) for g in [65,80]]
for c in configs:c.pop('grip_width_m')
assert configs[0]==configs[1],'Only grip width should differ'
manifest=json.loads((OUT/('cycle_camera.json' if CYCLES else 'physiology_camera.json' if PHYSIOLOGY else 'camera.json')).read_text('utf-8'));assert manifest['shared_for_both_cases']
physics=[]
for grip in [65,80]:
 case=OUT/f'grip_{grip}'/('cycle' if CYCLES else '')/('physiology' if PHYSIOLOGY else '');d=np.load(case/'fullbody_motion.npz');m=json.loads((case/'fullbody_metadata.json').read_text('utf-8'));t=np.load(case/'grip_traces.npz');report=json.loads((case/'grip_traces.json').read_text('utf-8'))
 if PHYSIOLOGY:
  audit=json.loads((case/'physiology_independent_checks.json').read_text('utf-8'))
  assert audit['status']=='passed' and audit['strength_ceiling_enforced'] is False
  assert m['force_model']=='active_passive_gh20_required_force' and m['strength_ceiling_enforced'] is False
 count=139 if CYCLES else 76
 assert d['force_N'].shape==(count,416) and np.isfinite(d['force_N']).all() and d['force_N'].min()>=0
 assert all(name in m['independent_coordinates'] for name in ['hip_flexion_r','hip_flexion_l'])
 if CYCLES:
  prep=json.loads((case.parent/'cycle_preparation.json').read_text('utf-8'));assert prep['status']=='passed'
  assert np.max(abs(d['qvel_native'][99]+d['qvel_native'][30]))<1e-10
  assert np.max(abs(d['qacc_native'][99]-d['qacc_native'][30]))<1e-10
  assert np.max(abs(d['force_N'][99]-d['force_N'][30]))>1.,'Descent forces merely replayed'
 assert m['max_balance_error']<.01 and abs(d['reserve'][:,:6]).max()<1e-4 and abs(d['reserve'][:,6:]).max()<.1
 assert m['max_IK_error_m']<.003
 span=np.ptp(d['physical_grips'][:,:,0],axis=1);assert abs(span-grip/100).max()<.003
 # A clavicle rack can have oblique forearms. Check side assignment and a
 # downward elbow position rather than forcing a perfectly vertical forearm.
 elbows=d['physical_elbow_centres'][0];hands=d['physical_grips'][0];centre_x=hands[:,0].mean()
 assert elbows[0,0]<centre_x and elbows[1,0]>centre_x and np.all(elbows[:,2]<hands[:,2]-.1)
 assert np.abs(d['qacc_native']).max()<30,'Uncontrolled framewise joint-rotation branch change'
 assert report['status']=='passed' and report['max_elbow_moment_balance_error_Nm']<.01
 duration=report['statistics_interval_s'][1];ascent=t['times']<=duration
 for name,summary in report['summaries'].items():
  assert np.isfinite(t[name]).all() and t[name].min()>-1e-6
  assert abs(summary['mean_ascent']-np.trapezoid(t[name][ascent],t['times'][ascent])/duration)<1e-7
  assert abs(summary['max_ascent']-t[name][ascent].max())<1e-7
 for i in range(1,count+1):
  with Image.open(case/'comparison_frames'/f'frame_{i:04d}.png') as im:assert im.size==tuple(manifest['size'])
 anatomy=json.loads((case/'anatomy_validation.json').read_text('utf-8'));assert anatomy['status']=='passed' and anatomy['lat_insertion_max_error_m']<1e-4
 if CYCLES:
  assert anatomy['separate_visible_native_hand_bone_objects']==0
  geometry=json.loads((case/'bar_geometry_validation.json').read_text());assert geometry['status']=='passed'
  assert -.0001<geometry['actual_start_surface_gap_m']<.003 and geometry['minimum_head_clearance_m']>.002
 physics.append({'grip_cm':grip,'sha256':hashlib.sha256((case/'fullbody_motion.npz').read_bytes()).hexdigest(),
  'hand_fit_error_m':m['max_IK_error_m'],'max_grip_span_error_m':float(abs(span-grip/100).max()),'max_joint_acceleration_native':float(abs(d['qacc_native']).max()),
  'initial_frontal_forearm_angle_deg':np.degrees(np.arctan2(abs(hands[:,0]-elbows[:,0]),hands[:,2]-elbows[:,2])).tolist()})
for name in ['grip_comparison_poster.png','grip_comparison_start.png','grip_comparison_lockout.png','grip_comparison_summary.png']:
 image=np.asarray(Image.open(OUT/name).convert('RGB'));assert image.shape==(1920,1080,3) and np.all(image[1536:]==[13,18,25])
r=imageio.get_reader(OUT/'ohp_grip_comparison_portrait.mp4');info=r.get_meta_data();assert info['size']==(1080,1920) and info['fps']==30 and r.count_frames()==layout['frames']
timeline=json.loads((OUT/'playback_timeline.json').read_text('utf-8'));assert len(timeline)==layout['frames']
steps=round(layout['cycle_duration_s']*30)
for rep in range(1,4):
 frames=[f['source_index'] for f in timeline if f['kind']=='cycle' and f['rep']==rep]
 assert frames==list(range(steps))
assert layout['language']=='en' and layout['repeats']==3
assert not any('\uac00'<=c<='\ud7a3' for c in json.dumps(layout,ensure_ascii=False))
error=0
for i in [0,30,60,90,layout['frames']-1]+[30+(rep-1)*steps+99 for rep in range(1,4) if CYCLES]:
 frame=r.get_data(i);error=max(error,int(abs(frame[1536:].astype(int)-[13,18,25]).max()))
alignment_error=0.;cursor_min_luma=255.
for rep in range(1,4):
 for source_index in ([30,99] if CYCLES else [30]):
  encoded=r.get_data(layout['intro_frames']+(rep-1)*steps+source_index)
  for panel in range(10):
   left=71+(panel%2)*532;expected=int(left+source_index/steps*434);top=895+(panel//2)*117
   columns=encoded[top+3:top+64,expected-2:expected+3].mean(2)
   cursor_min_luma=min(cursor_min_luma,float(np.median(columns,axis=0).max()))
  for grip,x in [(65,28),(80,560)]:
   case=OUT/f'grip_{grip}'/('cycle' if CYCLES else '')/('physiology' if PHYSIOLOGY else '')
   reference=np.array(Image.open(case/'comparison_frames'/f'frame_{source_index+1:04d}.png').convert('RGB'))
   # Exclude the decorative colored top border; test decoded body/time alignment.
   difference=abs(encoded[177:814,x:x+492].astype(float)-reference[3:].astype(float)).mean()
   alignment_error=max(alignment_error,float(difference))
r.close();assert error<=4 and alignment_error<3 and cursor_min_luma>100
report={'status':'passed','canvas':[1080,1920],'fps':30,'frames':layout['frames'],'bottom_reserved_px':384,'shared_camera':True,
 'language':'en','complete_cycles':CYCLES,'repeats':3,'decoded_source_alignment_mean_absolute_error':alignment_error,
 'decoded_expected_graph_cursor_min_median_luma':cursor_min_luma,
 'statistics_interval_s':layout['mean_max_interval_s'],'only_experimental_input_difference':'grip width 0.65 vs 0.80 m','encoded_bottom_max_channel_error':error,
 'backend':'active_passive_gh' if PHYSIOLOGY else 'legacy_minimum_torque','strength_ceiling_enforced':False,
 'cases':physics,'limits':'Numerical and display validation; no experimental validation of muscle co-contraction or tissue loads'}
(OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
