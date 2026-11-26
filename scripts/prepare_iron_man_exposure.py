from pathlib import Path
import json,numpy as np,hashlib,argparse
from retarget_measured_jump import MeasuredJumpRig
from build_contact_jump import contact_geometry
from atlas_mapping import map_atlas
from case_contours import overview
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/iron_man_comparison_final'
def integrate(t,f,a,b):
 tt=np.r_[a,t[(t>a)&(t<b)],b];ff=np.array([np.interp(tt,t,row) for row in f.T]).T;return np.trapezoid(ff,tt,axis=0)
def main():
 r=MeasuredJumpRig(87,1.78);b,m,d=r.b,r.m,r.d;q=b.q0.copy();r.arm(q,.15,.15,.12)
 for side in ['r','l']:
  q[m.jnt_qposadr[b.joints['hip_flexion_'+side]]]=.025;q[m.jnt_qposadr[b.joints['knee_angle_'+side]]]=.04
 q[m.jnt_qposadr[b.joints['flex_extension']]]=.015;b.pose(q);q[2]-=np.mean([d.site_xpos[m.site('subject_sole_reference_'+side).id,2] for side in ['r','l']]);b.pose(q);r.contact_stand_override=q.copy();points=contact_geometry(r)['points'];b.pose(q);standing=dict(times=np.array([0.]),qpos=q[None],body_xpos=d.xpos[None].copy(),body_xmat=d.xmat[None].copy(),contact_positions_m=np.array([[b.jacobian(bid,local)[1] for bid,local,*_ in points]]));cases={};upper=0
 for key in ['jump','squat']:
  folder=OUT/'render_cases'/key;z=dict(np.load(folder/'analysis/tensions.npz'));v=json.loads((folder/'analysis/visualization/data.json').read_text());meta=json.loads((folder/'summary.json').read_text());motion=dict(np.load(folder/'motion.npz'));start=.20 if key=='jump' else .25;end=meta['phase_times_s']['end_ascent'] if key=='squat' else meta['phase_times_s']['end'];F=z['force_N'];names=json.loads((folder/'analysis/tension_summary.json').read_text())['names']
  if key=='jump':
   ts=motion['times'];from scipy.interpolate import CubicSpline
   bodyids=[m.body(n+'_'+side).id for side in ['r','l'] for n in ['lunate','ulna','calcn','femur']];speed=np.max(np.linalg.norm(CubicSpline(ts,motion['body_xpos'][:,bodyids],axis=0)(ts,1),axis=2),axis=1);after=np.flatnonzero(ts>meta['phase_times_s']['landing']+.3)
   for i in after:
    mask=(ts>=ts[i])&(ts<=ts[i]+.2)
    if ts[-1]-ts[i]>=.2 and speed[mask].max()<.12:end=float(ts[i]);break
  element=integrate(z['times'],F,start,end);groups=integrate(np.array(v['times']),overview(v)[[1,0,2,3]].T,start,end);mapping=json.loads((folder/'absolute_render_mapping.json').read_text())['meshes']
  for item in mapping:
   ids=map_atlas(item['mesh'],names);upper=max(upper,float(element[ids].sum()))
  cases[key]=dict(start_s=start,end_s=end,duration_s=end-start,names=names,element_integral_Ns=element.tolist(),group_integral_mean_per_side_Ns=groups.tolist(),group_order=['QUADS','GLUTES','HAMSTRINGS','CALVES'],force_sha256=hashlib.sha256((folder/'analysis/tensions.npz').read_bytes()).hexdigest());np.savez_compressed(folder/'standing_pose.npz',**standing)
 upper=float(np.ceil(upper/1000)*1000)
 for key,c in cases.items():(OUT/'render_cases'/key/'analysis/exposure.json').write_text(json.dumps(dict(c,common_upper_Ns=upper)))
 (OUT/'tension_exposure.json').write_text(json.dumps(dict(metric='Integral of scalar muscle tension over one active movement',units='N s',common_upper_Ns=upper,three_replays_count_as_one=True,windows='Jump preparation to settled landing; squat descent start to ascent end',cases=cases),indent=2));print('FINAL_EXPOSURE_READY',[(k,c['duration_s']) for k,c in cases.items()],upper,flush=True)
if __name__=='__main__':main()
