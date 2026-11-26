"""Integrate existing tensions over one active movement; prepare display-only stance."""
from pathlib import Path
import json,hashlib
import numpy as np
from case_contours import overview
from launch_sources import load,analysis_path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/instagram_launch'

def integrate(t,f,a,b):
    tt=np.r_[a,t[(t>a)&(t<b)],b]
    ff=np.array([np.interp(tt,t,row) for row in np.asarray(f).T]).T
    return np.trapezoid(ff,tt,axis=0)

def main():
    from retarget_measured_jump import MeasuredJumpRig
    rig=MeasuredJumpRig();q=rig.b.q0.copy();rig.arm(q,.15,.15,.12)
    for s in ['r','l']:
        q[rig.m.jnt_qposadr[rig.b.joints['hip_flexion_'+s]]]=.025
        q[rig.m.jnt_qposadr[rig.b.joints['knee_angle_'+s]]]=.04
    q[rig.m.jnt_qposadr[rig.b.joints['flex_extension']]]=.015
    rig.b.pose(q);floor=np.mean([rig.d.site_xpos[rig.m.site('subject_sole_reference_'+s).id,2] for s in ['r','l']])
    q[2]-=floor;rig.b.pose(q)
    np.savez_compressed(OUT/'standing_pose.npz',times=np.array([0.]),qpos=q[None],body_xpos=rig.d.xpos[None].copy(),body_xmat=rig.d.xmat[None].copy())
    result={}
    jump_setup=json.loads((analysis_path(ROOT,'cmj')/'setup.json').read_text())
    for key,(a,b) in {'cmj':(jump_setup.get('movement_onset_s',.348),jump_setup['takeoff_s']),'squat_100':(.25,2.55)}.items():
        p=analysis_path(ROOT,key)
        z=np.load(p/'tensions.npz');v=load(ROOT,key)
        element=integrate(z['times'],z['force_N'],a,b)
        group=integrate(np.array(v['times']),overview(v)[[1,0,2,3]].T,a,b)
        result[key]=dict(start_s=a,end_s=b,duration_s=b-a,element_integral_Ns=element.tolist(),
            group_order=['QUADS','GLUTES','HAMSTRINGS','CALVES'],group_integral_mean_per_side_Ns=group.tolist(),
            group_mean_tension_per_side_N=(group/(b-a)).tolist(),names=v['names'],
            force_sha256=hashlib.sha256((p/'tensions.npz').read_bytes()).hexdigest())
    (OUT/'tension_exposure.json').write_text(json.dumps(dict(metric='Integral of nonnegative scalar muscle tension over physical time.',
        unit='N s',standing_pose_is_display_only=True,three_replays_counted_as_one_movement=True,
        not_energy_or_fatigue=True,windows='CMJ movement onset to takeoff; squat start of descent to end of ascent. No quiet lead-in or flight.',cases=result),indent=2))
    print('TENSION_EXPOSURE',json.dumps({k:dict(duration_s=v['duration_s'],groups_Ns=v['group_integral_mean_per_side_Ns']) for k,v in result.items()}),flush=True)

if __name__=='__main__':main()
