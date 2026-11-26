"""One-configuration grip fixture, without recomputing a movement."""
from pathlib import Path
import json
import numpy as np
from retarget_measured_jump import MeasuredJumpRig
from build_loaded_squat_power import SquatPath

ROOT=Path(__file__).resolve().parents[1]


def main():
    rig=MeasuredJumpRig();b,m,d=rig.b,rig.m,rig.d
    path=SquatPath(rig,100);q=path.pose_at_depth(0.);b.pose(q)
    folder=ROOT/'results/power_comparison/grip_fixture';folder.mkdir(parents=True,exist_ok=True)
    bar=d.xpos[path.chest]+d.xmat[path.chest].reshape(3,3)@path.bar_local
    bar_R=d.xmat[path.chest].reshape(3,3)@path.bar_reference_R.T
    np.savez_compressed(folder/'motion.npz',times=[0.],qpos=q[None],body_xpos=d.xpos[None].copy(),body_xmat=d.xmat[None].copy(),bar_position_m=bar[None],bar_rotation=bar_R[None],contact_positions_m=np.array([[p[2] for p in path.points]]))
    samples=[]
    for side in ['r','l']:
        hand=m.body('lunate_'+side).id;R=d.xmat[hand].reshape(3,3)
        for stem in ['lunate','secondmc','2proxph','midph2','distph2','thirdmc','3proxph','midph3','distph3']:
            bid=m.body(stem+'_'+side).id;p=d.xpos[bid].copy();local=R.T@(p-d.xpos[hand]);delta=p-bar;axis=bar_R[:,0];radial=delta-axis*(axis@delta)
            samples.append(dict(body=stem+'_'+side,hand_local_m=local.tolist(),radial_distance_to_bar_axis_m=float(np.linalg.norm(radial))))
    meta=dict(scale=b.report['global_normalization_scale'],body_names=[m.body(i).name for i in range(m.nbody)],phase_times_s=dict(bottom=0,start_ascent=0),bar_mass_kg=100,arm_fit=path.arm_fit,grip_samples=samples)
    (folder/'summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    b.pose(b.q0);np.savez_compressed(folder/'reference_pose.npz',body_xpos=d.xpos.copy(),body_xmat=d.xmat.copy())
    print(json.dumps(samples,indent=2))


if __name__=='__main__':main()
