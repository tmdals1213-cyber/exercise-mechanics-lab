"""Export actual native states for a lightweight browser replay, without retargeting."""
from pathlib import Path
import argparse, json
import numpy as np
import mujoco
from jump_monitor_data import ROOT, MONITOR, atomic

def export(source, target=None, label=None):
    source = Path(source)
    report = json.loads((source / 'control_summary.json').read_text())
    z = np.load(source / 'control_motion.npz')
    m = mujoco.MjModel.from_binary_path(str(source / 'control_model.mjb'))
    selected = ['pelvis', 'torso', 'head', 'humerus_r', 'humerus_l', 'ulna_r', 'ulna_l',
                'radius_r', 'radius_l', 'hand_r', 'hand_l', 'femur_r', 'femur_l',
                'thirdmc_r', 'thirdmc_l', 'tibia_r', 'tibia_l', 'talus_r', 'talus_l', 'calcn_r', 'calcn_l', 'toes_r', 'toes_l']
    ids = [i for name in selected for i in range(1, m.nbody) if m.body(i).name == name]
    parents = []
    for bid in ids:
        p = int(m.body_parentid[bid])
        while p and p not in ids: p = int(m.body_parentid[p])
        parents.append(ids.index(p) if p in ids else -1)
    parents[[m.body(i).name for i in ids].index('torso')] = [m.body(i).name for i in ids].index('pelvis')
    stride = max(1, round(len(z['times']) / 360))
    ix = np.arange(0, len(z['times']), stride)
    obj = dict(label=label or source.name, source=str(source.relative_to(ROOT)),
               status=report['status'], stable=report['stable_landing'],
               valid_distance_m=report.get('valid_contact_distance_m'),
               first_heel_m=report.get('first_contact_heel_distance_m'),
               takeoff_s=report.get('takeoff_s'), touchdown_s=report.get('touchdown_s'),
               names=[m.body(i).name for i in ids], parents=parents,
               t=z['times'][ix].round(4).tolist(), xyz=z['body_xpos'][ix][:, ids].round(5).tolist(),
               com=z['com'][ix].round(5).tolist(),
               grf=(z['force_world_N'][ix, 2] / (87 * 9.80665)).round(4).tolist(),
               phase=z['phase'][ix].astype(int).tolist(),
               velocity=z['velocity'][ix].round(4).tolist(),
               muscle_forces_computed=False, world_fixed_coordinates=True)
    atomic(target or MONITOR / 'motion.json', obj)
    return obj

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('source'); ap.add_argument('--label')
    args = ap.parse_args(); export(args.source, label=args.label)
