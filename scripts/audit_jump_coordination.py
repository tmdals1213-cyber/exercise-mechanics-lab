"""Independent geometric screening; equation balance is not human validation.

Signed foot ordering uses the instantaneous hip axis, not the camera/world x.
Shank centreline distance is a geometric diagnostic, not mesh collision proof.
"""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def segment_distance(a, b, c, d):
    u, v, w = b-a, d-c, a-c
    U = np.column_stack((u, -v))
    x = np.linalg.lstsq(U, -w, rcond=None)[0]
    candidates = [x] if np.all((x >= 0) & (x <= 1)) else []
    for s in (0., 1.):
        candidates.append(np.array([s, np.clip(v @ (w+s*u)/(v@v), 0, 1)]))
    for t in (0., 1.):
        candidates.append(np.array([np.clip(-u @ (w-t*v)/(u@u), 0, 1), t]))
    return min(float(np.linalg.norm(w+U@x)) for x in candidates)


def audit(folder):
    z = np.load(folder/'motion.npz')
    meta = json.loads((folder/'summary.json').read_text())
    names = meta['body_names']
    pos = z['body_xpos']
    body = lambda name: pos[:, names.index(name)]
    lateral = body('femur_l')-body('femur_r')
    lateral /= np.linalg.norm(lateral, axis=1)[:, None]
    points = z['contact_positions_m']
    assert points.shape[1] == 12, 'Expected six ordered sole points per side'
    right, left = points[:, :6], points[:, 6:]
    separation = np.einsum('ij,ij->i', left.mean(1)-right.mean(1), lateral)
    gap = (np.einsum('ikj,ij->ik', left, lateral).min(1)
           - np.einsum('ikj,ij->ik', right, lateral).max(1))
    shanks = np.array([segment_distance(a, b, c, d) for a, b, c, d in zip(
        body('tibia_r'), body('talus_r'), body('tibia_l'), body('talus_l'))])
    crossed = separation <= 0
    record = dict(
        status='rejected_foot_crossing' if crossed.any() else 'screen_passed_not_human_validated',
        source_motion='motion.npz',
        foot_order_preserved=bool(not crossed.any()),
        minimum_signed_foot_centre_separation_m=float(separation.min()),
        minimum_projected_sole_gap_m=float(gap.min()),
        minimum_shank_centreline_distance_m=float(shanks.min()),
        first_foot_crossing_time_s=float(z['times'][np.flatnonzero(crossed)[0]]) if crossed.any() else None,
        human_motion_validated=False,
        full_mesh_self_collision_checked=False,
        limitation='A passing geometric screen is necessary but does not validate realistic coordination or surface clearance.')
    (folder/'coordination_audit.json').write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.folder), indent=2))
