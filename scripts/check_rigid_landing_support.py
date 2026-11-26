"""Foundation-only landing servo calibration. These gains are fixed in later RL."""
from concurrent.futures import ProcessPoolExecutor
import json
from controlled_broad_jump import ROOT, run, build

def case(job):
    i, support, damping, ankle = job
    p=json.loads((ROOT/'results/controlled_broad_jump/rigid_landing_initial_v29/best_candidate.json').read_text())['parameters']
    p.update(landing_support_gain_fraction=support,landing_support_damping=damping,landing_ankle=ankle)
    folder=ROOT/f'results/controlled_broad_jump/rigid_support_v30/case_{i:02d}'
    r,z=run(p,folder,dt=.0005)
    return dict(i=i,support=support,damping=damping,ankle=ankle,status=r['status'],distance=r.get('valid_contact_distance_m'),end_z=float(z['com'][-1,2]),parameters=p)

if __name__=='__main__':
    jobs=[(i,s,d,a)for i,(s,d,a)in enumerate([(s,d,a)for s in [.3,.6,1.]for d in [20.,40.]for a in [.1,.35]])]
    with ProcessPoolExecutor(max_workers=4)as pool:
        rows=list(pool.map(case,jobs))
    out=ROOT/'results/controlled_broad_jump/rigid_support_v30';(out/'summary.json').write_text(json.dumps(rows,indent=2))
    for row in rows:print(row['i'],row['support'],row['damping'],row['ankle'],row['status'],row['distance'],row['end_z'],flush=True)
