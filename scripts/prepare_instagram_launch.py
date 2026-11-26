"""Stage existing checked mechanics for a simpler first reel; no new force solve."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np
from launch_sources import analysis_path,render_root

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/instagram_launch'


def main():
    OUT.mkdir(parents=True,exist_ok=True);provenance={}
    for key in ['cmj','squat_100']:
        analysis=analysis_path(ROOT,key);source=analysis if key=='cmj' else analysis.parent
        dest=render_root(ROOT)/key;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'motion.npz',dest/'motion.npz')
        visual=dest/'analysis/visualization';visual.mkdir(parents=True,exist_ok=True)
        for name in ['data.json','geometry.npz']:
            shutil.copy2(analysis/'visualization'/name,visual/name)
        data=json.loads((visual/'data.json').read_text())
        if key=='cmj':
            geo=np.load(visual/'geometry.npz');setup=json.loads((analysis/'setup.json').read_text())
            np.savez_compressed(dest/'reference_pose.npz',body_xpos=geo['rest_body_xpos'],body_xmat=geo['rest_body_xmat'])
            meta=dict(movement='cmj_replay',scale=data['scale'],body_names=data['body_names'],
                phase_times_s=dict(bottom=setup['countermovement_end_s'],start_ascent=setup['countermovement_end_s'],end_ascent=setup['takeoff_s']))
            (dest/'summary.json').write_text(json.dumps(meta,indent=2))
        else:
            for name in ['summary.json','reference_pose.npz']:shutil.copy2(source/name,dest/name)
        provenance[key]=dict(motion_sha256=hashlib.sha256((analysis/'motion.npz').read_bytes()).hexdigest(),
            force_sha256=hashlib.sha256((analysis/'tensions.npz').read_bytes()).hexdigest(),
            source=str(source.relative_to(ROOT)).replace('\\','/'))
    (OUT/'physics_provenance.json').write_text(json.dumps(dict(existing_force_values_changed=False,
        additional_jump_heights_solved=True,jump_is_prescribed_scenario=True,cases=provenance),indent=2))
    print('LAUNCH_RENDER_INPUTS_READY',flush=True)


if __name__=='__main__':main()
