"""Existing anatomical display, driven by the exact new native dynamics states."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,subprocess,sys
import numpy as np
from jump_monitor_data import ROOT,read,atomic

def main():
    solution=read(ROOT/'results/controlled_broad_jump/effort_launch_seed_v2/initial_solution.json');source=Path(solution['cases'][-1]['folder']);label='effort_seed_anatomy_v1';out=ROOT/'deliverables/controlled_jump_pilot'/label
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_control_jump_review.py'),str(source),'--label',label],check=True)
    original=np.load(source/'control_motion.npz');copy=np.load(out/'motion.npz')
    assert np.array_equal(original['qpos'],copy['qpos'])and np.array_equal(original['body_xpos'],copy['body_xpos'])
    atomic(out/'render_scope.json',dict(physics_source=str(source),physics_model=str(source/'control_model.mjb'),display_asset='assets/fullbody/fullbody_muscles_skin.glb',native_pose_copied_exactly=True,skin_and_muscle_mesh_are_display_geometry=True,individual_muscle_tensions_available=False,force_contours_enabled=False,model_scope='Sagittal rigid segments with 11 internal joint torques and3 unactuated root coordinates. Neither muscle meshes nor their deformation are the force solver.',end_of_motion='First native foot contact. A0.9s held display pose is not simulated recovery.'))
    blender=ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'
    def render(kind):
        script='render_fixed_side_jump_blender.py'if kind=='side'else'render_control_jump_blender.py';args=[str(blender),'--background','--threads','4','--python',str(ROOT/'scripts'/script),'--','--kg','0','--folder',str(out),'--analysis-study','--skip-existing']
        args+=['--world-side','--ortho-width','3.9','--camera-forward','1.45']if kind=='side'else['--matched-person-scale','--floor-review']
        with(out/f'{kind}_render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    with ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(render,['side','tracked']))
    subprocess.run([sys.executable,str(ROOT/'scripts/compose_fixed_side_multiview.py'),'--label',label],check=True)
    print('EFFORT_ANATOMICAL_PREVIEW_READY',out/'fixed_side_multiview_review.mp4',flush=True)

if __name__=='__main__':main()
