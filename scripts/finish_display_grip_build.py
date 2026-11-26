"""Finish the three already-running display-grip renders and verify delivery."""
from pathlib import Path
import json,subprocess,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'

def run(args,log):
    with (OUT/log).open('w',encoding='utf-8') as f:
        subprocess.run([str(a) for a in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    deadline=time.monotonic()+2400
    while True:
        done=[]
        for kg in [60,100,140]:
            folder=OUT/f'squat_{kg}';log=(folder/'render_display_grip.log').read_text(errors='replace')
            if 'Traceback' in log:raise RuntimeError(f'Render failed: {kg}')
            n=round(float(np.load(folder/'motion.npz')['times'][-1])*30)+1
            done.append(f'SQUAT_RENDER {kg} {n-1} {n}' in log and 'Blender quit' in log)
        if all(done):break
        if time.monotonic()>deadline:raise RuntimeError('Render completion deadline exceeded')
        time.sleep(5)
    blender=ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'
    run([blender,'--background','--python',ROOT/'scripts/render_squat_power_blender.py','--','--kg','100','--folder',OUT/'grip_fixture','--grip'],'grip_fixture/final_wide_render.log')
    for script in ['package_jump_squat_power.py','verify_jump_squat_power.py']:
        run([sys.executable,ROOT/'scripts'/script],script+'.log')
    run(['node',ROOT/'scripts/verify_jump_squat_power_viewer.cjs'],'viewer_final.log')
    run([sys.executable,ROOT/'scripts/record_jump_squat_power_build.py'],'manifest_final.log')
    (OUT/'display_grip_build_status.json').write_text(json.dumps(dict(status='passed',native_physics_changed=False,rendered_conditions=[60,100,140],checks=['display geometry clearance','native anchor invariance','mechanics','encoded videos','browser']),indent=2))
    print('DISPLAY_GRIP_DELIVERY_PASSED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        (OUT/'display_grip_build_status.json').write_text(json.dumps(dict(status='failed',reason=str(e)),indent=2));raise
