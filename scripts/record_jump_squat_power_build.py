"""Hash delivered motion/power/grasp inputs and artefacts after verification."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,platform,importlib.metadata

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'


def main():
    paths=list(OUT.rglob('*'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts').glob('*.cjs'))
    paths += [ROOT/'scripts'/n for n in ['jump_squat_power_viewer.html','power_comparison_viewer.html']]
    paths += [ROOT/'scenarios/muscle_force_literature.json']
    paths += [ROOT/n for n in ['models/grips/high_bar_28mm.json','scenarios/jump_box_squat_power.json','assets/fullbody/fullbody_muscles_skin.glb','assets/fullbody/torso_supplement.json','assets/fullbody/CREDITS.md','rebuild_jump_squat_power.cmd','rebuild_case_tensions.cmd','open_power_comparison.cmd','run.cmd','README.md']]
    for trial in ['s10_CMJ_t1','s10_CMJ_t2','s10_CMJ_t3']:
        paths += [ROOT/'results/measured_cmj'/trial/'mechanics_replay.npz',ROOT/'results/measured_cmj'/f'{trial}_measured.npz']
    paths += [ROOT/'results/measured_jump_visualization/data.json',ROOT/'results/measured_jump_analysis/s10_CMJ_t1/setup.json',ROOT/'results/measured_jump_analysis/s10_CMJ_t1/motion.npz']
    paths += list((ROOT/'results/measured_jump_visualization/anatomy/base').glob('frame_*.png'))
    files={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(paths)) if p.is_file() and p.name not in ['comparison_build_manifest.json','build_manifest.json'] and p.suffix!='.log'}
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','mujoco','opensim','matplotlib','Pillow','imageio-ffmpeg','osqp']},
                rebuild='rebuild_jump_squat_power.cmd',scope='Measured CMJ reference, prescribed loaded squats with checked contact/energy/grasp; box configuration diagnostic and rejected development trial, excluded from comparison.',files=files)
    (OUT/'comparison_build_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('COMPARISON_FILES_RECORDED',len(files),flush=True)


if __name__=='__main__':main()
