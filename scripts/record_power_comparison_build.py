"""Record the exact sources, anatomical frames and checks of the power baseline."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import platform

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/power_comparison'


def main():
    report=json.loads((OUT/'summary.json').read_text())
    paths=list(OUT.glob('*'))
    paths += [ROOT/'scripts'/name for name in ['analyze_power_comparison.py','package_power_comparison.py',
              'verify_power_comparison.py','verify_power_comparison_viewer.cjs',
              'power_comparison_viewer.html','record_power_comparison_build.py']]
    paths += [ROOT/name for name in ['open_power_comparison.cmd','rebuild_power_baseline.cmd',
              'scenarios/jump_box_squat_power.json','results/measured_cmj/summary.json',
              'results/measured_jump_visualization/data.json',
              'results/measured_jump_analysis/s10_CMJ_t1/setup.json',
              'results/measured_jump_analysis/s10_CMJ_t1/motion.npz',
              'results/measured_jump_analysis/s10_CMJ_t1/tensions.npz']]
    paths += list((ROOT/'results/measured_jump_visualization/portrait_frames').glob('frame_*.png'))
    paths += [ROOT/rel for row in report['trials'] for rel in row['provenance']]
    files={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,
           sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(set(paths)) if p.is_file() and p.name!='build_manifest.json'}
    data=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
              packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','matplotlib','Pillow','imageio-ffmpeg']},
              rebuild='rebuild_power_baseline.cmd',
              scope='Audited CMJ COM-power baseline. Box-jump and loaded-squat outputs remain uncomputed.',files=files)
    (OUT/'build_manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('POWER_BUILD_RECORDED',len(files),flush=True)


if __name__=='__main__':main()
