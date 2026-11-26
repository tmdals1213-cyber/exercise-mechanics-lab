"""Hash the delivered measured-motion analysis and its implementation."""
import json,hashlib,platform,importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/measured_jump_visualization'

def main():
    paths=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts').glob('*.cjs'))+[ROOT/'scripts/measured_jump_viewer.html']
    for folder in ['results/measured_jump_analysis/s10_CMJ_t1','results/measured_jump_visualization','datasets/cmj_calisti2025','results/knee_model_validity','datasets/knee_model_reference']:
        paths+=list((ROOT/folder).rglob('*'))
    paths += [ROOT/n for n in ['README.md','run.cmd','open_measured_jump.cmd','rebuild_measured_jump.cmd','open_jump_contact_lab.cmd','open_synthetic_jump.cmd','models/rmr/TSM_subject_noWeight.osim','assets/fullbody/fullbody_muscles_skin.glb']]
    paths += [ROOT/'results/measured_cmj'/n for n in ['reference_model/scaled_model.osim','s10_CMJ_t1/mechanics_replay.npz','s10_CMJ_t1_measured.npz','summary.json','input_checks.json']]
    files={}
    for p in sorted(set(paths)):
        if p.is_file() and p.name!='build_manifest.json' and p.suffix not in ['.log']:
            files[p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','mujoco','myo-sim','opensim','ezc3d','matplotlib','quadprog']}
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),packages=versions,rebuild='rebuild_measured_jump.cmd',scope='Conditional internal-joint muscle tensions from measured-motion retarget. Disclosed root force/moment and foot-registration residuals remain. No human muscle-force validation.',files=files)
    (OUT/'build_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(recorded_files=len(files))))

if __name__=='__main__':main()
