"""Curated upload staging: source, scenarios, notices and small model examples."""
from pathlib import Path
import argparse,shutil,json,time,hashlib,re,subprocess,sys,zipfile,importlib.metadata,compileall
from jump_monitor_data import ROOT,read,atomic
DEST=ROOT/'github_release';OUT=ROOT/'deliverables/github_release'

def digest(path):
    with path.open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def copy(source,target):
    target=DEST/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/source,target)

def write(path,text):
    p=DEST/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')

def distributable(path):
    rel=path.relative_to(DEST)
    return rel.parts[0]not in ['.git','results','deliverables','archive','.venv','.tools']and '__pycache__'not in rel.parts

def main():
    DEST.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'scripts').iterdir():
        if p.is_file()and p.suffix in ['.py','.html','.js']:copy(p.relative_to(ROOT),Path('scripts')/p.name)
    for p in (ROOT/'scenarios').rglob('*.json'):copy(p.relative_to(ROOT),p.relative_to(ROOT))
    for p in [ROOT/'docs/assumptions.md',ROOT/'docs/modeling_references.json',ROOT/'docs/modeling_references.html',ROOT/'assets/fullbody/CREDITS.md']:
        if p.exists():copy(p.relative_to(ROOT),Path('docs/upstream_notices')/p.name)
    for p in (ROOT/'models').rglob('*'):
        if p.is_file()and(p.name.startswith(('LICENSE','SOURCE'))or p.name=='README.md'):copy(p.relative_to(ROOT),Path('docs/upstream_notices')/p.relative_to(ROOT))
    packages=['opensim','numpy','scipy','mujoco','myo-sim','osqp','quadprog','pillow','imageio','imageio-ffmpeg','torch','stable-baselines3','gymnasium']
    pinned=[]
    for package in packages:
        try:pinned.append(f'{package}=={importlib.metadata.version(package)}')
        except importlib.metadata.PackageNotFoundError:pass
    write(Path('requirements/native.txt'),'--extra-index-url https://download.pytorch.org/whl/cpu\n'+'\n'.join(pinned)+'\n');write(Path('requirements/analysis.txt'),'\n'.join(x for x in pinned if not x.startswith(('opensim==','myo-sim==','torch==','stable-baselines3==','gymnasium==')))+'\n');copy(Path('requirements.txt'),Path('requirements/legacy-opensim.txt'))
    state=read(ROOT/'deliverables/jump_landing_live/status.json');snapshot={k:state.get(k)for k in ['status','training_episodes','policy_updates','model','verified','completion_rule','objective_version']};snapshot['snapshot_at']=time.time();atomic(DEST/'examples/long_jump/learning_definition.json',read(ROOT/'results/controlled_broad_jump/landing_stability_v3/definition.json'));snapshot['video_snapshot_is_separate']=True;atomic(DEST/'examples/long_jump/latest_checked_snapshot.json',snapshot)
    for name in ['trial_provenance.json','preview_four_side.png','preview_four_oblique.png']:
        p=ROOT/'deliverables/jump_rl_study_draft_v5'/name
        if p.exists():copy(p.relative_to(ROOT),Path('examples/long_jump')/name)
    for name in ['delivery_check.json','metrics.json']:
        copy(Path('deliverables/jump_rl_study_draft_v4')/name,Path('examples/long_jump/v4_'+name))
    if not(DEST/'examples/long_jump/standing_long_jump_rl_final.mp4').exists():copy(Path('deliverables/jump_rl_study_draft_v4/standing_long_jump_rl_draft_v4.mp4'),Path('examples/long_jump/approved_draft_v4.mp4'))
    for name in ['metrics.json','joint_comparison.json','completion_audit.json','summary.png','motion_preview.png']:
        p=ROOT/'deliverables/iron_man_timing_v6'/name
        if p.exists():copy(p.relative_to(ROOT),Path('examples/box_jump_vs_squat')/name)
    import numpy as np
    inverse_source=ROOT/'deliverables/jump_rl_study_draft_v5'
    if not read(inverse_source/'source_manifest.json',{}).get('stable_three_metre_complete'):inverse_source=ROOT/'deliverables/jump_rl_study_draft_v4'
    fixture=np.load(inverse_source/'best_checked_muscles.npz');fixture_file=DEST/'examples/long_jump/inverse_problem.npz';np.savez_compressed(fixture_file,**{k:fixture[k]for k in ['times','R','target','passive_force_N','reference_N']});fixture_meta=read(inverse_source/'best_checked_muscles.json');atomic(DEST/'examples/long_jump/inverse_metadata.json',dict(force_names=fixture_meta['all_force_names'],muscle_paths=fixture_meta['muscle_count'],source_label=inverse_source.name,fixture_sha256=digest(fixture_file),motion_is_fixed=True,operator_reuse_for_changed_motion_is_invalid=True))
    # Only the rendered analysis portion is bundled; omit the photographic hook.
    import imageio_ffmpeg
    excerpt=DEST/'examples/box_jump_vs_squat/analysis_excerpt.mp4'
    if not excerpt.exists():subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-ss','6','-i',str(ROOT/'deliverables/iron_man_timing_v6/iron_man_squat_vs_box_jump_final_v6.mp4'),'-t','6','-an','-c:v','libx264','-crf','20','-threads','2','-y',str(excerpt)],check=True)
    if(DEST/'examples/long_jump/standing_long_jump_rl_final.mp4').exists():
        oldvideo=DEST/'examples/long_jump/approved_draft_v4.mp4'
        if oldvideo.exists()and digest(oldvideo)==digest(ROOT/'deliverables/jump_rl_study_draft_v4/standing_long_jump_rl_draft_v4.mp4'):oldvideo.unlink()
    # Reference recordings and source anatomy archives stay external.
    assets=[]
    for directory in ['assets','models','datasets']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file()and p.suffix.lower()in ['.glb','.mjb','.osim','.npz','.xml']and 'local_retention'not in str(p):assets.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    atomic(DEST/'asset_manifest.json',dict(external_assets=assets,base_anatomy_source='https://huggingface.co/datasets/myohub/myosuite-assets',base_anatomy_file='composite/fullbody_muscles_skin.glb',notices='docs/upstream_notices',external_source_archives_bundled=False,registered_compiled_example_included=(DEST/'examples/long_jump/registered_model.mjb.gz').exists()))
    write(Path('.gitignore'),'.venv/\n.tools/\n__pycache__/\n*.pyc\n*.log\n.env*\n/results/\n/deliverables/\n/archive/\n/github_release/\n/models/\n/assets/\n/datasets/\n*.mjb\n*.mjb.gz\n*.mp4\n*.npz\n*.npy\n*.zip\n')
    write(Path('LICENSE_NOTICE.md'),'# Licensing status\n\nA license for the original project code has not been selected. No license is granted to third-party assets by this repository. See `docs/upstream_notices` for the preserved anatomy and model notices. External source archives, recordings and photographic reference material are not bundled. One registered compressed mechanical input is included under the preserved upstream MyoSim notice. Rendered anatomy examples retain the applicable source attributions.\n')
    write(Path('README.md'),'''# Exercise Mechanics Lab

Rigid-body exercise simulations, conditional muscle-tension analysis, and portrait visualizations. The current study uses PPO to improve standing-long-jump posture and timing under fixed virtual joint-torque limits.

![Four fixed-side comparisons](examples/long_jump/preview_four_side.png)

## Included

- `scripts/`: physics, inverse analysis, training, rendering and archived experiment tools.
- `scenarios/`: movement and model assumptions.
- `examples/long_jump/`: approved draft, four-view layout and checked numerical snapshot.
- `examples/box_jump_vs_squat/`: final analysis excerpt, summary and metrics.
- `docs/`: method, storage policy, reproducibility notes and upstream credits.
- `asset_manifest.json`: external model paths, sizes and SHA-256 hashes.

## Quick checks

Python 3.12 was used locally. From the repository root:

```sh
python -m venv .venv
python -m pip install -r requirements/analysis.txt
python -m compileall -q scripts
```

Open the example MP4s directly, or run the registered mechanical example:

```sh
python scripts/download_example.py
python scripts/analyze_example.py --dt 0.0005
python scripts/analyze_example.py --dt 0.0005 --set push_s=0.24 --output results/faster_push
python scripts/analyze_muscle_example.py
```

The compact input includes a pinned-version compiled model, contact definitions and parameters; it requires no historical rollout files. The inverse example re-solves the supplied fixed-motion moment-arm/torque problem. It is a separately labelled snapshot until the final3m case replaces both inputs. After changing forward motion, do not reuse this fixed inverse operator: recompute muscle paths and targets with the anatomical pipeline. Use `--dt 0.000125` for the provided high-resolution source case. Rebuilding the anatomical rig or recomputing its full muscle paths additionally requires the external assets in the manifest. Read `docs/REPRODUCIBILITY.md` before running training. `requirements/legacy-opensim.txt` records dependencies of the earlier OpenSim experiments and is separate from the native MuJoCo/PPO environment.

## Optional fresh learning run

```sh
python -m pip install -r requirements/native.txt
python scripts/start_example_training.py --episodes 48 --hours 1
```

This starts a fresh small PPO run from the registered mechanics, not from the original learned checkpoint. It does not promise the historical3m result in48 trials. Its workers use the supplied model, so the historical registration corpus is unnecessary. Longer runs are opt-in; completed trials and actual policy updates are distinct.

## Current method

The reduced model has three unactuated sagittal root coordinates and eleven internal joint torques. A 9→64→64→58 policy selects a whole-jump parameter sequence: posture/timing and five families of three-knot equivalent torque drive. PD/state feedback executes the sequence. Contacts generate ground reactions; no takeoff impulse or airborne centre-of-mass path is prescribed. Internal air control changes posture, not the ballistic COM path.

The target is retained landing distance ≥3.000 m with stable recovery at each of 0.5, 0.25, 0.125 and 0.0625 ms. Training stops only after all four pass. Numerical eligibility also includes conservation/ROM checks and a heuristic pre-takeoff ground-reaction screen below 5.5 bodyweights; that screen is a modelling choice, not a universal human capacity law. It is kept fixed during the final search. Read `examples/long_jump/latest_checked_snapshot.json` for actual progress; the older draft video is a separate frozen snapshot, not a claim that the target has already been reached.

The final search also uses bounded physical pattern search and continuation from the best four-step-stable PPO sample. Proposals are ranked by the minimum retained distance across all four time steps, with stability and eligibility required in each. A successful screen is repeated from rest without launch caching. This supplemental numerical search is counted separately from PPO trials and policy updates; final example provenance states which method found the result. Body properties, torque limits, contact settings and qualification thresholds remain fixed.

Muscle forces are separate inverse postprocessing: 438 model paths, net torque/support equilibrium, nonnegative force sharing and continuity. These conditional forces do not drive the learner and are not measured activation, fatigue, injury or validated individual-human force predictions.

## Examples and trial counts

The four-case video layout uses baseline → two intermediate improvements → final qualified candidate. `PPO TRIAL n` counts completed V3 PPO rollouts including failures. Worker episode numbers, verification reruns, CEM and recentering are excluded. Earlier inherited training is recorded separately. See `trial_provenance.json`; an older preview's final slot is replaced only after the target is qualified.

## Project status

This is an experimental research/visualization workspace. Core modules use repository-relative paths, while some archived experiment scripts retain local historical paths. The portability report lists them. Raw rollouts, checkpoints, render frames, machine environments and external anatomy source archives are excluded from the upload folder. No remote upload is performed by preparing this folder.

See `LICENSE_NOTICE.md` and `docs/upstream_notices` before distributing derived assets.
''')
    write(Path('docs/REPRODUCIBILITY.md'),'''# Reproducibility

1. Install the pinned native requirements in a Python 3.12 environment.
2. Obtain upstream models separately; match `asset_manifest.json` and retain their notices. The anatomy GLB is from MyoHub; additional surfaces and licenses are documented in the credits.
3. Rebuild local registration/initial-reference outputs using the model-foundation and controlled-jump scripts. A compiled MJB is not a portable model interchange format across arbitrary MuJoCo versions.
4. Resume a local experiment only after its `definition.json`, checkpoint and required initial references exist. This source package does not contain the original many-gigabyte rollout corpus or learned policy checkpoint. The supplied registered mechanical example can be analyzed independently of that history.
5. `configure_jump_three_metre_finish.py` applies the qualified 3m stopping rule. `compose_jump_rl_draft_v5.py` adds synchronized four-side and four-oblique comparison views after the approved opening.

The included metrics are retained evidence; they do not by themselves reproduce training. Recentring and population-based initialization are recorded separately from PPO gradient updates. A finite best candidate is not a global optimum.
''')
    write(Path('docs/STORAGE.md'),'''# Storage policy

Generated model binaries use an immutable SHA-256 cache and per-case hard links. No shared model inode is overwritten. Changing storage does not change model bytes or dynamics.

`maintain_project_storage.py` first writes a reviewable cleanup plan. Referenced video cases, qualification reports, baseline inputs, policy checkpoints, settings, scores and histories are protected. `--apply` removes only unchanged generated files within the project results/deliverables trees. Final MP4s and named previews remain. Large intermediate model/motion arrays and obsolete frame caches are omitted from Git.

Future failed raw rollouts have a bounded per-worker retention window; selected video cases remain protected. This is not an archive of every raw historical attempt. Lightweight outcome/parameter records retain the optimization history.
''')
    if(DEST/'examples/long_jump/standing_long_jump_rl_final.mp4').exists():
        final=ROOT/'deliverables/jump_rl_study_draft_v5';copy(Path('deliverables/jump_rl_study_draft_v5/qualified_final_validation.json'),Path('examples/long_jump/final_validation.json'));copy(Path('deliverables/jump_landing_live/final_search_counts.json'),Path('examples/long_jump/search_counts.json'))
        text=(DEST/'README.md').read_text(encoding='utf-8');text=text.replace('- `examples/long_jump/`: approved draft, four-view layout and checked numerical snapshot.','- `examples/long_jump/`: finalized study video, four-view comparisons and reproducible checked inputs.')
        text=text.replace('It is a separately labelled snapshot until the final3m case replaces both inputs.','The forward example and fixed inverse fixture use the same finalized motion.')
        text=text.replace('Use `--dt 0.000125` for the provided high-resolution source case.','The exact source timestep is recorded in `prepared_model.json`; `--dt` also supports independent timestep comparisons.')
        checked=read(final/'qualified_final_validation.json');minimum=min(r['retained_distance_m']for r in checked['cases']);text=text.replace('# Exercise Mechanics Lab\n','# Exercise Mechanics Lab\n\nQualified virtual result: **'+f'{minimum:.4f} m minimum across four time steps, all stable'+ '**. See `examples/long_jump/final_validation.json` and `search_counts.json`.\n',1)
        text=text.replace('the older draft video is a separate frozen snapshot, not a claim that the target has already been reached.','the included video shows the finalized case, while its approved opening preserves an earlier training snapshot.');write(Path('README.md'),text)
    if not read(OUT/'publication_state.json',{}).get('release_assets_published',False):
        text=(DEST/'README.md').read_text(encoding='utf-8');notice='## Binary bundle status\n\nThe checked input/video ZIP is prepared locally. GitHub Release asset upload is pending; the default release download URL is not live yet. Source, configuration, numerical evidence and previews are available here. If you already have `exercise-mechanics-github.zip`, run `python scripts/download_example.py --archive /path/to/exercise-mechanics-github.zip`. Large inputs and videos are intentionally excluded from the Git tree.\n\n';text=text.replace('## Quick checks\n',notice+'## Quick checks\n',1);write(Path('README.md'),text)
    portability=[];secrets=[]
    token=re.compile(r'(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    for p in DEST.rglob('*'):
        if not p.is_file()or '.git'in p.parts or p.suffix.lower()not in ['.py','.html','.js','.md','.json','.txt']:continue
        text=p.read_text(encoding='utf-8',errors='replace')
        if token.search(text):secrets.append(str(p.relative_to(DEST)))
        if p.suffix=='.py':
            for i,line in enumerate(text.splitlines(),1):
                if 'D:/Exercise_mechanics'in line or 'D:\\\\Exercise_mechanics'in line:portability.append(dict(file=p.relative_to(DEST).as_posix(),line=i,historical_local_path=True))
    assert not secrets,('Possible credential material; release not staged',secrets)
    atomic(DEST/'docs/portability_report.json',dict(legacy_local_path_locations=portability,source_files_not_rewritten=True,full_portability_claim=False))
    assert compileall.compile_dir(DEST/'scripts',quiet=2)
    allfiles=[p for p in DEST.rglob('*')if p.is_file()and distributable(p)and p.name!='RELEASE_MANIFEST.json'];assert max(p.stat().st_size for p in allfiles)<50_000_000
    atomic(DEST/'RELEASE_MANIFEST.json',dict(prepared_at=time.time(),credential_scan_passed=True,syntax_check_passed=True,publication_target='https://github.com/tmdals1213-cyber/exercise-mechanics-lab',local_preparation_manifest=True,original_code_license_selected=False,files=[dict(path=p.relative_to(DEST).as_posix(),bytes=p.stat().st_size,sha256=digest(p))for p in allfiles]))
    if not(DEST/'.git').exists():subprocess.run(['git','init','-b','main',str(DEST)],check=True,stdout=subprocess.DEVNULL)
    tracked=subprocess.check_output(['git','-C',str(DEST),'ls-files'],text=True).splitlines()
    for name in tracked:
        if name.endswith(('.mp4','.mjb.gz')):subprocess.run(['git','-C',str(DEST),'rm','--cached','--ignore-unmatch','--',name],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(DEST),'add','.'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    archive=OUT/'exercise-mechanics-github.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
        for p in DEST.rglob('*'):
            if p.is_file()and distributable(p):z.write(p,Path('exercise-mechanics')/p.relative_to(DEST))
    atomic(OUT/'release_check.json',dict(passed=True,folder=str(DEST),archive=str(archive),archive_bytes=archive.stat().st_size,source_files=len(allfiles),credential_scan_passed=True,syntax_check_passed=True,no_large_raw_corpus=True,registered_example_input_included=True,not_uploaded=True));print('GITHUB_UPLOAD_FOLDER_READY',str(DEST),archive.stat().st_size,flush=True)

if __name__=='__main__':main()
