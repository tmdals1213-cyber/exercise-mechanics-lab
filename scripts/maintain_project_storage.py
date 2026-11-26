"""Reviewable, reference-aware removal of generated intermediate data only."""
from pathlib import Path
import argparse,json,time,os,shutil,re
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/storage_maintenance';RULES=ROOT/'datasets/local_retention/protected_cases.json'
HEAVY={'.mjb','.npz','.npy','.bin'}
_worker_failures={}
_last_cache_cleanup=0.

def referenced_directories():
    keep=set()
    def visit(value):
        if isinstance(value,dict):
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
        elif isinstance(value,str)and len(value)<600:
            p=Path(value)
            if not p.is_absolute():return
            try:p=p.resolve()
            except OSError:return
            if not p.is_relative_to(ROOT.resolve()):return
            # A run root is not a request to preserve every historical episode.
            if p.is_file():p=p.parent
            if p.is_dir()and any((p/n).exists()for n in ['control_summary.json','motion.npz','control_motion.npz','reference_pose.npz']):keep.add(p)
    for p in (ROOT/'deliverables').rglob('source_manifest.json'):visit(read(p,{}))
    for p in [ROOT/'docs/standing_long_jump_working_state.json',ROOT/'results/controlled_broad_jump/landing_stability_v3/live.json']:
        value=read(p,{})
        if p.name=='live.json':value={k:value.get(k)for k in ['stable_candidates','verified','best_stable','best_distance']}
        visit(value)
    run=ROOT/'results/controlled_broad_jump'
    for p in run.rglob('validation_*.json'):visit(read(p,{}))
    for pattern in ['**/verified_best.json','**/best_stable.json','**/*leader*.json','**/seed.json','**/extension_seed*.json']:
        for p in run.glob(pattern):visit(read(p,{}))
    # Existing executable examples and anatomical rendering references are inputs.
    for p in (ROOT/'scripts').glob('*.py'):
        for rel in re.findall(r"['\"]((?:results|deliverables)/[^'\"\n]+)['\"]",p.read_text(encoding='utf-8',errors='replace')):
            candidate=ROOT/rel
            if candidate.is_file()and candidate.suffix.lower()in HEAVY:keep.add(candidate.parent.resolve())
            elif candidate.is_dir()and any((candidate/n).exists()for n in ['control_summary.json','reference_pose.npz']):keep.add(candidate.resolve())
    for rel in ['results/controlled_broad_jump/distance_stability_pilot_v1/stable_reference','deliverables/iron_man_contours_v5/render_cases/jump','deliverables/iron_man_contours_v5/render_cases/squat']:
        p=ROOT/rel
        if p.exists():keep.add(p.resolve())
    return keep

def protected(path,keep):return path in keep or not keep.isdisjoint(path.parents)

def plan():
    OUT.mkdir(parents=True,exist_ok=True);keep=referenced_directories();now=time.time();files=[];run=ROOT/'results/controlled_broad_jump';active=run/'landing_stability_v3'
    for p in run.rglob('*'):
        if not p.is_file()or p.suffix.lower()not in HEAVY:continue
        # These small arrays are the actual neural-network/gradient evidence
        # used by the approved learning animation, not per-episode models.
        if p.name.startswith(('network_update_','rollout_batch_update_')):continue
        resolved=p.resolve()
        if protected(resolved,keep):continue
        # Do not touch a live episode being written, including incomplete outputs.
        if active in p.parents and now-p.stat().st_mtime<600:continue
        st=p.stat();files.append(dict(path=str(resolved),size=st.st_size,mtime_ns=st.st_mtime_ns,reason='Unreferenced generated intermediate model/motion; scores, configurations and histories retained.'))
    legacy=['instagram_launch','iron_man_comparison','iron_man_comparison_final','iron_man_contours_v5','iron_man_timing_v6','iron_man_visual_hook_v3','iron_man_layout_v4','yun_sungbin_study','yun_160_box_jump_review','yun_reference_motion_review','yun_muscle_comparison_review','jump_motion_development']
    for name in legacy:
        for p in (ROOT/'deliverables'/name).rglob('frame_*.png'):
            if protected(p.resolve(),keep):continue
            st=p.stat();files.append(dict(path=str(p.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,reason='Reproducible frame cache; finished MP4s and named preview images retained.'))
    for name in ['athletic_box_jump','reference_box_jump','squat_comparison','power_comparison']:
        for p in (ROOT/'results'/name).rglob('*'):
            if not p.is_file()or p.suffix.lower()not in HEAVY or p.name in ['reference_pose.npz','standing_pose.npz']:continue
            if protected(p.resolve(),keep):continue
            st=p.stat();files.append(dict(path=str(p.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,reason='Legacy box/squat generated intermediate arrays; final media, statistics, plots and retained case inputs kept.'))
    rules=dict(protected_case_directories=sorted(str(p)for p in keep),future_failed_raw_per_worker=24,keep_scores_and_history=True);atomic(RULES,rules);result=dict(created_at=now,project_root=str(ROOT.resolve()),protected_directories=len(keep),files=files,logical_bytes=sum(f['size']for f in files),retained='All source-manifest cases, qualification evidence, executable input references, policies, parameter settings, score/history JSON, final MP4s and named previews.');atomic(OUT/'cleanup_plan.json',result);print('STORAGE_PLAN',len(files),round(result['logical_bytes']/1e9,2),'GB; protected',len(keep),flush=True)

def apply():
    p=read(OUT/'cleanup_plan.json');assert p and Path(p['project_root']).resolve()==ROOT.resolve();keep={Path(x).resolve()for x in read(RULES)['protected_case_directories']};before=shutil.disk_usage(ROOT).free;removed=0;skipped=0
    with(OUT/'removed_files.jsonl').open('a',encoding='utf-8')as audit:
        for row in p['files']:
            target=Path(row['path']).resolve();assert target.is_relative_to((ROOT/'results').resolve())or target.is_relative_to((ROOT/'deliverables').resolve());assert not protected(target,keep)
            if not target.is_file():skipped+=1;continue
            st=target.stat()
            if st.st_size!=row['size']or st.st_mtime_ns!=row['mtime_ns']:skipped+=1;continue
            target.unlink();audit.write(json.dumps(dict(at=time.time(),**row))+'\n');removed+=1
    result=dict(passed=True,at=time.time(),removed_files=removed,skipped_changed_files=skipped,free_before_bytes=before,free_after_bytes=shutil.disk_usage(ROOT).free,final_videos_and_referenced_cases_retained=True,retention_rules=str(RULES));atomic(OUT/'cleanup_result.json',result);print('STORAGE_CLEANED',json.dumps(result),flush=True)

def prune_worker_failures(worker,latest=None,row=None):
    global _last_cache_cleanup
    rules=read(RULES,{})
    if not rules:return
    keep={Path(p).resolve()for p in rules['protected_case_directories']};key=str(worker)
    if key not in _worker_failures:
        failed=[]
        for p in Path(worker).glob('episode_*/score.json'):
            if not(p.parent/'control_motion.npz').exists():continue
            prior=read(p)
            if prior and not prior['stable']and not protected(p.parent.resolve(),keep):failed.append(p.parent)
        _worker_failures[key]=sorted(failed)
    failed=_worker_failures[key]
    if latest is not None and row and not row['stable']and not protected(Path(latest).resolve(),keep)and Path(latest)not in failed:failed.append(Path(latest))
    excess=max(0,len(failed)-int(rules['future_failed_raw_per_worker']))
    for folder in failed[:excess]:
        assert folder.resolve().is_relative_to(Path(worker).resolve())
        for name in ['control_model.mjb','control_motion.npz']:
            target=folder/name
            if target.is_file():target.unlink()
    if excess:del failed[:excess]
    if time.time()-_last_cache_cleanup>60.:
        from model_file_cache import prune_unreferenced
        prune_unreferenced();_last_cache_cleanup=time.time()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args();apply()if args.apply else plan()
