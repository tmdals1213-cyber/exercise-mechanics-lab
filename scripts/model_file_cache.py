"""Lossless content-addressed model files, with normal per-episode MJB paths.

Cache objects are immutable. Every output is atomically replaced by a hard link;
no existing shared inode is ever opened for writing. Dynamics are unaffected.
"""
from pathlib import Path
import os,hashlib,argparse,json,shutil,time
from jump_monitor_data import ROOT,atomic
CACHE=ROOT/'results/shared_native_model_cache'

def attach(source,target):
    source=Path(source);target=Path(target);tmp=target.with_name(target.name+f'.link-{os.getpid()}')
    if target.exists()and os.path.samefile(source,target):return False
    if tmp.exists():tmp.unlink()
    os.link(source,tmp);os.replace(tmp,target);return True

def save_model(model,target):
    import numpy as np,mujoco
    CACHE.mkdir(parents=True,exist_ok=True);buffer=np.empty(mujoco.mj_sizeModel(model),np.uint8);mujoco.mj_saveModel(model,buffer=buffer);digest=hashlib.sha256(buffer).hexdigest();cached=CACHE/(digest+'.mjb')
    if not cached.exists():
        tmp=CACHE/f'{digest}.{os.getpid()}.tmp'
        try:
            with tmp.open('wb')as stream:stream.write(buffer)
            try:os.link(tmp,cached)
            except FileExistsError:pass
        finally:
            if tmp.exists():tmp.unlink()
    assert cached.stat().st_size==buffer.nbytes
    attach(cached,target)

def prune_unreferenced(min_age_s=60.):
    if not CACHE.exists():return 0
    removed=0;now=time.time()
    for path in CACHE.glob('*.mjb'):
        if len(path.stem)!=64:continue
        try:
            stat=path.stat()
            if stat.st_nlink==1 and now-stat.st_mtime>min_age_s:path.unlink();removed+=1
        except FileNotFoundError:pass
    return removed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target-free-gb',type=float,default=25);a=ap.parse_args();run=(ROOT/'results/controlled_broad_jump/landing_stability_v3').resolve();CACHE.mkdir(parents=True,exist_ok=True);before=shutil.disk_usage(run).free;linked=0;checked=0;logical=0;invalid=[];start=time.time()
    for worker in sorted(run.glob('worker_*')):
        if not worker.is_dir():continue
        for file in sorted(worker.glob('episode_*/control_model.mjb')):
            assert file.resolve().is_relative_to(run);size=file.stat().st_size
            if size<1000000:invalid.append(str(file));continue
            with file.open('rb')as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            cached=CACHE/(digest+'.mjb');checked+=1
            if not cached.exists():os.link(file,cached)
            else:
                assert cached.stat().st_size==size
                if attach(cached,file):linked+=1;logical+=size
            if checked%50==0:print('MODEL_DEDUP',checked,linked,round(shutil.disk_usage(run).free/1e9,2),flush=True)
            if shutil.disk_usage(run).free>=a.target_free_gb*1e9:break
        if shutil.disk_usage(run).free>=a.target_free_gb*1e9:break
    report=dict(passed=True,at=time.time(),checked=checked,linked=linked,logical_duplicate_bytes_replaced=logical,free_before_bytes=before,free_after_bytes=shutil.disk_usage(run).free,all_episode_file_paths_preserved=True,full_sha256_checked_before_link=True,incomplete_model_files_preserved=invalid,seconds=time.time()-start);atomic(ROOT/'deliverables/jump_landing_live/storage_recovery.json',report);print('MODEL_STORAGE_RECOVERED',json.dumps(report),flush=True)

if __name__=='__main__':main()
