"""Supervise initial-motion search -> refined checks -> actual joint PPO training.

Restartable from on-disk checkpoints; keeps a heartbeat for the read-only HTML.
"""
from pathlib import Path
import argparse, ctypes, datetime, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from zoneinfo import ZoneInfo
from jump_monitor_data import ROOT, MONITOR, RUN, read, atomic, event

def tomorrow_morning():
    now=datetime.datetime.now(ZoneInfo('Asia/Seoul'));date=now.date()
    if now.hour>=9:date+=datetime.timedelta(days=1)
    return datetime.datetime.combine(date,datetime.time(9),ZoneInfo('Asia/Seoul')).timestamp()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',type=float);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--search-label',default='rigid_full_initial_v31');ap.add_argument('--training-label',default='ppo_full_overnight_v31');a=ap.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);MONITOR.mkdir(parents=True,exist_ok=True)
    # OS lock prevents a double-click on Resume from creating duplicate jobs.
    import msvcrt
    lock=(RUN/'supervisor.lock').open('a+');lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('A jump supervisor is already running.')
    config=read(RUN/'config.json',{});deadline=a.deadline or config.get('deadline')or tomorrow_morning()
    if deadline<=time.time():deadline=tomorrow_morning()
    config.update(deadline=deadline,workers=a.workers,search_label=a.search_label,training_label=a.training_label,maximum_initialization_generations=4000,maximum_training_episodes=65536)
    atomic(RUN/'config.json',config)
    (RUN/'stop.request').unlink(missing_ok=True)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    state=dict(stage='initialization',started_at=time.time(),deadline=deadline,search_label=a.search_label,training_label=a.training_label,pid=os.getpid(),message='Jointly fitting 43 posture and timing parameters for a stable ≥ 2 m initial solution.')
    proc=None;stream=None;preview=None;preview_stream=None;last_preview=0.;preview_trial=None;restarts=0
    search=ROOT/'results/controlled_broad_jump'/a.search_label;train=ROOT/'results/controlled_broad_jump'/a.training_label;verified=ROOT/'results/controlled_broad_jump/two_metre_initial_solution_verified'
    base=ROOT/'results/controlled_broad_jump/rigid_landing_initial_v29/best_candidate.json'
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    def status(**values):
        state.update(values);state['heartbeat']=time.time();state['child_pid']=proc.pid if proc and proc.poll()is None else None;atomic(RUN/'status.json',state)
    def launch(script,args,logname):
        nonlocal proc,stream
        if stream:stream.close()
        stream=(RUN/logname).open('a',encoding='utf-8')
        proc=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        event('process_started',message=script,child_pid=proc.pid)
    def stop_child():
        if proc and proc.poll()is None:
            target=train if state['stage']=='training'else search
            atomic(target/'stop.request',dict(time=time.time()))
    def update_preview():
        nonlocal preview,preview_stream,last_preview,preview_trial
        if preview is not None and preview.poll()is not None:
            preview_stream.close()
            if preview.returncode==0:
                from export_jump_monitor_motion import export
                export(ROOT/'results/controlled_broad_jump/monitor_live_preview',label=f'{a.search_label} / trial {preview_trial}')
            preview=None
        if preview is not None:return
        if state['stage']=='training':
            cases=[d for d in train.glob('worker_*/best_episode')if(d/'control_summary.json').exists()]
            if cases and time.time()-last_preview>60:
                best=max(cases,key=lambda d:(read(d/'control_summary.json',{}).get('valid_contact_distance_m')or -1))
                from export_jump_monitor_motion import export
                export(best,label='PPO best valid sample / provisional until refined replay');last_preview=time.time()
            return
        best=read(search/'best_candidate.json',{});trial=best.get('trial')
        if trial is None or trial==preview_trial or time.time()-last_preview<60:return
        preview_trial=trial;last_preview=time.time()
        atomic(RUN/'preview_parameters.json',best)
        preview_stream=(RUN/'motion_preview.log').open('w',encoding='utf-8')
        preview=subprocess.Popen([sys.executable,str(ROOT/'scripts/controlled_broad_jump.py'),'--parameters-file',str(RUN/'preview_parameters.json'),'--label','monitor_live_preview','--dt','.0005'],stdout=preview_stream,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    event('supervisor_started',message='Overnight native initialization, refined replays and joint PPO',deadline=deadline)
    try:
        while True:
            status()
            try:update_preview()
            except (OSError,ValueError,KeyError,zipfile.BadZipFile)as exc:
                # A snapshot is dispensable; concurrent atomic best-case saves
                # must never stop the actual experiment.
                event('preview_retry',error=repr(exc));last_preview=time.time()
            if(RUN/'stop.request').exists()or time.time()>=deadline:
                stop_child();status(message='Stopping after the current computation block; checkpoints are preserved.')
                if not proc or proc.poll()is not None:
                    status(stage='stopped',message='Checkpointed. Resume restarts from saved search/policy state.');break
                time.sleep(2);continue
            proof=read(verified/'verification.json',{})
            if proof.get('status')=='passed':
                if state['stage']!='training':
                    if proc and proc.poll()is None:stop_child();time.sleep(2);continue
                    status(stage='training',message='Verified ≥ 2 m initial motion. PPO jointly learns all 43 posture/timing outputs.')
                    event('initial_motion_verified',message=f"{proof['cases'][-1]['valid_contact_distance_m']:.3f} m at finest timestep")
                if proc is None or proc.poll()is not None:
                    if proc is not None and proc.returncode!=0:restarts+=1;event('training_restart',message=f'Exit {proc.returncode}; restart {restarts}')
                    if restarts>4:raise RuntimeError('Training failed repeatedly; see training.log')
                    source=proof['cases'][-1]['folder'];args=['--source',source,'--capacity-file',verified/'fixed_virtual_capacity.json','--dt',proof['cases'][0]['dt_s'],'--label',a.training_label,'--deadline',deadline,'--workers',a.workers]
                    checkpoints=sorted([p for p in train.glob('policy_*.zip')if'_pending'not in p.stem and zipfile.is_zipfile(p)],key=lambda p:p.stat().st_mtime)
                    if checkpoints:args+=['--resume',checkpoints[-1]]
                    if read(train/'live_training.json',{}).get('status')=='episode_budget_completed':status(stage='complete',message='PPO episode budget completed; finite best sample saved.');break
                    launch('train_jump_overnight.py',args,'training.log')
            elif state['stage']=='verification':
                if proc is not None and proc.poll()is not None:
                    target=ROOT/'results/controlled_broad_jump'/state['verification_label'];report=read(target/'verification.json',{})
                    if proc.returncode==0 and report.get('status')=='passed':
                        # Publish only a completed passing bundle. Case paths
                        # intentionally keep pointing at preserved original runs.
                        shutil.copytree(target,verified,dirs_exist_ok=True,ignore=lambda directory,names:['verification.json']if Path(directory)==target else [])
                        atomic(verified/'verification.json',report)
                        event('refinement_passed',message=state['verification_label'])
                    else:
                        candidate=read(search/'feasible_candidate.json',{});digest=hashlib.sha256(json.dumps(candidate.get('parameters',{}),sort_keys=True).encode()).hexdigest();blocked=read(search/'rejected_parameter_hashes.json',[]);blocked.append(digest);atomic(search/'rejected_parameter_hashes.json',blocked)
                        (search/'feasible_candidate.json').unlink(missing_ok=True)
                        event('refinement_rejected',message='Retained diagnostic bundle; initialization resumes without counting this candidate as a record.')
                        status(stage='initialization',message='Refined replay rejected a candidate; continuing joint initialization.')
                    proc=None
            else:
                if proc is None or proc.poll()is not None:
                    feasible=read(search/'feasible_candidate.json')
                    if feasible:
                        label=f"rigid_verification_v31_trial_{feasible['trial']:06d}"
                        status(stage='verification',verification_label=label,message='Uncached replays at 0.5 / 0.25 / 0.125 ms, mechanics and energy checks.')
                        launch('finalize_two_metre_initial_solution.py',['--search-label',a.search_label,'--output-label',label],'verification.log')
                    else:
                        if proc is not None and proc.returncode!=0:restarts+=1;event('search_restart',message=f'Exit {proc.returncode}; restart {restarts}')
                        if restarts>4:raise RuntimeError('Initialization failed repeatedly; see initialization.log')
                        search.mkdir(parents=True,exist_ok=True);(search/'stop.request').unlink(missing_ok=True)
                        generations=4000
                        progress=read(search/'progress.json',{})
                        if progress.get('generation',0)>=3999:generations=progress['generation']+4001
                        args=['--label',a.search_label,'--base',base,'--generations',generations,'--workers',a.workers,'--deadline',deadline]
                        if(search/'optimizer_state.npz').exists():args+=['--resume']
                        launch('search_rigid_jump_overnight.py',args,'initialization.log')
            time.sleep(2)
    except Exception as exc:
        stop_child();event('supervisor_error',error=repr(exc));status(stage='error',message=repr(exc));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if stream:stream.close()
        lock.close()

if __name__=='__main__':main()
