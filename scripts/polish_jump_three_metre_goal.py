"""Same-domain local policy reseed at the latest qualified candidate."""
import sys,time,subprocess,os,hashlib,json
from jump_monitor_data import ROOT,read,atomic
RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3';WEB=ROOT/'deliverables/jump_landing_live'

def main():
    (RUN/'stop.request').write_text('Checkpoint for local policy polish around latest qualified2.987m candidate; physical limits unchanged.\n',encoding='utf-8');start=time.time()
    while time.time()-start<1200:
        state=read(RUN/'live.json')
        if state['status']in ['checkpointed','goal_found','budget_finished','failed']:break
        time.sleep(2)
    else:raise RuntimeError('No checkpoint; no configuration changed')
    if state['status']=='goal_found':print('GOAL_ALREADY_QUALIFIED',flush=True);return
    assert state['status']=='checkpointed',state['status'];time.sleep(4);archive=RUN/f'before_goal_polish_{state["training_episodes"]:06d}';archive.mkdir(exist_ok=False)
    for name in ['definition.json','live.json','verified_best.json','policy_latest.zip']:(archive/name).write_bytes((RUN/name).read_bytes())
    config=read(RUN/'definition.json');verified=read(RUN/'verified_best.json');seed=next(r for r in verified['cases']if abs(r['dt_s']-.00025)<1e-10);atomic(archive/'seed.json',seed);config['recenter_std_groups']=dict(posture_timing=.025,effort=.04,postcontact=.025);config['exploration_min_std_groups']=dict(posture_timing=.02,effort=.03);atomic(RUN/'definition.json',config);(RUN/'stop.request').unlink()
    args=[sys.executable,str(ROOT/'scripts/train_landing_stability.py'),'--label','landing_stability_v3','--resume','--episodes','16000','--hours','4','--reseed-parameters',str(archive/'seed.json'),'--reseed-source-algorithm','Latest3dt-qualified2.987m candidate, narrower local proposal distribution; not a PPO gradient update']
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with(WEB/'goal_polish_stdout.log').open('a',encoding='utf-8')as stdout,(WEB/'goal_polish_stderr.log').open('a',encoding='utf-8')as stderr:process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    record=dict(at=time.time(),pid=process.pid,archive=str(archive),original_deadline_preserved=True,physical_caps_and_absolute_bounds_unchanged=True,seed_min_retained_m=min(r['retained_distance_m']for r in verified['cases']),sampling_std=config['recenter_std_groups'],sampling_std_floors=config['exploration_min_std_groups'],not_a_PPO_update=True);atomic(WEB/'goal_polish_status.json',record);print('LOCAL_GOAL_POLISH_STARTED',process.pid,flush=True)

if __name__=='__main__':main()
