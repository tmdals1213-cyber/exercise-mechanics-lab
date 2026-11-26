"""Bounded physical pattern search beside PPO; every proposal gets all4 dt.

This is numerical local refinement, not an additional PPO gradient update.
Body, contact mechanics, torque capacities and eligibility stay unchanged.
Any successful cached screen is repeated from rest without launch caching.
"""
import copy,hashlib,json,time,sys,shutil,traceback,os
from pathlib import Path
import numpy as np
import train_landing_stability as trainer
from effort_launch_parameters import decode
from jump_monitor_data import ROOT,read,atomic
PARENT=ROOT/'results/controlled_broad_jump/landing_stability_v3'
RUN=ROOT/'results/controlled_broad_jump/robust_finish_v1'
WEB=ROOT/'deliverables/jump_robust_finish'
MAINWEB=ROOT/'deliverables/jump_landing_live'
DTS=[.0005,.00025,.000125,.0000625]
PROGRESS_NAME='robust_search_status.json'
USE_LAUNCH_CACHE=True

def parent_state():
    for attempt in range(30):
        state=read(PARENT/'live.json')
        if state:return state
        time.sleep(.05)
    raise RuntimeError('Parent live state unavailable after retries')

def assessment(rows):
    stable=all(r['stable']and r['eligible']for r in rows)
    distances=[r.get('retained_distance_m')or 0. for r in rows]
    spread=(max(distances)-min(distances))/max(distances)if max(distances)>0 else 1.
    passed=bool(stable and spread<.05)
    return dict(passed=passed,goal_met=bool(passed and min(distances)>=3.),minimum_m=min(distances),relative_distance_spread=spread)

def proposals(seed,config,rng):
    # First target the missing ~11mm through physical flight/foot reach.
    specs=[{'prepare_hip':.01,'prepare_knee':-.01},{'prepare_hip':.02,'prepare_knee':-.02},{'prepare_hip':.01},{'prepare_knee':-.01},{'prepare_hip':-.01},{'prepare_knee':.01}]
    for key,step in [('prepare_transition_s',.002),('prepare_landing_s',.002),('flight_transition_s',.002),('push_s',.0005),('pitch_top',.003),('ankle_delay',.003)]:
        specs.extend([{key:step},{key:-step}])
    for changes in specs:
        p=copy.deepcopy(seed)
        for key,change in changes.items():p[key]+=change
        yield p,dict(method='physical_coordinate_probe',changes=changes)
    # Thereafter all58 policy coordinates remain available, using short
    # correlated steps around the best four-step-stable proposal.
    i=0
    while True:
        current=read(WEB/'best_parameters.json',seed);action=trainer.encode_parameters(config,current,1.);base,items=trainer.map_items(config,1.)
        step=rng.normal(0,.004 if i%3 else .008,58)
        if i%3==1:
            mask=np.array([x['key'].startswith(('flight_','prepare_','landing_'))for x in items]);step*=mask
        elif i%3==2:
            step*=np.array([x['kind']=='equivalent_effort'or x['key']in ['push_s','pitch_top','ankle_delay','hip_top','knee_top','arm_delay_s']for x in items])
        p=decode(base,items,np.clip(action+step,-1,1));yield p,dict(method='bounded58_coordinate_local_proposal',normalized_step=step.tolist());i+=1

def promote(report,progress):
    stop=PARENT/'stop.request';stop.write_text('Checkpoint: independently confirmed four-timestep >=3m robust refinement found.\n',encoding='utf-8');start=time.time()
    while time.time()-start<1200:
        parent=parent_state()
        if parent['status']in ['checkpointed','goal_found','budget_finished','failed']:break
        time.sleep(2)
    else:raise RuntimeError('No safe PPO checkpoint; qualified report retained, not promoted')
    if parent['status']=='goal_found':return
    source=min(report['cases'],key=lambda r:r['retained_distance_m']);report['discovery_provenance']=dict(algorithm='bounded robust physical pattern search after PPO',candidate_number=progress['candidates'],supplemental_native_evaluations=progress['native_evaluations'],PPO_training_trials_at_discovery=progress.get('PPO_trials_at_discovery',progress['PPO_trials_at_start']),seed_report=str(WEB/'seed_validation.json'),not_a_PPO_update=True)
    file=PARENT/f'validation_robust_finish_{int(time.time())}.json';atomic(file,report);atomic(PARENT/'verified_best.json',report)
    value={k:source.get(k)for k in ['id','distance_m','retained_distance_m','stability','stable','fell']};value.update(motion_id=source['id'],min_fine_first_m=min(r['distance_m']for r in report['cases']),min_fine_retained_m=report['minimum_m'],goal_met=True,timesteps_s=DTS,supplemental_search=report['discovery_provenance'])
    parent.update(status='goal_found',stage='completed',goal_achieved=True,continue_after_goal=False,verified=value,supplemental_search=report['discovery_provenance'],heartbeat=time.time(),stage_note='PPO 후 별도 국소 탐색에서 네 시간 간격 모두 3m 이상·안정 착지를 확인했습니다. 순수 PPO 시행과 추가 탐색 횟수를 구분합니다.')
    shutil.copyfile(Path(source['folder'])/'motion.json',MAINWEB/'verified_motion.json');atomic(PARENT/'live.json',parent);atomic(MAINWEB/'status.json',parent)
    if stop.exists():stop.unlink()

def main():
    resuming=RUN.exists();assert not resuming or os.environ.get('ROBUST_SEARCH_RESUME')=='1','Use explicit resume, never overwrite a search'
    RUN.mkdir(parents=True,exist_ok=True);WEB.mkdir(parents=True,exist_ok=True);config=copy.deepcopy(read(RUN/'definition.json')if resuming else read(PARENT/'definition.json'));atomic(RUN/'definition.json',config);initial=copy.deepcopy(read(WEB/'seed_validation.json')if resuming else read(PARENT/'verified_best.json'));assert initial['passed']and{r['dt_s']for r in initial['cases']}==set(DTS);atomic(WEB/'seed_validation.json',initial)
    seed=initial['cases'][0]['parameters']
    if not resuming:atomic(WEB/'best_parameters.json',seed)
    best=min(r['retained_distance_m']for r in initial['cases']);rng=np.random.default_rng(112603)
    trainer.OUT=RUN;trainer.WEB=WEB;trainer.TARGET_FIRST=config['goal_first_contact_m'];trainer.TARGET_RETAINED=config['goal_retained_contact_m'];trainer.VERSION=config['score_version'];exp=trainer.Experiment(config,4);exp.state.update(status='running',stage='robust_local_search',stage_note='Separate numerical refinement, not PPO.')
    progress=read(WEB/'progress.json')if resuming else dict(algorithm='bounded robust physical pattern search; not PPO',started_at=time.time(),candidates=0,native_evaluations=0,best_minimum_m=best,required_timesteps_s=DTS,torque_contact_and_body_unchanged=True,PPO_trials_at_start=parent_state()['training_episodes'],deadline=time.time()+7200)
    progress.update(status='running',heartbeat=time.time());progress.pop('error',None);best=progress['best_minimum_m'];seen=set();completed=False
    for file in RUN.glob('validation_proposal_*.json'):
        old=read(file)
        if old and old.get('cases'):seen.add(hashlib.sha256(json.dumps(old['cases'][0]['parameters'],sort_keys=True).encode()).hexdigest())
    def publish():
        progress['heartbeat']=time.time();atomic(WEB/'progress.json',progress);atomic(MAINWEB/PROGRESS_NAME,progress)
    publish()
    try:
        for p,description in proposals(seed,config,rng):
            if time.time()>progress['deadline']or progress['candidates']>=120:break
            if parent_state()['status']=='goal_found':progress['status']='parent_goal_found';break
            try:trainer.encode_parameters(config,p,1.)
            except AssertionError:continue
            assert p['joint_torque_limits_Nm']==seed['joint_torque_limits_Nm']
            key=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
            if key in seen:continue
            seen.add(key);progress['candidates']+=1;progress['proposal']=description;progress['stage']='four_step_screen';publish()
            signature={k:v for k,v in p.items()if not k.startswith(('flight_','prepare_','landing_'))and k!='seconds'};cache_id=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
            rows=exp.batch([dict(cmd='rollout',base=p,dt=dt,kind='robust_local_search',cache_launch=USE_LAUNCH_CACHE,cache_id=cache_id,context=1.)for dt in DTS]);progress['native_evaluations']+=4;review=assessment(rows);report=dict(**review,cases=rows,completion_rule=config['completion_rule'],proposal=description,cached_screen=USE_LAUNCH_CACHE)
            progress['latest_cases']=[dict(dt_s=r['dt_s'],retained_m=r['retained_distance_m'],stable=r['stable'],eligible=r['eligible'])for r in rows];progress['latest_minimum_m']=review['minimum_m'];progress['latest_passed']=review['passed'];atomic(RUN/f'validation_proposal_{progress["candidates"]:04d}.json',report)
            if review['passed']and review['minimum_m']>best:
                best=review['minimum_m'];atomic(WEB/'best_parameters.json',p);atomic(WEB/'best_validation.json',report);progress['best_minimum_m']=best
            with(WEB/'proposals.jsonl').open('a',encoding='utf-8')as log:log.write(json.dumps(dict(candidate=progress['candidates'],at=time.time(),**review,proposal=description))+'\n')
            print('ROBUST_PROPOSAL',progress['candidates'],review['minimum_m'],review['passed'],review['goal_met'],flush=True);publish()
            if review['goal_met']:
                progress['stage']='uncached_four_step_confirmation';publish();cold=exp.batch([dict(cmd='rollout',base=p,dt=dt,kind='robust_confirmation',context=1.)for dt in DTS]);progress['native_evaluations']+=4;check=assessment(cold);report=dict(**check,cases=cold,completion_rule=config['completion_rule'],proposal=description,cached_screen=False);atomic(RUN/f'validation_cold_{progress["candidates"]:04d}.json',report)
                if check['goal_met']:
                    progress['stage']='checkpoint_parent_for_finalization';progress['PPO_trials_at_discovery']=parent_state()['training_episodes'];publish();promote(report,progress);completed=True;progress.update(status='goal_found',stage='qualified',best_minimum_m=check['minimum_m']);break
                progress['cold_confirmation_failed']=check;publish()
        if not completed and progress['status']=='running':progress['status']='local_budget_finished'
    except Exception as exc:progress.update(status='failed',error=repr(exc));traceback.print_exc();raise
    finally:exp.state.update(status='goal_found'if completed else'checkpointed',stage='completed');exp.close();publish()
    print('ROBUST_FINISH_EXIT',progress['status'],progress['best_minimum_m'],flush=True)

if __name__=='__main__':main()
