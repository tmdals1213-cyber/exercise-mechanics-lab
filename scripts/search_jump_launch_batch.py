"""Four candidate launches x four integrators in parallel, unchanged mechanics."""
import copy,json,time,traceback
import numpy as np
import search_robust_jump_finish as task
import train_landing_stability as trainer
from jump_monitor_data import ROOT,read,atomic
LABEL='robust_launch_batch_v1'
WEBNAME='jump_robust_launch_batch'
PROGRESS_NAME='robust_launch_status.json'

def proposals(seed):
    for delta in [-.00025,.00025,-.0005,.0005,-.000125,.000125,-.001,.001]:
        p=copy.deepcopy(seed);p['push_s']+=delta;yield p,dict(key='push_s',delta_s=delta)
    for stem in ['hip_flexion','knee_angle','ankle_angle']:
        for delta in [.001,-.001,.002,-.002]:
            p=copy.deepcopy(seed);p['propulsion_effort_knots_fraction'][stem]=[v+delta for v in p['propulsion_effort_knots_fraction'][stem]];yield p,dict(key=stem+'_effort_knots',delta_fraction=delta)
    for key,step in [('ankle_delay',.001),('pitch_top',.001),('shoulder_top',.002),('hip_extension_duration_fraction',.001),('knee_delay',.001)]:
        for delta in [step,-step]:
            p=copy.deepcopy(seed);p[key]+=delta;yield p,dict(key=key,delta=delta)

def main():
    task.RUN=ROOT/'results/controlled_broad_jump'/LABEL;task.WEB=ROOT/'deliverables'/WEBNAME;task.PROGRESS_NAME=PROGRESS_NAME
    assert not task.RUN.exists();task.RUN.mkdir(parents=True);task.WEB.mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(read(task.PARENT/'definition.json'));initial=read(task.PARENT/'verified_best.json');assert initial['passed']and{r['dt_s']for r in initial['cases']}==set(task.DTS);seed=initial['cases'][0]['parameters'];atomic(task.RUN/'definition.json',config);atomic(task.WEB/'seed_validation.json',initial)
    trainer.OUT=task.RUN;trainer.WEB=task.WEB;trainer.TARGET_FIRST=config['goal_first_contact_m'];trainer.TARGET_RETAINED=config['goal_retained_contact_m'];trainer.VERSION=config['score_version'];exp=trainer.Experiment(config,16);exp.state.update(status='running',stage='parallel_robust_launch_search')
    progress=dict(status='running',stage='parallel_four_candidate_four_step_screen',algorithm='bounded physical launch pattern search, not PPO',candidates=0,native_evaluations=0,best_minimum_m=min(r['retained_distance_m']for r in initial['cases']),PPO_trials_at_start=read(task.PARENT/'live.json')['training_episodes'],required_timesteps_s=task.DTS,torque_contact_and_body_unchanged=True,deadline=time.time()+5400)
    def publish():progress['heartbeat']=time.time();atomic(task.WEB/'progress.json',progress);atomic(task.MAINWEB/task.PROGRESS_NAME,progress)
    candidates=list(proposals(seed));completed=False;publish()
    try:
        for start in range(0,len(candidates),4):
            if time.time()>progress['deadline']or read(task.PARENT/'live.json')['status']=='goal_found':break
            group=candidates[start:start+4];commands=[]
            for p,description in group:
                trainer.encode_parameters(config,p,1.);assert p['joint_torque_limits_Nm']==seed['joint_torque_limits_Nm']
                commands.extend(dict(cmd='rollout',base=p,dt=dt,kind='robust_local_search',context=1.)for dt in task.DTS)
            progress.update(candidates=start+len(group),active_proposals=[d for p,d in group]);publish();rows=exp.batch(commands);progress['native_evaluations']+=len(rows)
            for i,(p,description)in enumerate(group):
                cases=rows[i*4:(i+1)*4];check=task.assessment(cases);report=dict(**check,cases=cases,completion_rule=config['completion_rule'],proposal=description,cached_screen=False);atomic(task.RUN/f'validation_proposal_{start+i+1:04d}.json',report)
                if check['passed']and check['minimum_m']>progress['best_minimum_m']:
                    progress['best_minimum_m']=check['minimum_m'];atomic(task.WEB/'best_validation.json',report);atomic(task.WEB/'best_parameters.json',p)
                with(task.WEB/'proposals.jsonl').open('a',encoding='utf-8')as log:log.write(json.dumps(dict(candidate=start+i+1,at=time.time(),**check,proposal=description))+'\n')
                print('ROBUST_LAUNCH_PROPOSAL',start+i+1,check['minimum_m'],check['passed'],check['goal_met'],flush=True);publish()
                if check['goal_met']:
                    progress['stage']='uncached_four_step_confirmation';publish();cold=exp.batch([dict(cmd='rollout',base=p,dt=dt,kind='robust_confirmation',context=1.)for dt in task.DTS]);progress['native_evaluations']+=4;confirmed=task.assessment(cold);report=dict(**confirmed,cases=cold,completion_rule=config['completion_rule'],proposal=description,cached_screen=False);atomic(task.RUN/f'validation_cold_{start+i+1:04d}.json',report)
                    if confirmed['goal_met']:
                        progress.update(stage='checkpoint_parent_for_finalization',PPO_trials_at_discovery=read(task.PARENT/'live.json')['training_episodes'],winning_proposal=start+i+1);publish();task.promote(report,progress);completed=True;progress.update(status='goal_found',stage='qualified',best_minimum_m=confirmed['minimum_m']);break
            if completed:break
        if not completed:progress['status']='local_batch_finished'
    except Exception as exc:progress.update(status='failed',error=repr(exc));traceback.print_exc();raise
    finally:exp.state.update(status='goal_found'if completed else'checkpointed',stage='completed');exp.close();publish()
    print('ROBUST_LAUNCH_EXIT',progress['status'],progress['best_minimum_m'],flush=True)

if __name__=='__main__':main()
