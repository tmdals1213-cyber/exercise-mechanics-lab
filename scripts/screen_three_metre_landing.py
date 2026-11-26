"""Cheap failing-timestep screen, then strict cold four-timestep confirmation."""
import copy,json,time,traceback
import train_landing_stability as trainer
import search_robust_jump_finish as task
from jump_monitor_data import ROOT,read,atomic

def main():
    task.RUN=ROOT/'results/controlled_broad_jump/postcontact_screen_v1';task.WEB=ROOT/'deliverables/jump_postcontact_screen';task.PROGRESS_NAME='postcontact_screen_status.json';assert not task.RUN.exists();task.RUN.mkdir(parents=True);task.WEB.mkdir(parents=True,exist_ok=True)
    config=read(task.PARENT/'definition.json');source=read(task.PARENT/'validation_landing_stability_v3_worker_01_episode_002507_1795672039.json');base=copy.deepcopy(source['cases'][0]['parameters']);stable=read(task.PARENT/'verified_best.json')['cases'][0]['parameters']
    for key in ['landing_hip','landing_knee','landing_ankle','landing_trunk','landing_shoulder','landing_transition_s']:base[key]=stable[key]
    atomic(task.RUN/'definition.json',config);atomic(task.WEB/'seed_validation.json',source);trainer.OUT=task.RUN;trainer.WEB=task.WEB;trainer.TARGET_FIRST=config['goal_first_contact_m'];trainer.TARGET_RETAINED=config['goal_retained_contact_m'];trainer.VERSION=config['score_version'];exp=trainer.Experiment(config,4);exp.state.update(status='running',stage='failing_timestep_screen')
    changes=[dict(landing_support_damping=70.,landing_ankle_support_damping=35.),dict(landing_response_trust_Nm=150.),dict(landing_support_damping=85.,landing_ankle_support_damping=40.),dict(landing_ankle=base['landing_ankle']+.005)]
    for key,step in [('landing_ankle',.005),('landing_trunk',.01),('landing_hip',.01),('landing_knee',.01),('landing_transition_s',.01)]:
        for delta in [-step,step]:changes.append({key:base[key]+delta})
    for gain in [60.,65.,75.,90.,110.]:changes.append(dict(landing_support_damping=gain,landing_ankle_support_damping=30.,landing_response_trust_Nm=200.))
    progress=dict(status='running',stage='coarse_failure_screen',algorithm='postcontact parameter search, not PPO;0.5ms reject screen then cold all4',candidates=0,native_evaluations=0,best_minimum_m=2.9888532922990696,required_timesteps_s=task.DTS,PPO_trials_at_start=task.parent_state()['training_episodes'],torque_contact_and_body_unchanged=True);completed=False
    def publish():progress['heartbeat']=time.time();atomic(task.WEB/'progress.json',progress);atomic(task.MAINWEB/task.PROGRESS_NAME,progress)
    publish()
    try:
        for start in range(0,len(changes),4):
            if task.parent_state()['status']=='goal_found':break
            group=[]
            for change in changes[start:start+4]:
                p=copy.deepcopy(base);p.update(change);trainer.encode_parameters(config,p,1.);assert p['joint_torque_limits_Nm']==base['joint_torque_limits_Nm'];group.append(p)
            progress.update(candidates=start+len(group),stage='coarse_failure_screen');publish();coarse=exp.batch([dict(cmd='rollout',base=p,dt=.0005,kind='postcontact_coarse_screen',context=1.)for p in group]);progress['native_evaluations']+=len(coarse)
            for i,(p,row)in enumerate(zip(group,coarse)):
                print('POSTCONTACT_SCREEN',start+i+1,row['retained_distance_m'],row['stable'],row['eligible'],flush=True)
                if not(row['stable']and row['eligible']and row['retained_distance_m']>=3.):continue
                progress['stage']='cold_four_timestep_confirmation';publish();rows=exp.batch([dict(cmd='rollout',base=p,dt=dt,kind='robust_confirmation',context=1.)for dt in task.DTS]);progress['native_evaluations']+=4;check=task.assessment(rows);error=max(abs(r['distance_m']-next(o['distance_m']for o in source['cases']if o['dt_s']==r['dt_s']))for r in rows);assert error<1e-10,('Unexpected precontact change',error)
                report=dict(**check,cases=rows,completion_rule=config['completion_rule'],proposal=dict(method='postcontact_pose_and_feedback_refinement',changes=changes[start+i],winning_screen=start+i+1,launch_and_airborne_controls_unchanged=True),cached_screen=False,first_contact_max_change_m=error);atomic(task.RUN/f'validation_cold_{start+i+1:04d}.json',report)
                print('POSTCONTACT_ALL4',start+i+1,check['minimum_m'],check['passed'],check['goal_met'],flush=True)
                if check['passed']:progress['best_minimum_m']=max(progress['best_minimum_m'],check['minimum_m'])
                if check['goal_met']:
                    progress.update(stage='checkpoint_parent_for_finalization',PPO_trials_at_discovery=task.parent_state()['training_episodes']);publish();task.promote(report,progress);completed=True;progress.update(status='goal_found',stage='qualified',best_minimum_m=check['minimum_m']);break
            publish()
            if completed:break
        if not completed:progress['status']='screen_finished'
    except Exception as exc:progress.update(status='failed',error=repr(exc));traceback.print_exc();raise
    finally:exp.state.update(status='goal_found'if completed else'checkpointed',stage='completed');exp.close();publish()

if __name__=='__main__':main()
