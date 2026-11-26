"""Gracefully checkpoint the running policy and apply the user's finish rule."""
import sys,time,subprocess,os
from jump_monitor_data import ROOT,read,atomic
RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3';WEB=ROOT/'deliverables/jump_landing_live'

def main():
    initial=read(RUN/'live.json');assert initial['status']in ['running','budget_finished','checkpointed','goal_found'];(RUN/'stop.request').write_text('Checkpoint to apply user rule: finish on three-timestep stable retained distance >=3.00m.\n',encoding='utf-8')
    start=time.time()
    while time.time()-start<1200:
        state=read(RUN/'live.json')
        if state['status']in ['checkpointed','budget_finished','goal_found','failed']:break
        time.sleep(2)
    else:raise RuntimeError('No completed checkpoint; configuration unchanged')
    assert state['status']!='failed',state.get('error');time.sleep(4)
    archive=RUN/f'before_finish_rule_{state["training_episodes"]:06d}';archive.mkdir(exist_ok=False)
    for name in ['definition.json','live.json','verified_best.json','policy_latest.zip']:(archive/name).write_bytes((RUN/name).read_bytes())
    config=read(RUN/'definition.json');config['completion_rule']=dict(min_retained_m=3.,all_three_timesteps_stable=True,stop_on_success=True,first_contact_target_required=False,user_requested=True);atomic(RUN/'definition.json',config);(RUN/'stop.request').unlink()
    args=[sys.executable,str(ROOT/'scripts/train_landing_stability.py'),'--label','landing_stability_v3','--resume','--episodes','16000','--hours','4'];env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with(WEB/'finish_rule_stdout.log').open('a',encoding='utf-8')as out,(WEB/'finish_rule_stderr.log').open('a',encoding='utf-8')as err:process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    atomic(WEB/'finish_rule_status.json',dict(status='launched',at=time.time(),pid=process.pid,archive=str(archive),completion_rule=config['completion_rule'],physical_caps_and_bounds_unchanged=True,reward_target_unchanged=True,previous_deadline_preserved=True,training_episodes_at_checkpoint=state['training_episodes'],policy_updates_at_checkpoint=state['policy_updates']))
    print('FINISH_AT_STABLE_THREE_METRES_SET',process.pid,state['training_episodes'],flush=True)

if __name__=='__main__':main()
