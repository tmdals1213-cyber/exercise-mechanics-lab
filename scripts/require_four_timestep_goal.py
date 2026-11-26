"""Apply the user's strict four-integrator stopping rule at a safe checkpoint."""
import os,sys,time,subprocess,shutil
from jump_monitor_data import ROOT,read,atomic
RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3'
WEB=ROOT/'deliverables/jump_landing_live'

def main():
    (RUN/'stop.request').write_text('Checkpoint: user requires >=3.000m and stable landing at all four tested time steps.\n',encoding='utf-8')
    start=time.time()
    while time.time()-start<1200:
        state=read(RUN/'live.json')
        if state['status']in ['checkpointed','goal_found','budget_finished','failed']:break
        time.sleep(2)
    else:raise RuntimeError('No safe checkpoint; no configuration changed')
    assert state['status']!='failed',state.get('error')
    time.sleep(4);archive=RUN/f'before_four_timestep_rule_{state["training_episodes"]:06d}';archive.mkdir(exist_ok=False)
    for name in ['definition.json','live.json','verified_best.json','policy_latest.zip']:shutil.copyfile(RUN/name,archive/name)
    config=read(RUN/'definition.json');config['completion_rule']=dict(min_retained_m=3.,required_timesteps_s=[.0005,.00025,.000125,.0000625],all_required_timesteps_stable=True,stop_on_success=True,first_contact_target_required=False,user_requested=True,numerical_eligibility_screen_unchanged=True)
    atomic(RUN/'definition.json',config)
    # Historical three-step results remain in the archive; they are not a
    # qualified best under the newly requested four-step protocol.
    state['previous_three_step_verified']=state.get('verified');state['verified']=None;state['tested_candidates']={};state.update(status='checkpointed',completion_rule=config['completion_rule']);atomic(RUN/'live.json',state)
    (RUN/'stop.request').unlink()
    args=[sys.executable,str(ROOT/'scripts/train_landing_stability.py'),'--label','landing_stability_v3','--resume','--episodes','40000','--hours','12','--renew-budget']
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with(WEB/'four_step_stdout.log').open('a',encoding='utf-8')as out,(WEB/'four_step_stderr.log').open('a',encoding='utf-8')as err:p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    atomic(WEB/'four_step_rule.json',dict(at=time.time(),pid=p.pid,archive=str(archive),completion_rule=config['completion_rule'],extended_hours=12,policy_weights_unchanged=True,torque_limits_unchanged=True,contact_and_eligibility_unchanged=True))
    print('FOUR_STEP_STRICT_SEARCH_STARTED',p.pid,flush=True)

if __name__=='__main__':main()
