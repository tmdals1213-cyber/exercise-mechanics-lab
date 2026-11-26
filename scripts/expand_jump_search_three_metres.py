"""Checkpoint-preserving authorized expansion; never change strength or physics."""
from pathlib import Path
import sys,os,time,copy,hashlib,json,subprocess
import numpy as np
from jump_monitor_data import ROOT,read,atomic
from train_landing_stability import map_items,encode_parameters
RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3';WEB=ROOT/'deliverables/jump_landing_live'

def main():
    start=time.time();state=read(RUN/'live.json');pid=state.get('coordinator_pid');atomic(WEB/'range_expansion_status.json',dict(status='waiting_for_checkpoint',requested_at=start,goal_retained_m=3.,fixed_torque_capacities=True));(RUN/'stop.request').write_text('User requested expanded posture/timing/drive search toward3m stable landing.\n',encoding='utf-8')
    while time.time()-start<1200:
        s=read(RUN/'live.json')
        if s['status']in['checkpointed','budget_finished','failed']and s['stage']not in['ppo_training','fine_timestep_validation']:break
        time.sleep(3)
    else:raise RuntimeError('Checkpoint did not finish; no live configuration changed')
    time.sleep(4);s=read(RUN/'live.json');episode=s['training_episodes'];config=read(RUN/'definition.json');old=copy.deepcopy(config);archive=RUN/f'before_three_metre_expansion_{episode:06d}';archive.mkdir(exist_ok=True)
    for name in ['definition.json','live.json','verified_best.json','policy_latest.zip']:(archive/name).write_bytes((RUN/name).read_bytes())
    v=read(RUN/'verified_best.json');seed=next(r for r in v['cases']if abs(r['dt_s']-.00025)<1e-10);caps=copy.deepcopy(seed['parameters']['joint_torque_limits_Nm'])
    for item in config['actions']:
        if item['kind']=='equivalent_effort':item.update(minimum=-.55,maximum=.55,half_span=.55)
        else:item['half_span']*=1.40 if item['key']not in['push_s','prepare_landing_s','landing_transition_s']else 1.65
    config.update(goal_first_contact_m=3.10,goal_retained_contact_m=3.,score_version='additive_recovery_v2.2_expanded_three_metre_search',search_range_revision='three_metre_v1',previous_parameter_bounds_preserved_in=str(archive/'definition.json'))
    _,before=map_items(old,1.);base,after=map_items(config,1.);assert base['joint_torque_limits_Nm']==caps
    changes=[]
    for a,b in zip(before,after):
        assert b['minimum']<=a['minimum']+1e-10 and b['maximum']>=a['maximum']-1e-10
        changes.append(dict(key=a['key'],unit=a['unit'],previous=[a['minimum'],a['maximum']],expanded=[b['minimum'],b['maximum']]))
    action=encode_parameters(config,seed['parameters']);assert action.shape==(58,)
    atomic(archive/'expanded_definition.json',config);atomic(archive/'seed.json',seed);atomic(archive/'range_audit.json',dict(changes=changes,coordinates=58,joint_ROM_unchanged=True,torque_limits_Nm=caps,torque_sha256=hashlib.sha256(json.dumps(caps,sort_keys=True).encode()).hexdigest(),old_samples_preserved=True,goal_retained_m=3.,goal_first_m=3.10))
    atomic(RUN/'definition.json',config);(RUN/'stop.request').unlink();atomic(WEB/'range_expansion_status.json',dict(status='launching',episode=episode,archive=str(archive),goal_retained_m=3.,fixed_torque_capacities=True))
    args=[sys.executable,str(ROOT/'scripts/train_landing_stability.py'),'--label','landing_stability_v3','--resume','--continue-after-goal','--episodes','16000','--hours','8','--reseed-parameters',str(archive/'seed.json'),'--reseed-source-algorithm','Verified candidate remapped into expanded posture/timing/drive bounds; not a PPO update']
    stdout=(WEB/'three_metre_stdout.log').open('a',encoding='utf-8');stderr=(WEB/'three_metre_stderr.log').open('a',encoding='utf-8');proc=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));atomic(WEB/'range_expansion_status.json',dict(status='running',launched_at=time.time(),pid=proc.pid,episode_at_extension=episode,goal_first_m=3.10,goal_retained_m=3.,archive=str(archive),fixed_torque_capacities=True,remaining_deadline=s.get('deadline'),bounds=changes));print('THREE_METRE_SEARCH_LAUNCHED',proc.pid,episode,flush=True)

if __name__=='__main__':main()

