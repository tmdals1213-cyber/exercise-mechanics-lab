"""Inspect the finished search, qualify the late winner, then resume locally."""
import sys,os,time,copy,subprocess,hashlib,json
from pathlib import Path
import numpy as np
from jump_monitor_data import ROOT,read,atomic
import train_landing_stability as trainer
RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3';WEB=ROOT/'deliverables/jump_landing_live'

def main():
    state=read(RUN/'live.json');assert state['status']in ['budget_finished','checkpointed'],state['status'];assert not(RUN/'stop.request').exists()
    archive=RUN/f'before_local_resume_{state["training_episodes"]:06d}';archive.mkdir(exist_ok=False)
    for name in ['definition.json','live.json','verified_best.json','best_stable.json','policy_latest.zip']:(archive/name).write_bytes((RUN/name).read_bytes())
    config=read(RUN/'definition.json');files=sorted(RUN.glob('rollout_batch_update_*.npz'))[-20:];actions=np.concatenate([np.load(f)['actions'].reshape(-1,58)for f in files]);recent=[r for r in state['history']if r['kind']=='ppo_train'][-480:]
    audit=dict(at=time.time(),previous_status=state['status'],previous_deadline=state['deadline'],previous_training_episodes=state['training_episodes'],previous_policy_updates=state['policy_updates'],best_single_dt_retained_m=state['best_stable']['retained_distance_m'],verified_min_retained_m=state['verified']['min_fine_retained_m'],last20_updates_action_boundary_fraction=float(np.mean(abs(actions)>=.8)),last20_updates_max_abs_action=float(abs(actions).max()),recent_stable_trials=sum(r['stable']and r['eligible']for r in recent),recent_trials=len(recent),decision='Keep existing absolute bounds, body, contacts, controller and torque caps; qualify late stable winner, recenter on best qualified parameters and resume local PPO.',new_std_groups=dict(posture_timing=.06,effort=.09,postcontact=.045),renewed_hours=4.,archive=str(archive),historical_data_preserved=True)
    atomic(archive/'audit.json',audit);atomic(WEB/'local_resume_audit.json',audit)
    trainer.OUT=RUN;trainer.TARGET_FIRST=config['goal_first_contact_m'];trainer.TARGET_RETAINED=config['goal_retained_contact_m'];trainer.VERSION=config['score_version'];exp=trainer.Experiment(config,3)
    try:
        exp.state.update(status='running',stage='late_candidate_validation',stage_note='종료 직전 2.99m 안정 후보를 정밀 시간 간격에서 확인합니다. 기존 변수 범위·토크 상한을 유지합니다.')
        row=read(RUN/'best_stable.json');key=exp.candidate_key(row['parameters'])
        if key not in exp.state['tested_candidates']:report=exp.validate(row)
        else:report=dict(already_tested=True,result=exp.state['tested_candidates'][key])
        atomic(archive/'late_candidate_qualification.json',report)
        exp.state.update(status='checkpointed',stage='reconfiguration',stage_note='정밀 확인 결과를 저장했습니다. 확인된 안정 후보 주위에서 추가 학습을 재개합니다.')
    except Exception as exc:
        exp.state.update(status='failed',stage='error',error=repr(exc));raise
    finally:exp.close()
    verified=read(RUN/'verified_best.json');seed=next(r for r in verified['cases']if abs(r['dt_s']-.00025)<1e-10);atomic(archive/'seed.json',seed)
    config['recenter_std_groups']=audit['new_std_groups'];atomic(RUN/'definition.json',config)
    assert seed['parameters']['joint_torque_limits_Nm']==config['stable_anchor']['joint_torque_limits_Nm']
    args=[sys.executable,str(ROOT/'scripts/train_landing_stability.py'),'--label','landing_stability_v3','--resume','--renew-budget','--continue-after-goal','--episodes','16000','--hours','4','--reseed-parameters',str(archive/'seed.json'),'--reseed-source-algorithm','Latest three-timestep-qualified stable candidate; same absolute bounds, local sampling restart, not a PPO update']
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');stdout=(WEB/'local_three_metre_stdout.log').open('a',encoding='utf-8');stderr=(WEB/'local_three_metre_stderr.log').open('a',encoding='utf-8');process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));audit.update(status='launched',pid=process.pid,launched_at=time.time(),seed_verified_min_retained_m=min(r['retained_distance_m']for r in verified['cases']),late_candidate_passed=report.get('passed'),command=args,torque_caps_sha256=hashlib.sha256(json.dumps(seed['parameters']['joint_torque_limits_Nm'],sort_keys=True).encode()).hexdigest());atomic(archive/'audit.json',audit);atomic(WEB/'local_resume_audit.json',audit)
    print('LOCAL_THREE_METRE_RESUMED',process.pid,audit['seed_verified_min_retained_m'],flush=True)

if __name__=='__main__':main()
