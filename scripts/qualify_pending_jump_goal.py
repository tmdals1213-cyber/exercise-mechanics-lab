"""Checkpoint learning and qualify up to four existing target-distance samples."""
import sys,time,subprocess,os
from pathlib import Path
import train_landing_stability as trainer
from jump_monitor_data import ROOT,read,atomic
RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3';WEB=ROOT/'deliverables/jump_landing_live'

def main():
    (RUN/'stop.request').write_text('Checkpoint for immediate qualification of existing >=3m PPO candidates.\n',encoding='utf-8');start=time.time()
    while time.time()-start<1200:
        state=read(RUN/'live.json')
        if state['status']in ['checkpointed','goal_found','budget_finished','failed']:break
        time.sleep(2)
    else:raise RuntimeError('No checkpoint available')
    if state['status']=='goal_found':print('GOAL_ALREADY_QUALIFIED',flush=True);return
    assert state['status']in ['checkpointed','budget_finished'];time.sleep(4);config=read(RUN/'definition.json');trainer.OUT=RUN;trainer.TARGET_FIRST=config['goal_first_contact_m'];trainer.TARGET_RETAINED=config['goal_retained_contact_m'];trainer.VERSION=config['score_version'];exp=trainer.Experiment(config,3);completed=False;reports=[]
    try:
        exp.state.update(status='running',stage='goal_candidate_qualification',stage_note='3m 이상 기존 PPO 후보를 우선 정밀 확인합니다. 추가 학습과 별도로 검증 횟수를 기록합니다.')
        candidates=[c for c in exp.state['stable_candidates']if c['retained_distance_m']>=3. and c['key']not in exp.state['tested_candidates']]
        # A modest target overshoot can be more robust than the farthest sample.
        candidates.sort(key=lambda c:(c['dt_s']>=.0005,abs(c['retained_distance_m']-3.025)))
        for candidate in candidates[:4]:
            report=exp.validate(read(Path(candidate['folder'])/'score.json'));reports.append(dict(id=candidate['id'],passed=report['passed'],goal_met=report['goal_met'],cases=[dict(dt_s=r['dt_s'],stable=r['stable'],retained_m=r['retained_distance_m'])for r in report['cases']]))
            if report['goal_met']:
                completed=True;exp.state.update(status='goal_found',stage='completed',goal_achieved=True,continue_after_goal=False,stage_note='3개 시간 간격에서 안정 유지 3m를 통과했습니다. 학습을 종료하고 최종 영상과 예제를 만듭니다.');break
        if not completed:exp.state.update(status='checkpointed',stage='reconfiguration',stage_note='현재 목표 후보를 정밀 확인했습니다. 미통과 사례도 보존하고 같은 체크포인트에서 학습을 이어갑니다.')
    except Exception as exc:exp.state.update(status='failed',stage='error',error=repr(exc));raise
    finally:exp.close()
    atomic(WEB/'target_qualification_audit.json',dict(at=time.time(),goal_qualified=completed,reports=reports,extra_PPO_updates=0,physical_conditions_unchanged=True))
    if completed:
        if(RUN/'stop.request').exists():(RUN/'stop.request').unlink()
        print('STABLE_THREE_METRE_GOAL_CONFIRMED',flush=True);return
    (RUN/'stop.request').unlink();args=[sys.executable,str(ROOT/'scripts/train_landing_stability.py'),'--label','landing_stability_v3','--resume','--episodes','16000','--hours','4'];env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with(WEB/'goal_polish_stdout.log').open('a',encoding='utf-8')as out,(WEB/'goal_polish_stderr.log').open('a',encoding='utf-8')as err:process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print('QUALIFICATION_FINISHED_RESUMED',process.pid,flush=True)

if __name__=='__main__':main()
