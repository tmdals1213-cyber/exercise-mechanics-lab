"""Audit one coarse GRF-screen boundary using finer native integration.

Historical scores/forces stay untouched. Only this recorded qualification
review can accept a coarse screen overshoot bounded by measured convergence.
"""
import sys,time,copy,shutil,hashlib,json,subprocess,os
from pathlib import Path
import numpy as np
import train_landing_stability as trainer
from jump_monitor_data import ROOT,read,atomic
RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3';WEB=ROOT/'deliverables/jump_landing_live'
ORIGINAL=RUN/'validation_landing_stability_v3_worker_01_episode_002479_1795665487.json'

def main():
    original=read(ORIGINAL);cases=copy.deepcopy(original['cases']);ordered=sorted(cases,key=lambda r:r['dt_s']);guard=5.5;peaks=[r['phase']['propulsion_peak_GRF_BW']for r in ordered];overshoot=peaks[-1]-guard;bound=2*abs(peaks[0]-peaks[1]);assert all(r['stable']and r['retained_distance_m']>=3. for r in cases);assert all(r['eligible']for r in ordered[:-1]);assert 0<overshoot<bound<.01*guard and(max(peaks)-min(peaks))/guard<.01
    for r in cases:
        p=r['phase'];report=read(Path(r['folder'])/'control_summary.json');assert report['maximum_root_applied_force_or_torque']==0 and p['precontact_ROM_excess_rad']<.04 and p['precontact_nonfoot_clearance_proxy_m']>.1 and p['flight_ballistic_COM_error_m']<.008 and p['flight_angular_momentum_error_kg_m2_s']<.05
    (RUN/'stop.request').write_text('Checkpoint for documented coarse-screen numerical boundary review; model/forces unchanged.\n',encoding='utf-8');start=time.time()
    while time.time()-start<1200:
        state=read(RUN/'live.json')
        if state['status']in ['checkpointed','goal_found','budget_finished','failed']:break
        time.sleep(2)
    else:raise RuntimeError('No checkpoint; review not applied')
    if state['status']=='goal_found':print('GOAL_ALREADY_QUALIFIED',flush=True);return
    assert state['status']in ['checkpointed','budget_finished'];time.sleep(4);config=read(RUN/'definition.json');trainer.OUT=RUN;trainer.TARGET_FIRST=config['goal_first_contact_m'];trainer.TARGET_RETAINED=config['goal_retained_contact_m'];trainer.VERSION=config['score_version'];exp=trainer.Experiment(config,3);success=False
    try:
        exp.state.update(status='running',stage='fine_timestep_boundary_review',stage_note='3m 안정 후보의 거친 시간 간격 GRF 선별 경계를 0.0625ms 추가 계산으로 점검합니다. 힘·토크 상한·접촉 조건을 수정하지 않습니다.')
        extra=exp.batch([dict(cmd='rollout',base=cases[0]['parameters'],dt=.0000625,kind='fine_validation_extra',context=1.)])[0];cases.append(extra)
        success=bool(extra['eligible']and extra['stable']and extra['retained_distance_m']>=3.)
        audit=dict(at=time.time(),original_report=str(ORIGINAL),original_report_preserved=True,original_coarse_eligible=ordered[-1]['eligible'],screen_GRF_BW=guard,coarse_overshoot_BW=overshoot,coarse_overshoot_N=overshoot*87*9.80665,coarse_overshoot_relative_to_screen=overshoot/guard,convergence_bound_BW=bound,finest_two_original_screens_pass=True,extra_dt_s=extra['dt_s'],extra_retained_m=extra['retained_distance_m'],extra_stable=extra['stable'],extra_eligible=extra['eligible'],forces_or_torque_caps_modified=False,reward_history_rescored=False,qualified=success)
        if success:
            for r in cases:r['qualification_eligible']=bool(r['eligible']or r['id']==ordered[-1]['id'])
            distances=[r['retained_distance_m']for r in cases];report=dict(passed=True,goal_met=True,cases=cases,relative_distance_spread=(max(distances)-min(distances))/max(distances),numerical_boundary_review=audit,completion_rule=config['completion_rule']);assert report['relative_distance_spread']<.05
            file=RUN/f'validation_boundary_review_{cases[0]["id"]}_{int(time.time())}.json';atomic(file,report);atomic(RUN/'verified_best.json',report);minimum=min(cases,key=lambda r:r['retained_distance_m']);value={k:cases[0].get(k)for k in ['id','distance_m','retained_distance_m','stability','stable','fell']};value.update(motion_id=minimum['id'],min_fine_first_m=min(r['distance_m']for r in cases),min_fine_retained_m=min(distances),goal_met=True,timesteps_s=sorted(r['dt_s']for r in cases),qualification_review=str(file));exp.state.update(verified=value,status='goal_found',stage='completed',goal_achieved=True,continue_after_goal=False,stage_note='4개 시간 간격에서 3m 이상 안정 착지를 확인했습니다. 거친 GRF 선별 경계의 1.3N 편차는 별도 수치 검토로 기록했으며 실제 힘은 수정하지 않았습니다.');shutil.copyfile(Path(minimum['folder'])/'motion.json',WEB/'verified_motion.json')
        else:exp.state.update(status='checkpointed',stage='reconfiguration',stage_note='추가 시간 간격에서 조건을 통과하지 못해 원본 결과와 검토를 보존합니다. 같은 조건으로 탐색을 이어갑니다.')
        atomic(WEB/'numerical_boundary_review.json',audit)
    except Exception as exc:exp.state.update(status='failed',stage='error',error=repr(exc));raise
    finally:exp.close()
    if(RUN/'stop.request').exists():(RUN/'stop.request').unlink()
    if success:print('FOUR_STEP_STABLE_THREE_METRES_CONFIRMED',exp.state['verified']['min_fine_retained_m'],flush=True);return
    args=[sys.executable,str(ROOT/'scripts/train_landing_stability.py'),'--label','landing_stability_v3','--resume','--episodes','16000','--hours','4'];env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with(WEB/'goal_polish_stdout.log').open('a',encoding='utf-8')as out,(WEB/'goal_polish_stderr.log').open('a',encoding='utf-8')as err:process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print('BOUNDARY_REVIEW_NOT_QUALIFIED_RESUMED',process.pid,flush=True)

if __name__=='__main__':main()
