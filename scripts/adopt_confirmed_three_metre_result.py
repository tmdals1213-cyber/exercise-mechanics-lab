"""Adopt one unchanged-controller result after all PPO publishers have exited."""
import copy,json,time,hashlib,shutil
from pathlib import Path
from jump_monitor_data import ROOT,read,atomic
RUN=ROOT/'results/controlled_broad_jump/landing_stability_v3'
WEB=ROOT/'deliverables/jump_landing_live'

def main():
    source=ROOT/'results/controlled_broad_jump/robust_postlanding_v1/validation_cold_0002.json';report=copy.deepcopy(read(source));assert report['passed']and report['goal_met']and not report['cached_screen'];assert {r['dt_s']for r in report['cases']}=={.0005,.00025,.000125,.0000625};assert all(r['stable']and r['eligible']and r['retained_distance_m']>=3. for r in report['cases'])
    cutoff=max(r['finished_at']for r in report['cases']);local=[];counts=[]
    for name in ['robust_finish_v1','robust_continuation_v1','robust_launch_batch_v1','robust_knee_refine_v1','robust_postlanding_v1','robust_landing_feedback_v1','postcontact_screen_v1']:
        rows=[]
        for file in(ROOT/'results/controlled_broad_jump'/name).glob('worker_*/episodes.jsonl'):
            for line in file.read_text(encoding='utf-8').splitlines():
                try:row=json.loads(line)
                except ValueError:continue
                if row.get('parameters')and row.get('finished_at'):rows.append(row)
        local.extend(rows);counts.append(dict(run=name,completed_native_evaluations=len(rows)))
    def unique(rows):return len({hashlib.sha256(json.dumps(r['parameters'],sort_keys=True).encode()).hexdigest()for r in rows})
    discovered=[r for r in local if r['finished_at']<=cutoff];history=[json.loads(s)for s in(RUN/'history.jsonl').read_text(encoding='utf-8').splitlines()if s.strip()];training=[r for r in history if r['kind']=='ppo_train'];state=read(RUN/'live.json');assert state['status']in ['checkpointed','goal_found','budget_finished']
    counters=dict(cutoff_finished_at=cutoff,local_parameter_proposals_at_discovery=unique(discovered),local_native_simulations_at_discovery=len(discovered),total_completed_local_parameter_proposals=unique(local),total_completed_local_native_simulations=len(local),PPO_completed_trials=len(training),PPO_policy_updates=state['policy_updates'],PPO_trials_at_discovery=sum(r['finished_at']<=cutoff for r in training),per_branch=counts,count_definition='Local proposal = unique full parameter dictionary with at least one completed native rollout. Cold repetitions count as native simulations, not new proposals. Includes candidates rejected by an early timestep screen; excludes the independent seed-reproduction audit.')
    proof=dict(algorithm='PPO followed by bounded numerical postcontact-posture refinement',candidate_number=counters['local_parameter_proposals_at_discovery'],winning_branch='robust_postlanding_v1',winning_branch_candidate_number=2,supplemental_native_evaluations=len(discovered),PPO_training_trials_at_discovery=counters['PPO_trials_at_discovery'],PPO_training_trials_at_stop=len(training),seed_report=str(ROOT/'deliverables/jump_robust_postlanding/seed_validation.json'),not_a_PPO_update=True,all_branch_counts=counters)
    report.update(discovery_provenance=proof,original_cold_report=str(source),original_cold_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),canonical_adoption_after_PPO_exit=True)
    # Only the original58 posture/effort coordinates changed in this result.
    seed=read(ROOT/'deliverables/jump_robust_postlanding/seed_validation.json')['cases'][0]['parameters'];p=report['cases'][0]['parameters'];assert p['joint_torque_limits_Nm']==seed['joint_torque_limits_Nm'];assert all(p[k]==seed[k]for k in ['landing_support_damping','landing_ankle_support_damping','landing_response_trust_Nm','contact_time_constant_s','landing_contact_time_constant_s','integration_method'])
    file=RUN/'validation_adopted_postlanding_three_metres.json';atomic(file,report);atomic(RUN/'verified_best.json',report);row=min(report['cases'],key=lambda r:r['retained_distance_m']);value={k:row.get(k)for k in ['id','distance_m','retained_distance_m','stability','stable','fell']};value.update(motion_id=row['id'],min_fine_first_m=min(r['distance_m']for r in report['cases']),min_fine_retained_m=report['minimum_m'],goal_met=True,timesteps_s=sorted(r['dt_s']for r in report['cases']),supplemental_search=proof)
    state.update(status='goal_found',stage='completed',goal_achieved=True,continue_after_goal=False,verified=value,supplemental_search=proof,heartbeat=time.time(),workers=[],stage_note='네 시간 간격 모두 3m 이상·안정 착지. 캐시 없이 재검증 완료. 기존 토크/접촉/피드백 강도를 유지하고 착지 후 자세6개만 조정한 결과로 확정했습니다. PPO와 추가 수치 탐색 횟수는 별도 집계합니다.');shutil.copyfile(Path(row['folder'])/'motion.json',WEB/'verified_motion.json');atomic(RUN/'live.json',state);atomic(WEB/'status.json',state);atomic(WEB/'final_search_counts.json',counters);print('CANONICAL_THREE_METRES_ADOPTED',report['minimum_m'],json.dumps(counters),flush=True)

if __name__=='__main__':main()
