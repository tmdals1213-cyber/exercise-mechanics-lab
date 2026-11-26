"""Separate terminal snapshots from stale active heartbeats; show actual goals."""
from jump_monitor_data import ROOT

def update(text):
    if 'function monitorFreshness('in text:return text
    edits=[
        ("budget_finished:'설정한 계산 범위 완료'", "budget_finished:'설정한 계산 범위 완료',checkpointed:'체크포인트 저장 / 재개 대기'"),
        ("function setup(canvas)","function monitorFreshness(s,now=Date.now()/1000){const terminal=['budget_finished','goal_found','checkpointed','failed'].includes(s.status)||s.stage==='completed';return{terminal,stale:!terminal&&now-s.heartbeat>15}}\nfunction setup(canvas)"),
        ("const age=Date.now()/1000-s.heartbeat,stale=age>15;", "const {terminal,stale}=monitorFreshness(s);$('target-label').textContent=`첫 접촉 ≥ ${fmt(s.target_first_m)}m · 착지 유지 ≥ ${fmt(s.target_retained_m)}m`;$('target-legend').textContent=`점선: 유지 거리 목표 ${fmt(s.target_retained_m)}m`;"),
        ("<b>첫 접촉 ≥ 2.85m · 착지 유지 ≥ 2.80m</b>","<b id=\"target-label\">현재 목표 불러오는 중…</b>"),
        ("<span>점선: 유지 거리 목표 2.80m</span>","<span id=\"target-legend\">점선: 현재 유지 거리 목표</span>"),
        ("target:2.80", "target:s.target_retained_m??2.80"),
        (":s.deadline?`최대 ${s.planned_training_episodes?.toLocaleString()}회 · 계산 종료 한도 ${new Date(s.deadline*1000).toLocaleTimeString('ko-KR')} · 목표 후보 검증 통과 시 조기 종료`:'';",":terminal?`계산 저장 완료 · ${labels[s.status]||s.status} · 마지막 갱신 ${new Date(s.heartbeat*1000).toLocaleString('ko-KR')}`:s.deadline?`최대 ${s.planned_training_episodes?.toLocaleString()}회 · 계산 종료 한도 ${new Date(s.deadline*1000).toLocaleTimeString('ko-KR')} · ${s.continue_after_goal?'목표 통과 후에도 추가 탐색':'목표 후보 검증 통과 시 조기 종료'}`:'';")
    ]
    for before,after in edits:
        assert text.count(before)==1,(before,text.count(before));text=text.replace(before,after)
    return text

def main():
    for path in [ROOT/'scripts/jump_landing_live.html',ROOT/'deliverables/jump_landing_live/index.html']:
        result=update(path.read_text(encoding='utf-8'));path.write_text(result,encoding='utf-8')
    print('MONITOR_STATUS_AND_GOALS_FIXED',flush=True)

if __name__=='__main__':main()
