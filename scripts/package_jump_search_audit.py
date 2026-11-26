"""Publish measured exploration limits; never label a plateau as an optimum."""
from pathlib import Path
import json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jump_monitor_data import ROOT,read,atomic

def main():
    audit=ROOT/'results/jump_search_space_audit_v1';search=read(audit/'search_audit.json');probe=read(audit/'local_search_summary.json');assert probe['qualified_at_3_timesteps'];out=ROOT/'deliverables/jump_search_audit';out.mkdir(exist_ok=True)
    root=ROOT/'results/controlled_broad_jump/landing_stability_v3';history=[json.loads(line)for line in(root/'history.jsonl').read_text().splitlines()];rows=[row for row in history if row['kind']=='ppo_train'][:search['training_episodes']];best=0.;curve=[]
    for i,row in enumerate(rows):
        if row['eligible']and row['stable']:best=max(best,row['retained_distance_m'])
        curve.append(best)
    local=read(audit/'local_search_history.json');localbest=[];value=0.
    for row in local:
        if row['eligible']and row['stable']:value=max(value,row['retained_distance_m'])
        localbest.append(value)
    plt.rcParams.update({'figure.facecolor':'#0d1726','axes.facecolor':'#111f2e','axes.edgecolor':'#50677c','text.color':'#e7eff7','axes.labelcolor':'#b5c6d6','xtick.color':'#b5c6d6','ytick.color':'#b5c6d6','grid.color':'#30465b','font.size':10})
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained');ax[0].plot(np.arange(1,len(curve)+1),curve,color='#f4a46d');ax[0].set_title('Recorded PPO samples / stable best');ax[0].set_xlabel('Actual PPO training episodes');ax[0].set_ylabel('Retained distance (m)');ax[1].plot(np.arange(1,len(localbest)+1),localbest,color='#77dac7');ax[1].axhline(probe['initial_250us_stable_m'],color='#f4a46d',linestyle=':');ax[1].set_title('Same58 variables / recentered CEM');ax[1].set_xlabel('Separate native evaluations (not PPO updates)');ax[1].set_ylabel('Retained distance at0.25ms (m)')
    for a in ax:a.grid(alpha=.4);a.set_ylim(2.15,2.6)
    fig.suptitle('A search plateau does not establish the model maximum');fig.savefig(out/'search_comparison.png',dpi=150);plt.close(fig)
    atomic(out/'summary.json',dict(search_audit=search,local_probe=probe,interpretation='Same58 coordinates and original context1 bounds contain a better three-timestep-qualified solution. Conservative distribution and poor reuse of the best feasible sample are current bottlenecks. This does not establish global adequacy of the staged motion basis or a human strength limit.'))
    table=''.join(f"<tr><td>{r['key']}</td><td>{r['physical_one_sigma']:.4g} {r['unit']}</td><td>{r['nominal_lower']:.3g} – {r['nominal_upper']:.3g}</td></tr>"for r in search['ranges']if r['key']in ['push_s','hip_bottom','knee_bottom','ankle_delay','prepare_landing_s','landing_knee'])
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>점프 탐색 공간 점검</title><style>:root{{color-scheme:dark}}body{{background:#0d1726;color:#e7eff7;font:16px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:0 22px}}h1{{font-size:27px}}h2{{font-size:21px}}img{{width:100%}}a{{color:#77dac7}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #30465b}}table{{width:100%;border-collapse:collapse;font-size:13px}}.note{{border-left:3px solid #f4a46d;padding-left:16px}}.scroll{{overflow:auto}}small{{color:#aac0d2}}</style><h1>2.46m는 최적해인가?</h1><p><a href="/landing_live.html">실시간 학습 모니터</a></p><p class="note"><b>현재 정체를 근력 한계나 최적해로 판단할 수 없습니다.</b> 같은58개 변수, 같은 기존 허용 범위, 같은 제어기·토크 상한으로 별도 탐색을 하자 더 나은 안정 해를 찾았습니다. 0.25ms 유지 거리는 {probe['initial_250us_stable_m']:.3f} → {probe['best_250us_stable_m']:.3f}m, 3개 시간 스텝을 모두 통과한 최소 유지 거리는 {probe['min_3dt_retained_m']:.3f}m입니다.</p><img src="search_comparison.png" alt="Actual PPO plateau and separate local parameter search"><h2>실제 탐색 폭</h2><p>점검 시점 {search['training_episodes']}회 학습 중 최고 안정 표본은 {search['best_training_sample_episode']}번째에서 나왔고 이후 {search['episodes_since_best_training_sample']}회 동안 갱신되지 않았습니다. 정규화 행동의 표준편차는 대부분 약0.10이었고, 전체 행동 성분 중 절댓값0.8 이상에 도달한 비율은0%였습니다. 실제 물리 변수로 바꾸면 도약 목표 시간의 표준편차가 약7ms, 깊이 관련 관절 목표 각도는 약1도였습니다.</p><div class="scroll"><table><tr><th>변수</th><th>실제 탐색 표준편차</th><th>기존 context1 허용 범위</th></tr>{table}</table></div><p>기존 최고 해의 전 물리 계산 스텝을 재계산해 확인한 추진 구간 토크 피크는 상한 대비 고관절 약65%, 무릎 약30%, 발목 약68%였습니다. 이 사실이 더 높은 힘을 항상 써야 한다는 뜻은 아니며, 2.46m가 모든 관절의 상한에 막힌 결과라는 증거가 없다는 뜻입니다.</p><h2>58개 변수로 표현하지 못하는 것</h2><p>현재 신경망은 한 번에 동작 전체의 자세·타이밍43개와 추진 중 등가 토크 곡선15개를 출력합니다. 단계별 목표 자세를 정해진 보간 형태로 연결하며, 토크 보조 곡선은 근육군이 아닌5개 관절 계열에3개의 자유 중간점을 둡니다. 매 순간 신체 상태를 관측해 토크를 출력하는 학습형 제어기는 아니며, 착지 피드백은 별도의 규칙 기반 제어입니다. 동작은 시상면·좌우 대칭으로 제한돼 있습니다.</p><p>따라서 현재 변수들이 모든 가능한 점프 동작의 최적해를 표현한다고 보장할 수 없습니다. 범위 확대, 관절별 동작 곡선의 자유점, 팔꿈치·체간의 독립적인 시간 변화, 상태 피드백 정책은 각각 표현력을 바꾸는 별도 확장입니다. 변수 수만 늘리기 전에 현재 변수의 활용과 탐색 초기화 문제를 먼저 분리했습니다.</p><h2>적용한 다음 탐색</h2><p>기존 실제 정책과 실패 기록을 보존하고, 새로 검증한 최고 해로 Gaussian 행동 분포를 옮깁니다. 추진력 탐색 폭과 민감한 착지 자세 탐색 폭을 분리하며, 안정 표본 비율에 따라 폭을 조절합니다. 같은58개 변수의 원래 허용 범위와 힘 한계를 유지한 실제 PPO가 이어집니다. 분포 재초기화와 CEM 탐색을 PPO 갱신 횟수에 포함하지 않습니다.</p><p>별도 검사는 CEM3세대·39회 네이티브 계산이며 PPO 학습이 아닙니다. 이 개선만으로 전역 최적해나 충분한 표현력을 증명하지는 않습니다.</p><p><a href="summary.json">변수 범위·실제 표본·정밀 검증 근거</a></p><small>Same body, fixed joint torque capacities and native physics. Best found in a restricted parameterized motion space.</small><script>if(location.protocol!=='file:'){{document.querySelector('img').src='/search_audit/search_comparison.png';document.querySelector('a[href="summary.json"]').href='/search_audit/summary.json'}}</script></html>'''
    (out/'index.html').write_text(html,encoding='utf-8');print('SEARCH_AUDIT_REPORT_READY',probe['min_3dt_retained_m'],flush=True)

if __name__=='__main__':main()
