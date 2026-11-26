"""Package and independently inspect the encoded study draft."""
from pathlib import Path
import json, hashlib
import numpy as np
import imageio_ffmpeg
from PIL import Image
from jump_monitor_data import ROOT, read, atomic

OUT=ROOT/'deliverables/jump_rl_study_draft_v1'

def main():
    m=read(OUT/'source_manifest.json'); delivery=read(OUT/'draft_delivery.json'); timeline=read(OUT/'timeline.json'); metrics=read(OUT/'metrics.json')
    cases={c['key']:c for c in m['cases']}; failures=[s['case'] for s in timeline['segments'] if s.get('case') and cases[s['case']]['record'].get('fell')]
    assert len(failures)==6 and len({cases[k]['source'] for k in failures})==6
    follow=[]
    for key in failures:
        c=cases[key];r=read(Path(c['folder'])/'control_summary.json');after=c['duration_s']-r['first_body_floor_contact_s'];assert after>=.3
        follow.append(dict(key=key,source=c['source'],fall_followthrough_s=after,display_continuation=c.get('display_continuation')))
    trials=[cases[s['case']]['record']['training_trial']for s in timeline['segments']if s['kind']=='montage'];assert trials==sorted(trials)
    assert m['actual_training_rollouts_used']==48*m['actual_completed_PPO_updates']
    assert all(x['source_replay_qpos_error']<1e-10 for x in metrics['cases'].values())
    path=OUT/'standing_long_jump_rl_draft.mp4';stream=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24');meta=next(stream);assert tuple(meta['size'])==(1080,1920) and meta['fps']==30
    count=0;footer_error=0;bg=np.array([12,24,39]);samples={round(t*30):name for t,name in [(1.6,'hook'),(6.5,'learning'),(17.8,'fall'),(23.0,'checked'),(34.5,'comparison'),(37.5,'ending')]}
    # BG is imported from the compositor so an aesthetic edit does not stale this check.
    from compose_jump_rl_draft import BG
    bg=np.array(BG,dtype=int)
    for data in stream:
        a=np.frombuffer(data,dtype=np.uint8).reshape(1920,1080,3);footer_error=max(footer_error,int(np.abs(a[1536:].astype(int)-bg).max()))
        if count in samples:Image.fromarray(a).save(OUT/f'encoded_{samples[count]}.png')
        count+=1
    assert count==delivery['frames'],(count,delivery['frames']);assert footer_error<=4,footer_error
    qa=dict(passed=True,frames=count,fps=meta['fps'],duration_s=count/30,resolution=meta['size'],bottom_blank_px=384,encoded_footer_max_rgb_error=footer_error,distinct_failed_native_trials=follow,montage_trial_order=trials,source_qpos_replay_matches=True,push_off_only_comparison=True,individual_muscle_tension_available=False,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    atomic(OUT/'delivery_check.json',qa)
    initial=m['baseline_retained_m'];best=m['best_checked_min_retained_m']
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Standing long jump — study draft</title>
<style>body{margin:0;background:#0c1827;color:#e9eff5;font:16px/1.7 system-ui}main{max-width:1060px;margin:auto;padding:28px}h1{font-size:27px}h2{font-size:20px;margin-top:30px}a{color:#77dac7}video{width:min(100%,420px);max-height:82vh;border-radius:16px;background:#0c1827}section{display:grid;grid-template-columns:minmax(280px,430px) 1fr;gap:32px}small{color:#a8bdce}.tag{color:#77dac7}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #23374a}pre{white-space:pre-wrap;font:inherit;background:#142438;padding:16px;border-radius:10px}@media(max-width:760px){section{display:block}main{padding:18px}}</style>
<main><h1>Can AI find the best jump?</h1><p class="tag">38초 세로 영상 · 첫 편집 초안 · 연구 진행 중</p><section><div><video controls playsinline preload="metadata" poster="/rl_draft/preview_hook.png" src="/rl_draft.mp4"></video><p><a href="/rl_draft.mp4" download>MP4 다운로드</a></p></div><div>
<h2>구성</h2><p>질문 → 같은 신체와 토크 한계 → 실제 시행·점수·정책 업데이트 → 실패와 개선 → 현재 최선 2회 → 도약 부하 비교 → 열린 결말.</p>
<p>실제 서로 다른 실패 6건을 사용했습니다. 접촉 순간에서 멈추지 않고 넘어지는 과정도 보여줍니다. 현재 결과는 정상 속도와 0.65×로 한 번씩 재생합니다.</p>
<h2>이번 편집에 고정한 기록</h2><p>178 cm · 87 kg, <b>912회 학습 시행 / 19회 PPO 업데이트</b>까지의 스냅샷입니다. 계산은 별도로 계속됩니다.</p><p>안정 착지 후 유지 거리: <b>INITIAL → BEST m</b>. 표시된 최선은 전역 최적해를 의미하지 않습니다.</p>
<h2>비교 지표</h2><p>양발 지면반력의 합력 크기(BW)와 오른쪽 고관절·무릎·발목의 순 관절 토크 크기(Nm). 모두 도약 구간 피크입니다. 곡선은 도약 시작부터 이륙까지 위상 정렬했습니다.</p><p><b>개별 근육 장력은 이번 제어 모델에서 아직 계산하지 않습니다.</b> 해부학 표면은 동작 표시용이며, 관절 토크를 근육 장력으로 바꿔 표시하지 않았습니다. 착지 충격 피크는 이번 비교 범위에 포함하지 않았습니다.</p>
</div></section><h2>분석 범위</h2><p>강체 분절·관절, 중력, 마찰 접촉을 사용합니다. 58개 자세·타이밍·도약 힘 파라미터를 정책이 한 번에 정하고, PD 추종과 착지 상태 제어가 실제 동작을 만듭니다. 매 프레임 전체 근육을 직접 제어하는 정책은 아닙니다. 가상 관절 토크 상한은 비교 중 고정했으며 개인의 실측 근력으로 해석할 수 없습니다.</p><p>RESEED는 같은 탐색 범위의 CEM 진단 결과로 탐색 중심을 옮긴 단계입니다. PPO 정책 업데이트 횟수에 포함하지 않았습니다. 신경망 색상은 실제 체크포인트와 전방 계산을 사용했고, 축약한 노드와 이동 펄스는 계산 흐름의 개념 표시입니다.</p>
<h2>English caption</h2><pre>CAPTION</pre>
<h2>자료와 출처</h2><p><a href="/rl_draft/metrics.json">도약 지표</a> · <a href="/rl_draft/timeline.json">편집 순서</a> · <a href="/rl_draft/delivery_check.json">영상 검수</a> · <a href="/rl_draft/source_manifest.json">원본 시행·체크포인트 기록</a> · <a href="/rl_draft/ANATOMY_CREDITS.md">해부학 모델 전체 크레딧</a></p><p><a href="https://mujoco.readthedocs.io/">MuJoCo</a> · <a href="https://arxiv.org/abs/1707.06347">PPO</a> · <a href="https://huggingface.co/datasets/myohub/myosuite-assets">MyoHub / BodyParts3D © DBCLS</a> · <a href="https://github.com/JohanBellander/BodyExplorer">BodyExplorer / Z-Anatomy</a></p><small>BodyParts3D CC BY 4.0; supplementary Z-Anatomy CC BY-SA 4.0 and mapped BP3D CC BY-SA 2.1 Japan; MakeHuman/MPFB skin CC0. Individual notices and modifications are retained in the full credits. Public reference footage has not been reused.</small></main></html>'''
    page=page.replace('INITIAL',f'{initial:.2f}').replace('BEST',f'{best:.2f}').replace('CAPTION',(OUT/'caption_en.txt').read_text(encoding='utf-8'))
    page=page.replace('표시된 최선은 전역 최적해를 의미하지 않습니다.','표시된 최선은 전역 최적해를 의미하지 않습니다. BEST CHECKED는 0.5·0.25·0.125ms 간격에서 안정 착지를 확인한 후보의 최소 유지 거리입니다. 학습 그래프의 SAMPLED 기록은 단일 간격의 표본 결과여서 이 값과 다를 수 있습니다.')
    (OUT/'index.html').write_text(page,encoding='utf-8')
    state=ROOT/'docs/standing_long_jump_working_state.json';s=read(state);s['latest_video_draft']=dict(folder=str(OUT),video=str(path),status='encoded_and_checked',duration_s=count/30,frozen_training_trials=912,frozen_PPO_updates=19,individual_muscle_tension=False,ending='SEARCH CONTINUES',notes='Native distinct failed trials; full fall follow-through; push-off GRF and right joint torques. Background training separate.');atomic(state,s)
    print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
