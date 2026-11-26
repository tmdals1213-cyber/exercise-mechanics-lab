"""Check native provenance, no-COM renders, force caps and the encoded reel."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,imageio_ffmpeg
from PIL import Image
from jump_monitor_data import ROOT,read,atomic
from build_motion_study_reel import BG
OUT=ROOT/'deliverables/jump_rl_study_draft_v2'

def main():
    m=read(OUT/'source_manifest.json');d=read(OUT/'draft_delivery.json');timeline=read(OUT/'timeline.json');metrics=read(OUT/'metrics.json');cases={c['key']:c for c in m['cases']};source_checks=[];fall_ids=[];ref=None
    for c in m['cases']:
        f=Path(c['folder']);src=Path(c['source']);z=np.load(f/'motion.npz');original=np.load(src/'control_motion.npz');assert np.array_equal(z['qpos'],original['qpos']);setup=read(f/'study_render_setup.json');assert setup['com_marker_visible']is False
        r=read(f/'control_summary.json');names=read(f/'joint_names.json');caps=np.array([r['parameters']['joint_torque_limits_Nm'][n]for n in names]);ratio=float(np.max(abs(z['joint_net_torque_Nm'])/caps));assert ratio<=1.000001,(c['key'],ratio)
        if ref is None:ref=r['parameters']['joint_torque_limits_Nm']
        assert ref==r['parameters']['joint_torque_limits_Nm']
        folder=f/c['render_dir'];images=list(folder.glob('frame_*.png'));assert images;last=max(int(p.stem.split('_')[-1])for p in images);assert last==round(c['duration_s']*c['render_fps'])
        # Test framing at anatomical extrema and around actual contact events.
        x=z['body_xpos'][:,1:,1];height=z['body_xpos'][:,1:,2];ids={0,len(z['times'])-1,int(np.unravel_index(x.argmin(),x.shape)[0]),int(np.unravel_index(x.argmax(),x.shape)[0]),int(np.unravel_index(height.argmax(),height.shape)[0])};edges=[]
        for i in ids:
            fi=int(np.clip(round(z['times'][i]/c['duration_s']*last),round(c['render_start_s']*c['render_fps']),last));p=folder/f'frame_{fi:04d}.png'
            with Image.open(p)as raw:bbox=raw.getchannel('A').getbbox()
            if bbox:assert 3<bbox[0]and bbox[2]<1077 and 192<bbox[1]and bbox[3]<1012,(c['key'],fi,bbox)
            edges.append(dict(frame=fi,bbox=bbox))
        if c['role']=='montage':
            assert c['record']['fell'];follow=float(z['times'][-1])-r['first_body_floor_contact_s'];assert follow>=.45;fall_ids.append(c['record']['id'])
        source_checks.append(dict(key=c['key'],exact_native_qpos=True,com_marker_visible=False,max_sampled_net_torque_cap_ratio=ratio,framing=edges))
    assert len(fall_ids)==len(set(fall_ids))==12
    assert [s['case']for s in timeline['segments']if s['kind']=='final_stage']==m['final_four_keys']and len(m['final_four_keys'])==4
    a,b=[cases[k]for k in['policy_before','policy_after']];assert a['record']['fell']and not a['record']['stable']and b['record']['stable'];assert m['policy_example_updates']==[38,40]
    example=[]
    for key,c in [('policy_before',a),('policy_after',b)]:
        q=read(Path(m['run'])/f'forward_update_{c["record"]["checkpoint_update"]:04d}_after.json');error=float(abs(np.array(q['gaussian_mean'])-c['record']['action']).max());assert error<1e-5;example.append(dict(key=key,checkpoint_action_error=error))
    if '--sources-only'in sys.argv:
        atomic(OUT/'source_check.json',dict(passed=True,native_sources=source_checks,policy_example=example,distinct_native_falls=12,all_torque_limits_identical=True));print('V2_NATIVE_SOURCE_CHECK_PASS',len(source_checks),flush=True);return
    stream=imageio_ffmpeg.read_frames(d['video'],pix_fmt='rgb24');meta=next(stream);assert tuple(meta['size'])==(1080,1920)and meta['fps']==30;count=0;error=0;bg=np.array(BG);target={round((s['start_s']+min(1.,(s['end_s']-s['start_s'])/2))*30):s['kind']for s in timeline['segments']if s['kind']in['rl_example','montage','final_stage','four_stage_summary']}
    for raw in stream:
        im=np.frombuffer(raw,dtype=np.uint8).reshape(1920,1080,3);error=max(error,int(abs(im[1536:].astype(int)-bg).max()))
        if count in target:Image.fromarray(im).save(OUT/f'encoded_{target[count]}_{count:04d}.png')
        count+=1
    assert count==d['frames']and error<=4,(count,error)
    qa=dict(passed=True,frames=count,fps=30,resolution=meta['size'],duration_s=count/30,footer_reserved_px=384,encoded_footer_max_rgb_error=error,native_sources=source_checks,policy_example=example,distinct_native_falls=12,ghosts_per_trial=3,final_four_keys=m['final_four_keys'],all_torque_limits_identical=True,individual_muscle_tension=False,video_sha256=hashlib.sha256(Path(d['video']).read_bytes()).hexdigest());atomic(OUT/'delivery_check.json',qa)
    caption='Can AI learn to jump farther — without getting stronger?\nFixed torque limits. Thousands of simulated attempts.\nFrom the first stable jump to four stages of progress.\nAn ongoing biomechanics study. The search is still running.\n';(OUT/'caption_en.txt').write_text(caption,encoding='utf-8')
    distances=' → '.join(f'{x:.2f}'for x in m['final_four_distances_m'])+' m'
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jump study · second draft</title><style>body{margin:0;background:#0c1827;color:#e9eff5;font:16px/1.7 system-ui}main{max-width:1100px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:20px;margin-top:28px}a{color:#77dac7}.layout{display:grid;grid-template-columns:minmax(280px,440px) 1fr;gap:30px}video{width:100%;max-height:85vh;border-radius:16px}small{color:#a8bdce}.card,pre{padding:16px;background:#142438;border-radius:12px}pre{white-space:pre-wrap;font:inherit}img{width:min(100%,360px)}@media(max-width:760px){.layout{display:block}main{padding:18px}}</style><main><h1>Biomechanics × reinforcement learning</h1><p>영상 폼 V2 · DURATION초 · 1080×1920 · 하단 384px 비움</p><div class="layout"><div><video controls playsinline preload="metadata" poster="/rl_draft_v2/preview_four_stages.png" src="/rl_draft_v2.mp4"></video><p><a href="/rl_draft_v2.mp4" download>MP4 다운로드</a> · <a href="/rl_draft.html">이전 초안</a></p></div><div><div class="card"><b>추가 탐색</b><p id="live">상태 불러오는 중…</p><a href="/landing_live.html">실시간 동작·학습 모니터</a></div><h2>이번 수정</h2><p>무게중심 구 제거 → 토크 제어 소개 → 브레인·실제 신경망과 정책 전후 예시 → 실제 실패 12건 및 개선안의 잔상 → 초기부터 최장 안정 착지까지 네 단계 → 도약 부하 비교.</p><h2>마지막 네 단계</h2><p><b>DISTANCES</b></p><p>세 계산 간격에서 안정 착지를 확인한 후보입니다. 네 번째 후보는 이번 영상용 임시 기록이며, 추가 탐색이 끝나면 그 부분을 교체합니다. 전역 최적해라는 의미는 아닙니다.</p><h2>학습이 동작에 반영되는 예시</h2><p>같은 조건의 실제 정책 평균 동작을 업데이트 38과 40에서 비교했습니다. 두 PPO 업데이트 사이에 실제 학습 시행 96회가 있습니다. 그 사이 별도의 탐색 중심 재설정은 없습니다. 실제 제어 변수의 작은 차이가 토크와 착지 결과에 반영된 사례이며, 특정 변수 하나의 인과 효과를 분리한 실험은 아닙니다.</p></div></div><h2>네 개 지표와 관절 표시</h2><p>GRF는 양발 지면반력 합력 크기/BW, HIP·KNEE·ANKLE은 오른쪽 순 관절 토크 크기/Nm입니다. 관절의 호 길이는 고정 토크 상한 대비 사용 비율, 화살표는 부호, 각도는 실제 모델 좌표입니다. 동작 그래프는 전체 동작을 위상 정렬했고, 화면 열마다 원시 데이터의 극값을 보존했습니다. 마지막 표는 도약 구간 피크만 비교합니다.</p><p>착지의 매우 짧은 힘 피크는 강체 접촉 모델에 민감합니다. 실측 인체의 착지 부하를 검증했다는 뜻은 아닙니다. 개별 근육 장력은 아직 계산하지 않습니다. 해부학 표면은 실제 관절 동작 표시용입니다.</p><h2>학습과 브레인 표시의 범위</h2><p>58개 출력은 자세·타이밍 43개와 도약의 등가 토크 구동 15개입니다. 정책은 한 점프의 파라미터를 정하고, PD 제어·접촉·착지 상태 제어가 움직임을 만듭니다. 매 프레임 근육 발화를 학습한 모델은 아닙니다. 브레인 그림과 펄스는 개념 그림이며, 신경망은 실제 체크포인트를 축약한 표시입니다. RESEED는 CEM을 사용한 별도 탐색 중심 조정이며 PPO 업데이트로 세지 않았습니다.</p><p>영상은 2,400회·50업데이트까지의 기록을 고정했습니다. 이후 탐색은 최신 검증 후보로 중심을 옮겨 같은 신체·토크 상한·변수 범위 안에서 별도로 재개했고, 모든 실패와 체크포인트를 보존합니다.</p><h2>English caption</h2><pre>CAPTION</pre><h2>자료·출처</h2><p><a href="/rl_draft_v2/source_manifest.json">동작·학습 기록</a> · <a href="/rl_draft_v2/metrics.json">네 단계 힘·토크</a> · <a href="/rl_draft_v2/timeline.json">편집 순서</a> · <a href="/rl_draft_v2/delivery_check.json">검수</a> · <a href="/rl_draft_v2/ANATOMY_CREDITS.md">해부학 모델 전체 크레딧</a></p><p><a href="https://mujoco.readthedocs.io/">MuJoCo</a> · <a href="https://arxiv.org/abs/1707.06347">PPO</a> · <a href="https://huggingface.co/datasets/myohub/myosuite-assets">MyoHub / BodyParts3D © DBCLS</a> · <a href="https://github.com/JohanBellander/BodyExplorer">BodyExplorer / Z-Anatomy</a></p><small>BodyParts3D CC BY 4.0; supplementary Z-Anatomy CC BY-SA 4.0, mapped BP3D CC BY-SA 2.1 Japan; MakeHuman/MPFB CC0. See individual component notices and modification records in full credits. No reference video footage is republished.</small></main><script>async function live(){try{let s=await(await fetch('/landing_live/status.json',{cache:'no-store'})).json();let v=s.verified?.min_fine_retained_m;document.getElementById('live').textContent=`${s.status} · ${s.training_episodes} trials · ${s.policy_updates} PPO updates · checked ${v?.toFixed(2)??'—'} m`;}catch(e){document.getElementById('live').textContent='실시간 모니터 링크에서 상태를 확인해 주세요.';}}live();setInterval(live,5000);</script></html>'''
    html=html.replace('DURATION',f'{count/30:.1f}').replace('DISTANCES',distances).replace('CAPTION',caption);(OUT/'index.html').write_text(html,encoding='utf-8')
    state=ROOT/'docs/standing_long_jump_working_state.json';s=read(state);s['latest_video_draft']=dict(version=2,video=d['video'],folder=str(OUT),duration_s=count/30,com_marker=False,policy_example_updates=[38,40],native_failed_trials=12,ghosts=3,final_four_keys=m['final_four_keys'],final_four_checked_m=m['final_four_distances_m'],last_solution_replaceable=True,individual_muscle_tension=False);s['latest_search_extension']=read(ROOT/'deliverables/jump_landing_live/extension_launch.json');atomic(state,s)
    print('V2_PACKAGE_CHECKED',count,error,distances,flush=True)

if __name__=='__main__':main()
