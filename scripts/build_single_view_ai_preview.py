"""One-view layout prototype: actual untrained inference + actual seed motion.

No PPO learning, invented rewards or weight updates are depicted. The separate
storyboard describes how real training checkpoints will populate the final reel.
"""
from pathlib import Path
import json,math
import numpy as np,torch,imageio_ffmpeg
from PIL import Image,ImageDraw
from stable_baselines3 import PPO
from jump_monitor_data import ROOT,read,atomic
from build_motion_study_reel import font,BG,FG,MUTED,ACC

OUT=ROOT/'deliverables/jump_ai_single_view_concept'
SOURCE=ROOT/'deliverables/controlled_jump_pilot/effort_seed_anatomy_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    design=read(ROOT/'results/controlled_broad_jump/effort_launch_seed_v2/policy_design.json');model=PPO.load(ROOT/'deliverables/standing_long_jump_effort_seed/effort_launch_policy_initial.zip',device='cpu');obs=np.array(design['observation'],np.float32);acts=[];x=torch.as_tensor(obs[None])
    with torch.no_grad():
        for layer in model.policy.mlp_extractor.policy_net:
            x=layer(x)
            if isinstance(layer,torch.nn.Tanh):acts.append(x[0].numpy().copy())
        mean=model.policy.action_net(x)[0].numpy();std=model.policy.log_std.exp().numpy()
    assert len(acts)==2 and all(len(a)==64 for a in acts)and len(mean)==52 and np.max(abs(mean))==0
    atomic(OUT/'inference_evidence.json',dict(checkpoint='deliverables/standing_long_jump_effort_seed/effort_launch_policy_initial.zip',training_episodes=0,observation=obs.tolist(),hidden_activations=[a.tolist()for a in acts],mean_offsets=mean.tolist(),std=std.tolist(),diagram_actual_layers=[7,64,64,52],diagram_display_nodes=[7,8,8,3],diagram_is_compressed=True,weight_update_depicted=False))
    z=np.load(SOURCE/'motion.npz');r=read(SOURCE/'control_summary.json');camera=read(SOURCE/'fixed_side_camera_v2.json');files=sorted((SOURCE/'fixed_side_render_v2').glob('frame_*.png'));T=float(z['times'][-1]);lead=.70;hold=1.05;duration=lead+T+hold;fps=30;frames=round(duration*fps)
    images=[]
    for file in files:
        with Image.open(file)as im:images.append(im.convert('RGBA').crop((0,190,1080,980)))
    toe=camera['initial_foremost_sole_y_m'];ppm=1080/camera['ortho_width_m'];centre=camera['camera_center_forward_m'];cz=camera['camera_center_z_m'];top=330;ground=top+525+cz*ppm-190;zero=540-centre*ppm;distance=r['first_contact_heel_distance_m'];flight=np.flatnonzero((z['times']>=r['takeoff_s'])&(z['times']<=r['touchdown_s']))
    levels=[abs(obs),np.mean(abs(acts[0]).reshape(8,8),axis=1),np.mean(abs(acts[1]).reshape(8,8),axis=1),np.zeros(3)];node_x=[610,705,800,895];node_y=[np.linspace(180,344,len(a))for a in levels]
    def text(d,xy,value,size=24,color=FG,bold=False):d.text(xy,value,font=font(size,bold),fill=color)
    def project(v):return(540+(-v[1]+toe-centre)*ppm,top+525-(v[2]-cz)*ppm-190)
    def render(clock):
        t=float(np.clip(clock-lead,0,T));fi=min(round(t/T*(len(images)-1)),len(images)-1);im=Image.new('RGB',(1080,1920),BG);im.paste(images[fi],(0,top),images[fi]);d=ImageDraw.Draw(im)
        text(d,(45,22),'@biomechanics_ksm',20,MUTED,True);text(d,(45,65),'AI + PHYSICS',44,FG,True);text(d,(45,128),'FIXED JOINT LIMITS',23,ACC,True)
        active=min(3,int(clock/lead*4))if clock<lead else -1
        for k in range(3):
            for a,y0 in enumerate(node_y[k]):
                for b,y1 in enumerate(node_y[k+1]):
                    col=(51,92,111)if active==k else(29,49,66);d.line((node_x[k],y0,node_x[k+1],y1),fill=col,width=1)
        for k,values in enumerate(levels):
            for j,value in enumerate(values):
                radius=7 if k<3 else 10;bright=active==k;v=float(np.clip(value,0,1));col=(106,220,204)if bright else tuple(round(a+(b-a)*v)for a,b in zip((40,66,84),(99,165,172)));x0=node_x[k];y0=node_y[k][j];d.ellipse((x0-radius,y0-radius,x0+radius,y0+radius),fill=col,outline=(134,188,196)if bright else(54,86,108),width=1)
        for y,label in zip(node_y[-1],['POSTURE','TIMING','EFFORT']):text(d,(913,y-9),label,15,MUTED,True)
        for px,value in zip(node_x,['7','64','64','52']):text(d,(px-9,369),value,16,MUTED)
        d.line((35,ground,1045,ground),fill=(91,116,138),width=2)
        for meter in np.arange(0,3.01,.5):
            px=zero+meter*ppm;d.line((px,ground-6,px,ground+13),fill=MUTED,width=2);text(d,(px-15,ground+19),f'{meter:.1f}',18,MUTED)
        d.line((zero,ground-32,zero,ground+15),fill=FG,width=3)
        # The native anatomical render already contains the COM marker.
        # Do not draw a second marker using an independently rounded time.
        contact=t>=r['touchdown_s']-.001
        if contact:
            d.line((zero,ground,zero+distance*ppm,ground),fill=ACC,width=5);u=float(np.clip((clock-lead-T)/.65,0,1));n=round(u*len(flight));points=[project(z['com'][j])for j in flight[:n]]
            if len(points)>1:d.line(points,fill=ACC,width=3)
        stage=0 if clock<lead else 1 if not contact else 2
        for k,(px,label)in enumerate([(45,'PROPOSE'),(405,'SIMULATE'),(805,'SCORE')]):
            text(d,(px,1210),label,22,ACC if k==stage else(87,111,131),True)
        if contact:
            text(d,(45,1300),'CONTACT CANDIDATE',24,MUTED,True);text(d,(45,1342),f'{distance:.2f} m',58,ACC,True);text(d,(45,1422),'FIRST CONTACT',20,MUTED,True);text(d,(600,1422),'RECOVERY UNRESOLVED',20,ACC,True)
        else:
            text(d,(45,1330),'TIMING + POSTURE + EFFORT',24,MUTED,True)
        assert np.all(np.asarray(im)[1536:]==BG)
        return im
    render(lead+(r['takeoff_s']+r['touchdown_s'])/2).save(OUT/'layout_preview.png');render(duration-.05).save(OUT/'layout_ending.png')
    writer=imageio_ffmpeg.write_frames(str(OUT/'single_view_layout_preview.mp4'),(1080,1920),fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None)
    try:
        for k in range(frames):writer.send(np.asarray(render(k/fps)))
    finally:writer.close()
    storyboard=dict(status='layout_and_story_proposal_not_a_learning_result',language='English in video; concise supporting Korean notes',portrait=[1080,1920],bottom_reserved_px=384,main_view='One world-fixed side view with the same anatomical model; a compact AI panel shares the frame.',screen_words=['SAME LIMITS','SEARCH','BEST FOUND'],proposed_duration_s=20,shots=[dict(time_s=[0,2.5],purpose='Hook and initial motion',on_screen='SAME LIMITS',content='Show the actual starting policy jump, fixed camera/ruler and baseline marker.'),dict(time_s=[2.5,7],purpose='One readable learning batch, optionally two',on_screen='TRY / SCORE / UPDATE',content='Actual proposal -> selected actual jump -> recorded reward -> actual PPO batch update. Do not depict weight updates after every frame or individual rollout.'),dict(time_s=[7,13],purpose='Fast optimization montage',on_screen='SEARCH',content='Different real candidate motions, actual episode/update count, best-so-far progress. Use native replay for selected actions, not a repeated clip with changed numbers.'),dict(time_s=[13,20],purpose='Best found replay',on_screen='BEST FOUND',content='Highest qualified candidate, replayed normally then slower as timing permits; retain the initial takeoff line and baseline distance mark. No global-optimum claim.')],metrics=dict(primary='First-contact heel projection in the declared model domain',constraint='Shared fixed net joint-torque capacity; not individual muscle-force limits',recovery='Separate assessment, not proof of human recoverability'),preview=dict(file='single_view_layout_preview.mp4',uses_actual_seed_anatomy=True,uses_actual_untrained_policy_forward_activations=True,training_episodes=0,weight_update_depicted=False,final_label='INITIAL SEED',note='This preview establishes the one-view layout only. The planned optimization montage requires actual new PPO training.'),required_training_evidence=['Initial model and before/after checkpoints for early PPO updates','Actual observations/actions/log-probabilities/rewards','Native qpos/body transforms for selected episodes','Mean policy versus sampled best, separately evaluated','Fine-timestep qualification of the candidate shown at the end'],proposed_caption_lines=['Fixed joint-torque limits.','AI adjusts timing, posture, and effort in simulation.','First-contact distance; landing recovery evaluated separately.','Best found within this model.'])
    storyboard['preview'].update(final_label='CONTACT CANDIDATE',recovery_status='Failed in the250us full replay; the preview stops at first contact and is not a stable landing.')
    storyboard['required_training_evidence'].append('A completed-jump final candidate must pass recovery qualification. A falling first-contact candidate must never be labelled a stable/optimal full jump.')
    atomic(OUT/'storyboard.json',storyboard);atomic(ROOT/'docs/jump_ai_optimization_reel_plan.json',storyboard)
    page=PAGE
    page=page.replace('<h1>AI와 실제 동작을 한 화면에</h1>','<h1>AI와 실제 동작을 한 화면에</h1><p class="note"><b>이전 구도 미리보기입니다.</b> 현재 2.89m 후보는 안정 착지에 실패했습니다. 정지 화면은 버틴 상태가 아닙니다. 이 시도는 폐기하지 않고 거리와 회복 점수·낙상 감점을 모두 다음 학습에 사용했습니다. <a href="/distance_stability.html">낙상까지 보여주는 새 영상과 실제 PPO 연결 결과</a></p>')
    (OUT/'index.html').write_text(page,encoding='utf-8');print('SINGLE_VIEW_AI_LAYOUT_READY',OUT/'single_view_layout_preview.mp4',frames,flush=True)

PAGE='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI + Jump / Single-view concept</title><style>:root{color-scheme:dark}body{background:#0d1726;color:#edf3fb;font:16px/1.7 system-ui;max-width:1080px;margin:30px auto;padding:0 22px}h1{font-size:27px}h2{font-size:19px;margin-top:32px}.muted,small{color:#a6bcd1}video{display:block;max-height:85vh;max-width:100%;margin:20px auto}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #2c4054}a{color:#79d9c8}.note{border-left:3px solid #ef9558;padding-left:18px}code{color:#ef9558}</style><h1>AI와 실제 동작을 한 화면에</h1><p class="note"><b>아래 영상은 구도 미리보기입니다.</b> 현재 초기해의 실제 인체 렌더링과, 학습 전 52출력 정책의 실제 forward 값을 사용했습니다. 학습·보상 향상·역전파를 완료한 영상은 아닙니다. 네트워크 그림은 읽기 쉽게 압축했습니다.</p><video id="preview" controls playsinline></video><h2>최종 영상 구상 / 약 20초</h2><table><tr><th>구간</th><th>화면</th><th>핵심 문구</th></tr><tr><td>0–2.5s</td><td>기준 점프. 한 카메라·같은 힘 한계를 보여줌.</td><td>SAME LIMITS</td></tr><tr><td>2.5–7s</td><td>한두 번은 제안→동작→평가→실제 학습 배치 갱신을 읽을 수 있게 보여줌.</td><td>TRY / SCORE / UPDATE</td></tr><tr><td>7–13s</td><td>서로 다른 실제 후보 동작과 탐색 진행을 빠르게 재생.</td><td>SEARCH</td></tr><tr><td>13–20s</td><td>이번 탐색의 최선 후보를 크게, 천천히 재생. 기존 거리 표식으로 비교.</td><td>BEST FOUND</td></tr></table><h2>근육 힘 제약과 관절 토크 제약은 구분</h2><p>현재는 각 관절의 <b>순 회전력 한계</b>를 고정합니다. 같은 순 토크라도 길항근 동시수축과 근육 힘 분배는 달라질 수 있으므로, 개별 근육의 힘 상한을 보장하지 않습니다. 두 관절 이상에 걸친 근육의 능력 공유도 현재의 독립 토크 상한에는 들어 있지 않습니다.</p><p><code>τ = R(q) F</code>에서 근육 힘과 모멘트암을 연결하는 모델을 추가하면 근육 힘 제약을 직접 다룰 수 있습니다. 현재 상한은 기준 동작 피크의 1.5배로 만든 가상 비교 조건이며, 개인의 실측 최대 근력을 뜻하지 않습니다.</p><p>OpenSim도 순 관절 모멘트를 개별 근육 힘으로 나누는 것을 별도 최적화 문제로 다룹니다. <a href="https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089624/Getting%20Started%20with%20Static%20Optimization">공식 문서</a></p><h2>실제 학습 영상에서 지킬 것</h2><p>새 정책은 현재 <b>학습 0회</b>입니다. 초기해 탐색 CEM을 PPO 학습으로 바꾸어 표시하지 않습니다. 최종 제작에서는 실제 체크포인트·출력·보상·동작을 연결하고, 마지막 수치는 전역 최적해가 아닌 <b>이번 조건에서 찾은 최선</b>으로 표시합니다. 근육은 중립색으로 두고 필요하면 계산된 순 관절 토크 사용률을 별도 JOINT EFFORT로 표시합니다.</p><p>세로 1080×1920, 하단 384px 비움. 큰 설명문 대신 AI 패널·동작·거리 변화에 집중하고, 가정과 출처는 게시글이나 마지막 작은 줄로 보냅니다.</p><p><a href="storyboard.json">구상 데이터</a> · <a href="inference_evidence.json">실제 초기 정책 forward 근거</a></p><script>const v=document.getElementById('preview');v.src=location.protocol==='file:'?'single_view_layout_preview.mp4':'/ai_single_view.mp4';v.poster=location.protocol==='file:'?'layout_preview.png':'/ai_single_view.png';</script></html>'''

if __name__=='__main__':main()
