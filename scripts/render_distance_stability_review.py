"""Show actual fall, real batch updates, and the next actual policy motion."""
from pathlib import Path
import json,subprocess,sys,math
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
from jump_monitor_data import ROOT,read,atomic
from build_motion_study_reel import font,BG,FG,MUTED,ACC

RUN=ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1'
OUT=ROOT/'deliverables/jump_distance_stability_review'

def prepare_final(summary):
    label='distance_stability_next_v1';out=ROOT/'deliverables/controlled_jump_pilot'/label
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_control_jump_review.py'),summary['final_evaluation']['folder'],'--label',label],check=True)
    args=[str(ROOT/'.tools/blender-4.5.3-windows-x64/blender.exe'),'--background','--threads','4','--python',str(ROOT/'scripts/render_fixed_side_jump_blender.py'),'--','--kg','0','--folder',str(out),'--analysis-study','--world-side','--ortho-width','4.2','--camera-forward','1.6','--skip-existing']
    with(out/'side_render.log').open('w',encoding='utf-8')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return out

def main():
    summary=read(RUN/'pilot_summary.json');assert summary['training_episodes']==32
    OUT.mkdir(parents=True,exist_ok=True);final=prepare_final(summary)
    sources=[ROOT/'deliverables/controlled_jump_pilot/distance_stability_fall_v1',final]
    rows=[]
    scores=[read(RUN/'comparison.json')['long_distance_fall'],summary['final_evaluation']]
    for folder in sources:
        z=np.load(folder/'motion.npz');r=read(folder/'control_summary.json');cam=read(folder/'fixed_side_camera_v2.json');files=sorted((folder/'fixed_side_render_v2').glob('frame_*.png'))
        assert len(files)==round(z['times'][-1]*30)+1
        images=[]
        for path in files:
            with Image.open(path)as image:images.append(image.convert('RGBA').crop((0,190,1080,1015)))
        rows.append(dict(folder=folder,z=z,r=r,cam=cam,images=images,T=float(z['times'][-1])))
    first=rows[0]['T']+.8;update_time=1.4;second=rows[1]['T']/.70+.8;end_hold=1.6;duration=first+update_time+second+end_hold
    forwards=[read(RUN/'forward_update_0000_before.json'),read(RUN/'forward_update_0002_after.json')]
    def text(d,xy,s,size=24,color=FG,bold=False):d.text(xy,s,font=font(size,bold),fill=color)
    def draw(clock):
        updating=first<=clock<first+update_time;idx=0 if clock<first else 1;row=rows[idx];r=row['r'];z=row['z'];cam=row['cam'];score=scores[idx]
        local=clock if idx==0 else max(0,clock-first-update_time)*.70;t=min(row['T'],local)
        if updating:t=rows[0]['T'];row=rows[0];r=row['r'];z=row['z'];cam=row['cam'];score=scores[0]
        fi=min(round(t/row['T']*(len(row['images'])-1)),len(row['images'])-1)
        im=Image.new('RGB',(1080,1920),BG);im.paste(row['images'][fi],(0,360),row['images'][fi]);d=ImageDraw.Draw(im)
        text(d,(45,22),'@biomechanics_ksm',20,MUTED,True)
        text(d,(45,65),'DISTANCE + CONTROL',42,FG,True)
        title='UPDATE'if updating else'INITIAL TRY'if idx==0 else'NEXT TRY'
        text(d,(45,138),title,30,ACC,True)
        text(d,(45,185),'32 TRIALS / 2 UPDATES'if idx else'FIXED JOINT LIMITS',20,MUTED,True)
        if idx and not updating:text(d,(45,222),'0.70x speed',20,MUTED)
        # Compressed actual forward activations. Only recorded batch updates pulse.
        f=forwards[idx];hidden=f['policy_hidden_activations'];arrays=[np.abs(f['observation']),np.mean(np.abs(hidden[2]).reshape(8,8),axis=1),np.mean(np.abs(hidden[4]).reshape(8,8),axis=1),np.array([np.mean(abs(np.array(f['gaussian_mean'])[a:b]))for a,b in [(0,37),(37,52),(52,58)]])]
        xs=[605,710,815,920];ys=[np.linspace(165,300,len(a))for a in arrays]
        for k in range(3):
            for y0 in ys[k]:
                for y1 in ys[k+1]:d.line((xs[k],y0,xs[k+1],y1),fill=(48,93,107)if updating else(31,50,66),width=1)
        for k,a in enumerate(arrays):
            for j,v in enumerate(a):
                x,y=xs[k],ys[k][j];col=ACC if updating else tuple(round(a+(b-a)*float(np.clip(v,0,1)))for a,b in zip((49,76,93),(129,209,194)));d.ellipse((x-6,y-6,x+6,y+6),fill=col)
        for x,s in zip(xs,['7','64','64','58']):text(d,(x-9,322),s,16,MUTED)
        ppm=1080/cam['ortho_width_m'];zero=540-cam['camera_center_forward_m']*ppm;ground=360+525+cam['camera_center_z_m']*ppm-190
        d.line((35,ground,1045,ground),fill=(102,124,142),width=2)
        for meter in np.arange(0,3.51,.5):
            x=zero+meter*ppm
            if 20<x<1060:d.line((x,ground-5,x,ground+12),fill=MUTED,width=2);text(d,(x-15,ground+21),f'{meter:.1f}',18,MUTED)
        d.line((zero,ground-30,zero,ground+15),fill=FG,width=3)
        contact=t>=r['touchdown_s'];fell=r['first_body_floor_contact_s']is not None and t>=r['first_body_floor_contact_s']
        if contact:
            x=zero+score['distance_m']*ppm;d.line((zero,ground,x,ground),fill=ACC,width=4)
            mask=(z['times']>=r['takeoff_s'])&(z['times']<=r['touchdown_s']);points=[]
            for v in z['com'][mask]:points.append((540+(-v[1]+cam['initial_foremost_sole_y_m']-cam['camera_center_forward_m'])*ppm,360+525-(v[2]-cam['camera_center_z_m'])*ppm-190))
            if len(points)>1:d.line(points,fill=ACC,width=2)
        d.line((45,1240,1035,1240),fill=(48,68,86))
        text(d,(45,1270),'FIRST CONTACT',19,MUTED,True);text(d,(45,1308),f"{score['distance_m']:.2f} m"if contact else'--',52,ACC,True)
        text(d,(630,1270),'RECOVERY',19,MUTED,True);text(d,(630,1313),'FALL'if fell else'IN PROGRESS',36,(244,143,103)if fell else FG,True)
        if updating:
            text(d,(45,1410),'SCORE  >  PPO  >  NEXT TRY',27,ACC,True)
        elif contact:
            text(d,(45,1410),f"DISTANCE + RECOVERY - FALL = {score['reward']:.2f}",24,MUTED,True)
        else:text(d,(45,1410),'TRY  >  SCORE  >  UPDATE',25,MUTED,True)
        text(d,(45,1495),'MODEL PILOT / 178 CM / 87 KG',17,MUTED)
        assert np.all(np.asarray(im)[1536:]==BG)
        return im
    draw(rows[0]['r']['first_body_floor_contact_s']+.22).save(OUT/'fall.png');draw(duration-.1).save(OUT/'next.png')
    writer=imageio_ffmpeg.write_frames(str(OUT/'distance_stability_learning.mp4'),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None)
    try:
        for k in range(round(duration*30)):writer.send(np.asarray(draw(k/30)))
    finally:writer.close()
    atomic(OUT/'evidence.json',dict(training_summary=str(RUN/'pilot_summary.json'),episode_sources=[str(row['folder'])for row in rows],frames=round(duration*30),fps=30,duration_s=round(duration*30)/30,unmodified_native_poses=True,actual_falls_included=True,post_contact_freeze_disguised_as_stability=False,actual_policy_updates=2,training_episodes=32,diagram_actual_architecture=[7,64,64,58],diagram_is_compressed=True,initial_score=scores[0],next_score=scores[1],reward_weights_are_design_choices=True,converged_or_stable_solution_claimed=False,bottom_reserved_px=384))
    comparisons=read(RUN/'comparison.json');table=''
    for title,value in [('Stable reference',comparisons['stable_reference']),('Initial attempt',scores[0]),('Next policy mean',scores[1])]:
        table+=f"<tr><td>{title}</td><td>{value['distance_m']:.3f} m</td><td>{value['stability']:.5f}</td><td>{'Fall'if value['fell']else'No fall'}</td><td>{value['reward']:.3f}</td></tr>"
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Distance + control</title><style>:root{color-scheme:dark}body{background:#0d1726;color:#edf3fb;font:16px/1.7 system-ui;max-width:1000px;margin:30px auto;padding:0 22px}h1{font-size:28px}h2{font-size:21px}video{max-width:100%;max-height:85vh;display:block;margin:20px auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #2c4054}a{color:#79d9c8}.note{border-left:3px solid #ef9558;padding-left:18px}small{color:#a6bcd1}</style><h1>넘어진 시도도 학습에 사용</h1><p>첫 접촉에서 끊지 않고, 골반이 지면에 닿은 뒤까지 실제 강체 접촉 계산을 이어갑니다. 2.88m 후보의 첫 접촉 거리는 기존 계산과 정확히 일치했습니다. 영상의 두 번째 동작은 실제 PPO 갱신 후 평균 정책을 다시 계산한 결과입니다.</p><video controls playsinline src="distance_stability_learning.mp4" poster="fall.png"></video><p class="note"><b>32회 시도, 실제 PPO 배치 갱신 2회.</b> 아직 안정 착지를 찾은 결과는 아닙니다. 실패를 숨기지 않고 다음 탐색에 전달하는 연결을 검증했습니다. 58개 출력(자세·타이밍 43개, 등가 관절 토크 곡선 15개)과 정책 표준편차를 함께 학습합니다.</p><h2>거리 이득과 착지 실패를 각각 보존</h2><table><tr><th>Case</th><th>First contact</th><th>Recovery S</th><th>Outcome</th><th>Reward</th></tr>TABLE</table><p><b>R = D + 0.8S − 0.8·fall</b>. D는 첫 발 접촉 때의 뒤꿈치 투영 거리(m), S는 0–1 회복 점수입니다. S는 지면에 몸이 닿기 전 버틴 시간, 마지막 구간의 COM 높이·속도·각운동량으로 계산합니다. 이후 넘어져도 D는 사라지지 않습니다. 도약 전후 역학 검사를 통과하지 못한 수치적 실패만 별도 -3으로 처리합니다.</p><p>0.8 가중치는 이번 탐색의 설계값입니다. 사람의 착지 성공 확률이나 공인 멀리뛰기 기록으로 해석하지 않습니다. 현재 실패 후보들은 S가 매우 작아 안정성 차이가 약하게 전달됩니다. 안정적인 영역으로 이동하려면 탐색 범위·초기 분포 및 회복 점수의 민감도를 추가 조정해야 합니다. 짧은 실험만으로 최적화 완료를 주장하지 않습니다.</p><p>같은 힘 한계 아래에서 약 2m의 기존 안정 기준은 더 높은 총보상을 받습니다. 거리가 큰 실패 후보도 모든 동작과 점수 성분을 남기고, 거리·회복 점수의 Pareto 후보를 별도 보존했습니다.</p><h2>계산·표현 범위</h2><p>추가한 접촉은 질량이 없는 머리·몸통·골반 구형 근사체입니다. 충돌 후 표시를 위한 단순 강체 접촉이며 연부조직이나 손·무릎을 이용한 회복, 발가락 제어와 회복 스텝은 아직 없습니다. 발 접촉과 내부 관절 회전력으로 움직이며 비행 COM 궤적을 지정하지 않습니다. 인체 메시와 근육은 표시용 중립색이고, 개별 근육 장력은 이번 제어 모델에서 계산하지 않습니다.</p><p><a href="evidence.json">영상과 학습 근거</a> · <a href="pilot_summary.json">실제 학습 결과</a> · <a href="updates.json">갱신별 가중치 변화</a> · <a href="pareto.json">거리·회복 후보 보존</a></p><small>178cm / 87kg, sagittal rigid-body model, MuJoCo, net joint-torque limits. These are virtual comparison conditions, not personalized human strength.</small></html>'''.replace('TABLE',table)
    html=html.replace('src="distance_stability_learning.mp4"','id="review" src="distance_stability_learning.mp4"')
    html=html.replace('</html>','<script>if(location.protocol!=="file:"){document.getElementById("review").src="/distance_stability.mp4";document.getElementById("review").poster="/distance_stability.png";document.querySelectorAll("a").forEach(a=>{a.href="/distance_stability/"+a.getAttribute("href")})}</script></html>')
    (OUT/'index.html').write_text(html,encoding='utf-8')
    import shutil
    for name in ['pilot_summary.json','updates.json','pareto.json','audit.json']:
        if(RUN/name).exists():shutil.copyfile(RUN/name,OUT/name)
    print('DISTANCE_STABILITY_VIDEO_READY',OUT/'distance_stability_learning.mp4',flush=True)

if __name__=='__main__':main()
