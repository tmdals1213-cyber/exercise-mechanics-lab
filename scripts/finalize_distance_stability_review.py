"""Verify the encoded clip and publish an explicit current-state override."""
from pathlib import Path
import json,shutil
import numpy as np,imageio_ffmpeg
from jump_monitor_data import ROOT,read,atomic
from build_motion_study_reel import BG

def main():
    out=ROOT/'deliverables/jump_distance_stability_review';run=ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1'
    evidence=read(out/'evidence.json');reader=imageio_ffmpeg.read_frames(str(out/'distance_stability_learning.mp4'),pix_fmt='rgb24');meta=next(reader);count=0;error=0
    from PIL import Image
    w,h=meta['size'];assert(w,h)==(1080,1920)
    for raw in reader:
        frame=np.frombuffer(raw,np.uint8).reshape(h,w,3);error=max(error,int(abs(frame[1536:].astype(int)-np.array(BG)).max()));count+=1
        if count==round(3.7*30):Image.fromarray(frame).save(out/'encoded_update.png')
    assert count==evidence['frames']and error<=3
    audit=read(run/'audit.json');assert audit['status']=='passed'
    shutil.copyfile(run/'audit.json',out/'audit.json')
    check=dict(status='passed',encoded_frames=count,resolution=[w,h],fps=meta['fps'],duration_s=count/meta['fps'],reserved_footer_max_RGB_error=error,actual_fall_shown=True,physics_audit=audit,video=str(out/'distance_stability_learning.mp4'))
    atomic(out/'delivery_check.json',check)
    for target in [ROOT/'deliverables/jump_ai_single_view_concept/index.html',ROOT/'deliverables/standing_long_jump_effort_seed/index.html']:
        page=target.read_text(encoding='utf-8')
        marker='<p style="border-left:3px solid #ef9558;padding-left:18px"><b>새 결과:</b> 실패 후보도 거리·회복 점수·낙상 감점과 함께 다음 학습에 사용했습니다. <a href="/distance_stability.html">낙상까지 이어지는 영상과 실제 PPO 32회 연결 검사</a></p>'
        if '실패 후보도 거리·회복 점수·낙상 감점과 함께'not in page:page=page.replace('</h1>','</h1>'+marker,1)
        target.write_text(page,encoding='utf-8')
    file=ROOT/'docs/standing_long_jump_working_state.json';state=read(file,{})
    state['LATEST_DISTANCE_STABILITY_LEARNING_2026_11_25']=dict(user_direction='Keep failed longer jumps; show complete falls; distance increase and stability reduction both feed the next RL policy.',current_status='Bounded actual PPO pilot completed, not converged.',video=str(out/'distance_stability_learning.mp4'),report=str(out/'index.html'),url='http://127.0.0.1:8765/distance_stability.html',training_episodes=32,policy_updates=2,trainable_outputs=58,policy_initialization='Long-distance falling candidate, all equivalent effort and 6 recovery variables included.',reward='D + 0.8*S - 0.8*fall; D retained on valid falling jumps, invalid precontact dynamics receive -3.',reward_weights_are_design_choices=True,initial_first_contact_m=2.8817403416746563,next_mean_first_contact_m=2.8901932196202162,both_fall=True,stable_reference_reward=2.6325321453837955,first_body_contact_s=2.3175,physics_audit=str(run/'audit.json'),contacts='Existing rigid feet plus massless head/torso/pelvis floor proxies. No toes, hand/knee recovery contacts or stepping.',limitation='All32 local samples fell; multiplicative recovery score near zero makes stability differences weak. Before longer training, use an additive/shaped recovery measure and a distribution including the stable reference. Do not claim a stable solution.',no_current_training_process=True,individual_muscle_force_solver=False,scripts=['scripts/jump_recovery_learning.py','scripts/jump_distance_stability.py','scripts/render_distance_stability_review.py'],next_action='Improve continuous recovery reward sensitivity and initial exploration coverage, preserving the existing actual pilot evidence; then resume actual PPO and fine-step qualification.')
    atomic(file,state)
    print('DISTANCE_STABILITY_DELIVERY_VERIFIED',count,error,flush=True)

if __name__=='__main__':main()
