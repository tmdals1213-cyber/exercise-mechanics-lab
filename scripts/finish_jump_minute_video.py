"""Recompose the approved study under60s, preserving all physical results."""
from pathlib import Path
import shutil,subprocess,sys,hashlib,time
import numpy as np,imageio_ffmpeg
from jump_monitor_data import ROOT,read,atomic
from compose_jump_rl_draft_v4 import BG
OUT=ROOT/'deliverables/jump_rl_study_draft_v5'

def main():
    original=OUT/'standing_long_jump_rl_final.mp4';backup=OUT/'standing_long_jump_rl_full_length.mp4';short=OUT/'standing_long_jump_rl_60s.mp4';old=read(OUT/'delivery_check.json');delivery=read(OUT/'draft_delivery.json');assert old['passed']and delivery['duration_s']>60
    if not backup.exists():shutil.copyfile(original,backup)
    for name in ['draft_delivery.json','delivery_check.json','timeline_v5.json']:
        dest=OUT/('full_length_'+name)
        if not dest.exists():shutil.copyfile(OUT/name,dest)
    archived=read(OUT/'full_length_draft_delivery.json');archived['video']=str(backup);atomic(OUT/'full_length_draft_delivery.json',archived)
    atomic(OUT/'minute_video_status.json',dict(stage='composing',heartbeat=time.time(),target_s=59.8))
    subprocess.run([sys.executable,str(ROOT/'scripts/compose_jump_rl_draft_v5.py'),'--duration','59.8','--output',str(short)],cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    atomic(OUT/'minute_video_status.json',dict(stage='verifying',heartbeat=time.time()))
    stream=imageio_ffmpeg.read_frames(str(short),pix_fmt='rgb24');meta=next(stream);assert tuple(meta['size'])==(1080,1920)and meta['fps']==30;count=0;footer=0
    for raw in stream:
        frame=np.frombuffer(raw,np.uint8).reshape(1920,1080,3);footer=max(footer,int(abs(frame[1536:].astype(int)-np.array(BG)).max()));count+=1
    assert count==1794 and footer<=4
    check=dict(old,frames=count,footer_error=footer,video_sha256=hashlib.sha256(short.read_bytes()).hexdigest(),duration_s=count/30,preserved_prefix_s=33.,source_full_length_video_sha256=old['video_sha256'],presentation_timing_changed_only=True,physical_simulation_unchanged=True)
    # All targets are the explicitly named deliverables in this project.
    for p in [original,backup,short]:assert p.resolve().parent==OUT.resolve()
    shutil.copyfile(short,original);short.unlink();delivery=read(OUT/'draft_delivery.json');delivery['video']=str(original);atomic(OUT/'draft_delivery.json',delivery);atomic(OUT/'delivery_check.json',check)
    example=ROOT/'github_release/examples/long_jump';shutil.copyfile(original,example/'standing_long_jump_rl_final.mp4');atomic(example/'delivery_check.json',check)
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_github_release.py')],cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    publication=read(ROOT/'deliverables/github_release/publication_state.json',{});archive=ROOT/'deliverables/github_release/exercise-mechanics-github.zip';publication.update(video=str(original),video_sha256=check['video_sha256'],video_duration_s=count/30,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),updated_at=time.time());atomic(ROOT/'deliverables/github_release/publication_state.json',publication)
    atomic(OUT/'minute_video_status.json',dict(stage='complete',heartbeat=time.time(),video=str(original),duration_s=count/30,frames=count,passed=True));print('MINUTE_VIDEO_READY',str(original),count/30,flush=True)

if __name__=='__main__':main()
