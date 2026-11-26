"""Append credits to delivered videos, retaining originals and all physics."""
from pathlib import Path
import json,subprocess
import numpy as np
import imageio_ffmpeg
from video_credits import make_card,FPS

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'


def main():
    report=json.loads((OUT/'comparison_results.json').read_text())
    for key,media in report['media'].items():
        if media.get('credits'):continue
        _,credit=make_card(key);original=OUT/media['file'];path=OUT/(key+'_tension_credited.mp4')
        if original==path:raise RuntimeError('Credit input must be the retained original video')
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(original),
            '-loop','1','-framerate',str(FPS),'-i',str(OUT/credit['file']),
            '-filter_complex',f"[0:v]setpts=PTS-STARTPTS,setsar=1[a];[1:v]trim=duration={credit['duration_s']},setpts=PTS-STARTPTS,setsar=1[b];[a][b]concat=n=2:v=1:a=0[v]",
            '-map','[v]','-r',str(FPS),'-an','-c:v','libx264','-threads','4','-crf','18','-preset','medium',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        subprocess.run(command,check=True)
        frames,duration=imageio_ffmpeg.count_frames_and_secs(str(path));expected=media['frames']+credit['frames']
        if frames!=expected:raise RuntimeError(('Credit frame count',key,frames,expected))
        media.update(source_motion_video=original.name,source_motion_video_frames=media['frames'],
                     file=path.name,frames=frames,duration_s=duration,credits=credit)
        print('CREDITED_VIDEO',key,frames,flush=True)
    (OUT/'comparison_results.json').write_text(json.dumps(report,indent=2))
    from package_jump_squat_power import write_page
    write_page(report)


if __name__=='__main__':main()
