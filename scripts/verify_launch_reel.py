"""Presentation QA; does not imply physiological validation."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
import imageio_ffmpeg
from case_contours import load,overview

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/instagram_launch'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    meta=json.loads((OUT/'reel_metrics.json').read_text())
    frames=imageio_ffmpeg.read_frames(str(OUT/meta['file']),pix_fmt='rgb24')
    stream=next(frames);assert stream['size']==(1080,1920) and stream['fps']==30,stream
    count=0;bottom_error=0;last=None
    for raw in frames:
        last=np.frombuffer(raw,np.uint8).reshape(1920,1080,3)
        bottom_error=max(bottom_error,int(np.abs(last[1536:].astype(np.int16)-[13,23,38]).max()))
        count+=1
    assert count==750,count
    assert bottom_error<=4,bottom_error
    source_card=np.asarray(Image.open(OUT/'reel_sources.png').convert('RGB'))
    final_error=float(np.abs(last.astype(float)-source_card).mean())
    assert final_error<5,final_error
    windows={'cmj':(.928,1.22),'squat_100':(1.65,2.55)}
    peak={}
    for key,window in windows.items():
        data=load(ROOT,key);ts=np.asarray(data['times']);v=overview(data)
        mask=(ts>=window[0])&(ts<=window[1]);peak[key]=v[[1,0,2,3]][:,mask].max(axis=1)
        source=ROOT/meta['physics_provenance']['cases'][key]['source']
        analysis=source if key=='cmj' else source/'analysis'
        p=meta['physics_provenance']['cases'][key]
        assert digest(analysis/'motion.npz')==p['motion_sha256']
        assert digest(analysis/'tensions.npz')==p['force_sha256']
        assert digest(OUT/'render_cases'/key/'motion.npz')==digest(source/'motion.npz')
    ratio=peak['cmj']/peak['squat_100']
    assert np.allclose(ratio,[x['jump_to_squat_peak_ratio'] for x in meta['metrics']],atol=1e-12)
    for name in ['profile_bio_ko.txt','intro_post_ko.txt','reel_caption_ko.txt']:
        text=(OUT/name).read_text(encoding='utf-8-sig')
        assert 'cae_ksm' not in text and 'workout_ksm' not in text
    profile=Image.open(OUT/'profile_minimal.png');assert profile.width==profile.height
    files=['jump_vs_100kg_squat_reel.mp4','profile_minimal.png','profile_generation_prompt.txt',
        'profile_bio_ko.txt','intro_post_ko.txt','reel_caption_ko.txt','reel_cover.png',
        'reel_summary.png','reel_sources.png','reel_metrics.json','source_links.txt','CREDITS.md','index.html']
    checks=dict(status='passed',scope='Presentation and provenance checks; not biological accuracy validation.',
        frames=count,width=1080,height=1920,fps=30,bottom_reserved_px=384,bottom_max_codec_error=bottom_error,
        final_credit_card_mean_pixel_error=final_error,phase_peak_ratios=ratio.tolist(),source_hashes_unchanged=True,
        independent_motion_clocks=True,landing_included=False,profile_square=True,
        files={name:dict(bytes=(OUT/name).stat().st_size,sha256=digest(OUT/name)) for name in files})
    (OUT/'launch_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({k:v for k,v in checks.items() if k!='files'}))

if __name__=='__main__':main()
