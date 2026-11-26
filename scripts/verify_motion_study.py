"""Check revised video clocks, anatomy mapping and force provenance."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
import imageio_ffmpeg
from launch_sources import analysis_path,render_root

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/instagram_launch'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    meta=json.loads((OUT/'study_metrics.json').read_text());assert meta['repeats_per_motion']==3
    assert meta['new_jump_heights_solved'] is True and meta['landing_included'] is False
    original=json.loads((OUT/'physics_provenance.json').read_text())
    from prepare_tension_exposure import integrate
    exposure=meta['tension_exposure']
    for key,c in meta['cases'].items():
        p=analysis_path(ROOT,key)
        z=np.load(p/'tensions.npz');v=json.loads((p/'visualization/data.json').read_text())
        ids=[i for i,n in enumerate(v['names']) if n in ['recfem_r','recfem_l']];rf=z['force_N'][:,ids].sum(1)/2
        assert len(ids)==2 and abs(rf.max()-c['rectus_peak_N'])<1e-10
        assert sha(p/'tensions.npz')==c['force_sha256']==original['cases'][key]['force_sha256']
        assert sha(p/'motion.npz')==c['motion_sha256']==original['cases'][key]['motion_sha256']
        trace=np.loadtxt(OUT/f'{key}_study_forces.csv',delimiter=',',skiprows=1)
        assert np.allclose(trace[:,0],z['times']) and np.allclose(trace[:,1],rf)
        assert c['playback_rate']<.5 and abs(c['duration_s']/c['playback_rate']-6)<1e-10
        e=exposure['cases'][key]
        actual=integrate(z['times'],z['force_N'],e['start_s'],e['end_s'])
        assert np.allclose(actual,e['element_integral_Ns'],atol=1e-8)
        assert np.allclose(np.array(e['group_mean_tension_per_side_N'])*e['duration_s'],e['group_integral_mean_per_side_Ns'],atol=1e-8)
        mapping=json.loads((render_root(ROOT)/key/'study_muscle_mapping.json').read_text())
        assert 'absolute linear' in mapping['normalization'] and mapping['common_upper_N']>0
        rectus=[m for m in mapping['meshes'] if 'rectus femoris' in m['mesh'].lower()]
        assert len(rectus)>=2 and all(len(m['elements'])==1 and m['elements'][0].startswith('recfem_') for m in rectus)
        assert any('latissimus' in m['mesh'].lower() for m in mapping['meshes'])
        assert any('trapezius' in m['mesh'].lower() for m in mapping['meshes'])
    stream=imageio_ffmpeg.read_frames(str(OUT/meta['file']),pix_fmt='rgb24');header=next(stream)
    assert header['size']==(1080,1920) and header['fps']==30
    count=0;bottom=0;samples={};last=None
    for raw in stream:
        a=np.frombuffer(raw,np.uint8).reshape(1920,1080,3)
        bottom=max(bottom,int(np.abs(a[1536:].astype(np.int16)-[13,23,38]).max()))
        if count in [30,90,210,270,390,450]:samples[count]=a[190:1405].copy()
        last=a;count+=1
    assert count==26*30 and bottom<=4,(count,bottom)
    # Repeat anatomy and curve states; only the replay counter changes above y190.
    replay_errors=[float(np.abs(samples[x].astype(float)-samples[x+180]).mean()) for x in [30,90,210,270]]
    assert max(replay_errors)<2,replay_errors
    card=np.array(Image.open(OUT/'study_sources.png').convert('RGB'))
    end_error=float(np.abs(last.astype(float)-card).mean());assert end_error<5
    for file in ['profile_bio_en.txt','intro_post_en.txt','reel_caption_en.txt']:
        text=(OUT/file).read_text(encoding='utf-8');assert text.isascii() and 'cae_ksm' not in text and 'workout_ksm' not in text
    assert len((OUT/'profile_bio_en.txt').read_text().strip())<=150
    assert len((OUT/'reel_caption_en.txt').read_text().strip())<=2200
    assert .42<meta['standing_to_apex_COM_rise_m']<.44 and .297<meta['jump_height_m']<.303
    mechanics=json.loads((analysis_path(ROOT,'cmj')/'verification.json').read_text())
    assert all(mechanics['gates'].values()) and mechanics['exact_sticking_contact'] is False
    assert mechanics['checks']['sample_rate_peak_GRF_relative_difference']<.01
    integral_checks=dict(constant=integrate(np.array([0.,.3,1.]),np.full((3,1),7.),.2,.8)[0],
        linear=integrate(np.array([0.,.3,1.]),np.array([[0.],[.3],[1.]]),.2,.8)[0])
    assert abs(integral_checks['constant']-4.2)<1e-12 and abs(integral_checks['linear']-.3)<1e-12
    previous=json.loads((OUT/'launch_checks.json').read_text())
    assert sha(OUT/'profile_minimal.png')==previous['files']['profile_minimal.png']['sha256']
    report=dict(status='passed',scope='Video, exact muscle extraction and selected-source provenance; not human prediction validation.',
        frames=count,fps=30,size=[1080,1920],source_force_hashes_unchanged=True,source_motion_hashes_unchanged=True,
        prescribed_30cm_scenario=True,common_absolute_muscle_color_scale=True,
        exact_sticking_contact_claimed=False,rectus_femoris_extracted_separately=True,anatomy_latissimus_and_trapezius_mapped=True,
        repeats_per_motion=3,replay_state_mean_pixel_errors=replay_errors,bottom_codec_max_error=bottom,
        integral_analytic_checks=integral_checks,force_time_means_times_duration=True,display_pose_not_reanalyzed=True,
        final_credit_card_mean_pixel_error=end_error,profile_image_sha256=sha(OUT/'profile_minimal.png'),
        assets={file:sha(OUT/file) for file in [meta['file'],'study_metrics.json','profile_bio_en.txt','intro_post_en.txt','reel_caption_en.txt','index.html']})
    (OUT/'study_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
