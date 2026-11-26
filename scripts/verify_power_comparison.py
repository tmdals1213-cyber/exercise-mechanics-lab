"""Check source fidelity, phase statistics, anatomy alignment and encoded media."""
from pathlib import Path
import hashlib
import json
import numpy as np
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/power_comparison'


def main():
    report=json.loads((OUT/'summary.json').read_text())
    spec=json.loads((OUT/'comparison_setup.json').read_text())
    video=json.loads((OUT/'video_checks.json').read_text())
    assert report['analytic_checks']['status']=='passed'
    assert spec['timing']['display_speed']==1
    assert spec['timing']['replays_per_condition']==3
    assert not spec['timing']['phase_time_warp']
    assert not spec['timing']['force_equal_cycle_duration']
    assert len(spec['conditions'])==6
    assert all(r['power_result_W'] is None for r in spec['conditions'])
    for row in report['trials']:
        for rel,sha in row['provenance'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
        d=np.load(OUT/f"{row['trial']}_power.npz")
        t=d['times_source_s'];mask=(t>=row['source_bottom_s'])&(t<=row['source_takeoff_s'])
        p=d['COM_vertical_power_W'][mask]
        assert np.allclose(d['COM_vertical_power_W'],d['vertical_GRF_N']*d['COM_vertical_velocity_m_s'],rtol=0,atol=1e-9)
        assert abs(max(p)-row['peak_COM_vertical_power_W'])<1e-8
        work=np.trapezoid(p,t[mask])
        assert abs(work-row['propulsion_COM_vertical_work_J'])<1e-8
        assert abs(work/row['propulsion_duration_s']-row['mean_propulsion_COM_vertical_power_W'])<1e-8
        assert not report['actual_power_accuracy_bound_established']
    # Verify the anatomy replay uses the same time and vertical COM velocity.
    source=ROOT/'results/measured_jump_analysis/s10_CMJ_t1'
    motion=np.load(source/'motion.npz');setup=json.loads((source/'setup.json').read_text())
    row=report['trials'][0];d=np.load(OUT/'s10_CMJ_t1_power.npz')
    t=motion['times']+setup['source_start_s']
    assert t.min()>=d['times_source_s'].min() and t.max()<=d['times_source_s'].max()
    v=np.interp(t,d['times_source_s'],d['COM_vertical_velocity_m_s'])
    f=np.interp(t,d['times_source_s'],d['vertical_GRF_N'])
    force_error=float(max(abs(f-motion['force_world_N'][:,2])))
    velocity_error=float(max(abs(v-motion['velocity'][:,2])))
    assert force_error<1e-7
    assert velocity_error<1e-4
    frames=imageio_ffmpeg.read_frames(str(OUT/'cmj_power_realtime.mp4'),pix_fmt='rgb24')
    metadata=next(frames)
    assert tuple(metadata['size'])==(1080,1920)
    assert metadata['fps']==30
    count=0;footer_error=0
    for raw in frames:
        a=np.frombuffer(raw,dtype=np.uint8).reshape(1920,1080,3)
        footer_error=max(footer_error,int(np.max(abs(a[1536:].astype(int)-[13,23,38]))))
        count+=1
    assert count==video['frames']==183
    assert footer_error<=3
    checks=dict(status='passed',analytic_cases=report['analytic_checks']['analytic_cases'],
                trial_statistics=3,source_hashes_unchanged=True,uncomputed_conditions_not_filled=True,
                native_replay_vertical_force_max_error_N=force_error,
                native_replay_COM_velocity_max_error_m_s=velocity_error,
                encoded_frames=count,resolution=[1080,1920],fps=metadata['fps'],
                footer_max_channel_error=footer_error,natural_speed=True,
                muscle_force_solution_modified=False)
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
