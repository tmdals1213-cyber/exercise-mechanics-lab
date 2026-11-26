"""Package the audited power baseline at natural speed using existing anatomy.

The anatomical frames and their conditional muscle/joint colours are retained.
This does not synthesize an unmeasured box jump, squat or landing.
"""
from pathlib import Path
import json

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/power_comparison'
VIS=ROOT/'results/measured_jump_visualization'
BG=(13,23,38); FG=(237,243,251); MUTED=(166,188,209); GREEN=(140,226,213)


def font(n,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',n)


def label(draw,xy,value,size=23,color=FG,bold=False):
    draw.text(xy,str(value),font=font(size,bold),fill=color)


def draw_frame(i,replay,meta,ts,power_t,power,force,velocity,row,setup):
    im=Image.open(VIS/f'portrait_frames/frame_{i:04d}.png').convert('RGB')
    draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,1080,220),fill=BG)
    draw.rectangle((0,1030,1080,1920),fill=BG)
    label(draw,(54,30),'COUNTERMOVEMENT JUMP',40,bold=True)
    label(draw,(54,88),'180 cm / 88 kg  |  Measured motion',25)
    label(draw,(54,126),'Conditional muscle tension / joint reaction',21,MUTED)
    label(draw,(762,93),f'1x  |  Replay {replay}/3',21,GREEN)
    source_t=float(ts[i]+setup['source_start_s'])
    bottom=row['source_bottom_s'];to=row['source_takeoff_s']
    phase='COUNTERMOVEMENT' if source_t<bottom else 'PROPULSION' if source_t<to else 'FLIGHT'
    label(draw,(54,177),phase,26,GREEN,True)
    label(draw,(762,177),f'{ts[i]:.2f} s',25)
    p=float(np.interp(source_t,power_t,power))
    f=float(np.interp(source_t,power_t,force))
    v=float(np.interp(source_t,power_t,velocity))
    label(draw,(54,1055),'VERTICAL COM POWER',25,bold=True)
    label(draw,(737,1038),f'{p/1000:+.2f} kW',40,GREEN,True)
    label(draw,(54,1094),f'Ground force {f:,.0f} N   /   COM velocity {v:+.2f} m/s',22,MUTED)
    x0,x1,y0,y1=104,1000,1150,1360
    visible=(power_t>=setup['source_start_s'])&(power_t<=setup['source_start_s']+ts[-1])
    xs=x0+(power_t[visible]-setup['source_start_s'])/ts[-1]*(x1-x0)
    lower,upper=-2.,4.
    def yy(kw):return y1-(kw-lower)/(upper-lower)*(y1-y0)
    bx=x0+(bottom-setup['source_start_s'])/ts[-1]*(x1-x0)
    ex=x0+(to-setup['source_start_s'])/ts[-1]*(x1-x0)
    draw.rectangle((round(bx),y0,round(ex),y1),fill=(34,49,56))
    for n in [-2,0,2,4]:
        y=round(yy(n));draw.line((x0,y,x1,y),fill=(57,76,94),width=1)
        label(draw,(55,y-12),str(n),18,MUTED)
    points=list(zip(xs,yy(power[visible]/1000)))
    draw.line(points,fill=GREEN,width=4)
    cursor=x0+ts[i]/ts[-1]*(x1-x0)
    draw.line((round(cursor),y0,round(cursor),y1),fill=FG,width=3)
    cy=yy(p/1000);draw.ellipse((cursor-6,cy-6,cursor+6,cy+6),fill=FG)
    label(draw,(56,1127),'kW',18,MUTED)
    label(draw,(x0,1369),'0.0 s',18,MUTED)
    label(draw,(x1-57,1369),f'{ts[-1]:.1f} s',18,MUTED)
    for x,title,value in [(54,'PEAK',f"{row['peak_COM_vertical_power_W']/1000:.2f} kW"),
                          (394,'PROPULSION MEAN',f"{row['mean_propulsion_COM_vertical_power_W']/1000:.2f} kW"),
                          (766,'PROPULSION',f"{row['propulsion_duration_s']:.3f} s")]:
        label(draw,(x,1422),title,19,MUTED)
        label(draw,(x,1454),value,32,bold=True)
    label(draw,(54,1505),'Same measured repetition replayed 3 times. Landing excluded.',18,MUTED)
    assert np.all(np.asarray(im)[1536:]==BG)
    return im


def make_video(report):
    meta=json.loads((VIS/'data.json').read_text())
    setup=json.loads((ROOT/'results/measured_jump_analysis/s10_CMJ_t1/setup.json').read_text())
    power=np.load(OUT/'s10_CMJ_t1_power.npz'); row=report['trials'][0]
    ts=np.asarray(meta['times']); fps=30; duration=float(ts[-1]-ts[0])
    sample_times=np.arange(round(duration*fps)+1)/fps
    frame_ids=np.array([int(np.argmin(abs(ts-t))) for t in sample_times])
    timing_error=float(np.max(abs(ts[frame_ids]-sample_times)))
    assert timing_error<.012
    writer=imageio_ffmpeg.write_frames(str(OUT/'cmj_power_realtime.mp4'),(1080,1920),fps=fps,
        codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,
        output_params=['-crf','18','-preset','medium','-movflags','+faststart'])
    writer.send(None);count=0
    try:
        for replay in range(1,4):
            for i in frame_ids:
                im=draw_frame(i,replay,meta,ts,power['times_source_s'],power['COM_vertical_power_W'],
                              power['vertical_GRF_N'],power['COM_vertical_velocity_m_s'],row,setup)
                writer.send(np.asarray(im));count+=1
                if replay==1 and i==meta['poster_index']:
                    im.save(OUT/'poster.png');im.crop((0,0,1080,1536)).save(OUT/'preview.png')
            # Explicit end hold separates replays without changing motion speed.
            for _ in range(12):writer.send(np.asarray(im));count+=1
            print('POWER_REALTIME_REPLAY',replay,flush=True)
    finally:writer.close()
    encoded,seconds=imageio_ffmpeg.count_frames_and_secs(str(OUT/'cmj_power_realtime.mp4'))
    assert encoded==count
    checks=dict(status='passed',display_speed=1.,fps=fps,replays=3,physics_repetitions=1,
                source_motion_duration_s=duration,source_to_display_max_time_error_s=timing_error,
                frames=encoded,duration_s=seconds,end_hold_per_replay_s=.4,
                bottom_reserved_px=384,landing_included=False,computed_power_sample_rate_Hz=250,
                note='Motion and statistics use original physical seconds; replay holds are excluded from means and work.')
    (OUT/'video_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


def make_page(report):
    rows=report['trials']
    cells=''.join(f"<tr><td>Trial {i+1}</td><td>{r['peak_COM_vertical_power_W']/1000:.2f}</td><td>{r['mean_propulsion_COM_vertical_power_W']/1000:.2f}</td><td>{r['propulsion_duration_s']:.3f}</td><td>{r['propulsion_COM_vertical_work_J']:.0f}</td><td>{100*r['force_velocity_peak_relative_difference']:.1f}%</td></tr>" for i,r in enumerate(rows))
    page=(ROOT/'scripts/power_comparison_viewer.html').read_text(encoding='utf-8')
    page=page.replace('<!--TRIAL_ROWS-->',cells)
    (OUT/'index.html').write_text(page,encoding='utf-8')
    (OUT/'CREDITS.md').write_text((VIS/'CREDITS.md').read_text(encoding='utf-8')
        +'\n\nMeasured CMJ: Calisti, Mohr & Federolf (2025), https://doi.org/10.6084/m9.figshare.28890545, CC BY 4.0.\nCOM power analysis and natural-speed composition are derived outputs.\n',encoding='utf-8')


def main():
    report=json.loads((OUT/'summary.json').read_text())
    make_video(report);make_page(report)
    print('POWER_BASELINE_PACKAGED',OUT,flush=True)


if __name__=='__main__':main()
