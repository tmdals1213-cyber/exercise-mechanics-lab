"""Natural-speed power comparison: measured CMJ and prescribed loaded squats."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_loaded_squat_power import contact_solver
from compose_case_contours import video

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'
BG=(13,23,38);FG=(237,243,251);MUTED=(166,188,209);GREEN=(140,226,213)


def font(n,b=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if b else 'C:/Windows/Fonts/arial.ttf',n)
def text(d,p,s,n=22,c=FG,b=False):d.text(p,str(s),font=font(n,b),fill=c)


def sensitivity(row,data):
    t=data['times'];phase=row['phase_times_s'];mask=(t>=phase['start_ascent'])&(t<=phase['end_ascent'])
    F=data['GRF_N'][mask];v=data['system_velocity_m_s'][mask];com=data['system_COM_m'][mask]
    pts=data['contact_positions_m'][mask];forces=data['contact_forces_N'][mask]
    original_M=np.cross(pts,forces).sum(1);mass=88+row['bar_mass_kg'];weight=mass*9.80665
    gravity_F=np.zeros_like(F);gravity_F[:,2]=weight;gravity_M=np.cross(com,gravity_F)
    allocate=contact_solver(pts[0],mass);results=[]
    for ratio in [.8,.9,1.,1.1,1.2]:
        force=gravity_F+(F-gravity_F)/ratio**2;moment=gravity_M+(original_M-gravity_M)/ratio**2
        feasible=True
        for i in range(0,len(force),3):
            try:allocate(np.r_[force[i],moment[i]])
            except RuntimeError:feasible=False;break
        p=force[:,2]*v[:,2]/ratio
        results.append(dict(ascent_duration_s=row['prescribed_timing_s']['ascent']*ratio,
                            time_ratio=ratio,contact_feasible=feasible,
                            peak_COM_power_W=float(p.max()) if feasible else None,
                            status='same path, gravity unchanged, inertial wrenches scaled by time squared' if feasible else 'contact wrench infeasible; not a valid planted-foot squat'))
    return results


def video_power_only(key,row,data):
    cmj=key=='cmj';fps=30
    if cmj:
        source=ROOT/'results/measured_jump_visualization';visual=json.loads((source/'data.json').read_text())
        source_t=np.asarray(visual['times']);T=float(source_t[-1]);setup=json.loads((ROOT/'results/measured_jump_analysis/s10_CMJ_t1/setup.json').read_text());offset=setup['source_start_s']
        times=data['times_source_s']-offset;power=data['COM_vertical_power_W'];F=data['vertical_GRF_N']
        bottom=row['source_bottom_s']-offset;up=bottom;end=row['source_takeoff_s']-offset
        peak=row['peak_COM_vertical_power_W'];mean=row['mean_propulsion_COM_vertical_power_W'];ascent=row['propulsion_duration_s']
    else:
        source=OUT/key;T=float(data['times'][-1]);source_t=np.arange(round(T*30)+1)/30
        times=data['times'];power=data['COM_power_W'];F=data['GRF_N'][:,2]
        bottom=row['phase_times_s']['bottom'];up=row['phase_times_s']['start_ascent'];end=row['phase_times_s']['end_ascent']
        peak=row['peak_COM_vertical_power_W'];mean=row['mean_ascent_COM_vertical_power_W'];ascent=row['prescribed_timing_s']['ascent']
    display_times=np.minimum(np.arange(round(T*fps)+1)/fps,T);cache=[]
    for k,t in enumerate(display_times):
        i=int(np.argmin(abs(source_t-t)))
        image_path=source/(f'anatomy/base/frame_{i:04d}.png' if cmj else f'render/frame_{i:04d}.png')
        body=Image.open(image_path).convert('RGBA')
        im=Image.new('RGB',(1080,1920),BG);im.paste(body,(0,220),body);d=ImageDraw.Draw(im)
        text(d,(54,30),'JUMP / SQUAT POWER',42,b=True)
        text(d,(54,92),'COUNTERMOVEMENT JUMP' if cmj else f"HIGH-BAR SQUAT / {row['bar_mass_kg']} KG",29,b=True)
        text(d,(54,138),'180 cm / 88 kg  |  '+('Measured reference' if cmj else 'Prescribed motion'),22,MUTED)
        phase=('COUNTERMOVEMENT' if t<up else 'PROPULSION' if t<end else 'FLIGHT') if cmj else ('READY' if t<.25 else 'DESCENT' if t<bottom else 'BOTTOM' if t<up else 'ASCENT' if t<end else 'LOCKOUT')
        text(d,(54,181),phase,24,GREEN,True);text(d,(820,181),f'{t:.2f} s',24)
        text(d,(185,1102),'FRONT OBLIQUE',18,MUTED);text(d,(813,1102),'POSTERIOR' if cmj else 'SIDE',18,MUTED)
        p=float(np.interp(t,times,power));f=float(np.interp(t,times,F))
        text(d,(54,1141),'VERTICAL COM POWER',24,b=True);text(d,(774,1127),f'{p/1000:+.2f} kW',34,GREEN,True)
        x0,x1,y0,y1=104,1000,1202,1360;low,high=-3.,4.
        Y=lambda p:y1-(p/1000-low)/(high-low)*(y1-y0)
        X=lambda s:x0+s/T*(x1-x0)
        d.rectangle((round(X(up)),y0,round(X(end)),y1),fill=(34,49,56))
        for value in [-3,0,2,4]:
            y=round(Y(value*1000));d.line((x0,y,x1,y),fill=(55,74,92));text(d,(55,y-10),value,18,MUTED)
        selected=(times>=0)&(times<=T)
        d.line([(X(float(tt)),Y(float(pp))) for tt,pp in zip(times[selected],power[selected])],fill=GREEN,width=4)
        x=X(t);d.line((round(x),y0,round(x),y1),fill=FG,width=3)
        text(d,(x0,1370),'0.0 s',17,MUTED);text(d,(x1-58,1370),f'{T:.1f} s',17,MUTED)
        for x,title,value in [(54,'PEAK',f'{peak/1000:.2f} kW'),(390,'POSITIVE-PHASE MEAN',f'{mean/1000:.2f} kW'),(790,'DURATION',f'{ascent:.3f} s')]:
            text(d,(x,1416),title,18,MUTED);text(d,(x,1448),value,32,b=True)
        text(d,(54,1505),f'Ground force {f:,.0f} N  |  1x speed  |  3 replays',18,MUTED)
        assert np.all(np.asarray(im)[1536:]==BG)
        cache.append(im)
    path=OUT/f'{key}_comparison.mp4';writer=imageio_ffmpeg.write_frames(str(path),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-movflags','+faststart']);writer.send(None);count=0
    try:
        for replay in range(3):
            for im in cache:writer.send(np.asarray(im));count+=1
            for _ in range(12):writer.send(np.asarray(cache[-1]));count+=1
    finally:writer.close()
    actual,secs=imageio_ffmpeg.count_frames_and_secs(str(path));assert actual==count
    peak_time=row['peak_power_source_time_s']-offset if cmj else float(times[np.argmax(np.where((times>=up)&(times<=end),power,-np.inf))])
    poster=cache[int(np.argmin(abs(display_times-peak_time)))];poster.save(OUT/f'{key}_comparison.png')
    return dict(file=path.name,frames=count,duration_s=secs,fps=30,display_speed=1,replays=3,independent_clock=True,bottom_reserved_px=384,landing_included=False if cmj else None)


def plots(cmj,records):
    colors=['#efb45b','#a3a1ff','#70b8f7'];fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    d=np.load(OUT/'s10_CMJ_t1_power.npz');tt=d['times_source_s']-cmj['source_bottom_s'];mask=(tt>=0)&(tt<=cmj['propulsion_duration_s']);ax[0,0].plot(tt[mask],d['COM_vertical_power_W'][mask]/1000,color='#169f8c',label='Measured CMJ / Trial 1',lw=2.5)
    labels=['CMJ'];peaks=[cmj['peak_COM_vertical_power_W']/1000];means=[cmj['mean_propulsion_COM_vertical_power_W']/1000]
    for color,(row,data) in zip(colors,records):
        phase=row['phase_times_s'];t=data['times']-phase['start_ascent'];mask=(t>=0)&(data['times']<=phase['end_ascent'])
        label=f"Squat {row['bar_mass_kg']} kg";labels.append(label);peaks.append(row['peak_COM_vertical_power_W']/1000);means.append(row['mean_ascent_COM_vertical_power_W']/1000)
        ax[0,0].plot(t[mask],data['COM_power_W'][mask]/1000,color=color,label=label,lw=2)
        ax[1,0].plot(data['system_velocity_m_s'][mask,2],data['GRF_N'][mask,2]/1000,color=color,label=label,lw=2)
        s=row['tempo_sensitivity'];good=[x for x in s if x['contact_feasible']]
        ax[1,1].plot([x['ascent_duration_s'] for x in good],[x['peak_COM_power_W']/1000 for x in good],'o-',color=color,label=label)
    x=np.arange(4);ax[0,1].bar(x-.17,peaks,.34,color='#169f8c',label='Peak');ax[0,1].bar(x+.17,means,.34,color='#9bc9c2',label='Positive-phase mean');ax[0,1].set_xticks(x,labels);ax[0,1].set_ylabel('Vertical COM power [kW]');ax[0,1].set_title('Same output metric; different motion evidence')
    for i,p in enumerate(peaks):ax[0,1].text(i-.17,p+.08,f'{p:.2f}',ha='center',fontsize=9)
    ax[0,1].set_ylim(0,max(peaks)+.6)
    ax[0,0].set(title='Actual seconds from start of positive phase',xlabel='Time [s] · no time warping',ylabel='Vertical COM power [kW]')
    ax[1,0].set(title='Squat force / COM speed · simultaneous values',xlabel='Vertical system COM velocity [m/s]',ylabel='Vertical ground reaction [kN]')
    ax[1,1].set(title='Same squat path, different assumed ascent duration',xlabel='Ascent duration [s]',ylabel='Peak vertical COM power [kW]')
    for a in ax.flat:a.grid(alpha=.2);a.legend(fontsize=9)
    fig.suptitle('Jump / loaded squat · 180 cm / 88 kg',fontsize=20);fig.savefig(OUT/'jump_squat_comparison.png',dpi=160);plt.close(fig)
    for row,data in records:
        fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained');t=data['times']
        for k,(stem,label) in enumerate([('hip','Hip'),('knee','Knee'),('ankle','Ankle')]):
            ids=[i for i,n in enumerate(row['coordinates']) if n.startswith(stem)]
            ax[0,0].plot(t,data['net_joint_moments_Nm'][:,ids].mean(1),label=label)
            ax[0,1].plot(t,data['net_joint_powers_W'][:,ids].mean(1),label=label)
            ax[1,0].plot(t,np.linalg.norm(data['net_joint_reactions_N'][:,ids],axis=2).mean(1),label=label)
        ax[1,1].plot(t,data['COM_power_W'],label='System COM');ax[1,1].plot(t,data['bar_power_W'],label='Bar only');ax[1,1].plot(t,data['segment_energy_rate_W'],label='All segments + bar energy rate',ls=':')
        for a,title,unit in zip(ax.flat,['Net joint moment · mean per side','Net joint power · mean per side','Net intersegmental force · mean per side','Different mechanical power definitions'],['Nm','W','N','W']):a.set(title=title,ylabel=unit,xlabel='Actual time [s]');a.legend();a.grid(alpha=.2)
        fig.suptitle(f"High-bar squat / {row['bar_mass_kg']} kg · prescribed motion",fontsize=17);fig.savefig(OUT/f"squat_{row['bar_mass_kg']}_joint_curves.png",dpi=150);plt.close(fig)


def main():
    baseline=json.loads((OUT/'summary.json').read_text());cmj=baseline['trials'][0];records=[];media={}
    media['cmj']=video('cmj',cmj,np.load(OUT/'s10_CMJ_t1_power.npz'))
    for kg in [60,100,140]:
        folder=OUT/f'squat_{kg}';row=json.loads((folder/'summary.json').read_text());data=np.load(folder/'motion.npz')
        assert row['all_mechanics_gates_passed'];row['tempo_sensitivity']=sensitivity(row,data)
        row['conditional_tension_audit']=json.loads((folder/'tension_audit.json').read_text());assert row['conditional_tension_audit']['status']=='passed'
        records.append((row,data));media[f'squat_{kg}']=video(f'squat_{kg}',row,data)
        with (folder/'power.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['time_s','GRF_vertical_N','system_COM_velocity_m_s','system_COM_power_W','bar_power_W','segment_energy_rate_W']);w.writerows(zip(data['times'],data['GRF_N'][:,2],data['system_velocity_m_s'][:,2],data['COM_power_W'],data['bar_power_W'],data['segment_energy_rate_W']))
        print('SQUAT_VIDEO_PACKAGED',kg,flush=True)
    plots(cmj,records)
    development=None;box_review=''
    candidate=OUT/'box_60_model/summary.json'
    if candidate.exists():
        trial=json.loads(candidate.read_text())
        development=dict(height_m=.6,status=trial['status'],released_in_comparison=False,
                         failed_checks=[key for key,value in trial.get('gates',{}).items() if not value],
                         landing_included=False)
        if development['failed_checks']:
            box_review='<p>A continuous 60 cm development trial was rejected: its feet did not clear and reach the box top. Force balance alone is insufficient. Its model power is excluded from this comparison; impact and landing are also unresolved.</p>'
    report=dict(status='measured_CMJ_and_three_mechanically_checked_prescribed_squats',CMJ_reference=cmj,squats=[r for r,d in records],media=media,
                common_metric='Vertical COM translational pseudo-work rate: vertical ground reaction times body-plus-bar COM velocity.',
                evidence_difference='CMJ uses measured plates and marker-derived kinematics. Squats prescribe cadence/posture and derive loads from the model.',
                equal_power_is_not_equal_muscle_load_or_fatigue=True,box_jump_dynamic_results_completed=False,
                squat_muscle_forces_computed=True,squat_muscle_tensions_independently_validated=False,tension_selection='whole-history normalized force magnitude and continuity; no strength ceiling',assumed_personal_strength_or_1RM=None,box_development_review=development,
                limitations=['Squat cadence is an input, not a prediction from body/bar mass.','2.5 cm heel elevation, 15 degree toe-out and a fixed generic depth are explicit scenario assumptions.','Net intersegmental forces exclude muscle compression and are not articular contact forces.','No fatigue, metabolic cost, injury or damage is inferred.','Box-jump trajectories and landing still require a separate solution.'])
    (OUT/'comparison_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_page(report)
    print('JUMP_SQUAT_COMPARISON_READY',flush=True)


def write_page(report):
    cmj=report['CMJ_reference'];media=report['media'];box_review=''
    development=report.get('box_development_review')
    if development and development.get('failed_checks'):
        box_review='<p>A continuous 60 cm development trial was rejected: its feet did not clear and reach the box top. Force balance alone is insufficient. Its model power is excluded from this comparison; impact and landing are also unresolved.</p>'
    low_review=OUT/'box_feasibility_review.json'
    if low_review.exists():
        review=json.loads(low_review.read_text())
        if not review['box_muscle_comparison_ready']:
            box_review+='<p>The 40 cm feasibility revision corrected liftoff recognition before the nominal toe-support phase, and reduced excessive trunk rotation. Foot clearance and placement still fail; the trial remains excluded from muscle/power comparisons. <a href="box_feasibility_review.json">Latest low-box checks and comparison scope</a>.</p>'
    page=(ROOT/'scripts/jump_squat_power_viewer.html').read_text(encoding='utf-8')
    rows=f"<tr><td>CMJ / Trial 1</td><td>{cmj['peak_COM_vertical_power_W']/1000:.2f}</td><td>{cmj['mean_propulsion_COM_vertical_power_W']/1000:.2f}</td><td>{cmj['propulsion_duration_s']:.3f}</td><td>Measured motion / plates</td></tr>"
    rows+=''.join(f"<tr><td>Squat {r['bar_mass_kg']} kg</td><td>{r['peak_COM_vertical_power_W']/1000:.2f}</td><td>{r['mean_ascent_COM_vertical_power_W']/1000:.2f}</td><td>{r['prescribed_timing_s']['ascent']:.3f}</td><td>Prescribed model scenario</td></tr>" for r in report['squats'])
    page=page.replace('<!--ROWS-->',rows).replace('<!--BOX_DEVELOPMENT_REVIEW-->',box_review).replace('<!--MEDIA_JSON-->',json.dumps(media));(OUT/'comparison.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--page-only',action='store_true');args=parser.parse_args()
    if args.page_only:write_page(json.loads((OUT/'comparison_results.json').read_text()))
    else:main()
