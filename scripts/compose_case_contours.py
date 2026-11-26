"""Portrait comparison with absolute-N anatomy, nine time contours and power."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from case_contours import ROWS,rgb,joint_rgb,overview,load
from video_credits import make_card

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'
BG=(13,23,38);FG=(237,243,251);MUTED=(166,188,209);GREEN=(140,226,213)

def font(n,b=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if b else 'C:/Windows/Fonts/arial.ttf',n)
def text(d,p,s,n=20,c=FG,b=False):d.text(p,str(s),font=font(n,b),fill=c)

def cmj_body(source,i,contour,t):
    base=np.array(Image.open(source/f'anatomy/base/frame_{i:04d}.png').convert('RGBA'))
    mask=np.array(Image.open(source/f'anatomy/mask/frame_{i:04d}.png').convert('RGB')).astype(int)
    times=np.array(contour['times']);F=np.array(contour['group_tension_N']);J=np.array(contour['joint_force_N'])
    f=np.array([np.interp(t,times,x) for x in F]);j=np.array([np.interp(t,times,x) for x in J])
    ids=np.rint(mask[:,:,0]/4).astype(int);valid=(abs(mask[:,:,1]-(255-mask[:,:,0]))<5)&(abs(mask[:,:,2]-128)<5)&(ids>0)&(ids<=len(f))
    wanted=np.vstack([np.zeros(3),rgb(f)])[np.clip(ids,0,len(f))]
    shade=np.clip(.35+.85*base[:,:,:3].mean(2)/255,.35,1.15)
    base[:,:,:3][valid]=(.15*base[:,:,:3]+.85*np.clip(wanted*shade[:,:,None],0,255))[valid].astype(np.uint8)
    valid=(abs(mask[:,:,1]-(255-mask[:,:,0]))<5)&(abs(mask[:,:,2]-32)<5)
    ids=np.clip(np.rint(mask[:,:,0]/16).astype(int),0,len(j));colors=joint_rgb(np.r_[0,j])[ids]
    base[:,:,:3][valid]=np.clip(colors*shade[:,:,None],0,255)[valid].astype(np.uint8)
    return Image.fromarray(base)

def video(key,row,data,poster_only=False):
    cmj=key=='cmj';box=row.get('movement')=='box_jump_model';contour=load(ROOT,key);ct=np.array(contour['times']);rows=overview(contour)
    if cmj:
        source=ROOT/'results/measured_jump_visualization';source_t=np.array(json.loads((source/'data.json').read_text())['times']);T=float(source_t[-1])
        setup=json.loads((ROOT/'results/measured_jump_analysis/s10_CMJ_t1/setup.json').read_text());offset=setup['source_start_s']
        times=data['times_source_s']-offset;power=data['COM_vertical_power_W'];force=data['vertical_GRF_N']
        bottom=up=row['source_bottom_s']-offset;end=row['source_takeoff_s']-offset
        peak=row['peak_COM_vertical_power_W'];mean=row['mean_propulsion_COM_vertical_power_W'];duration=row['propulsion_duration_s'];title='COUNTERMOVEMENT JUMP'
        evidence='Measured motion / conditional sharing'
    else:
        source=OUT/key;T=float(data['times'][-1]);source_t=np.arange(round(T*30)+1)/30
        times=data['times'];power=data['COM_power_W'];force=data['GRF_N'][:,2]
        phase=row['phase_times_s'];bottom=phase['bottom'];up=phase['start_ascent'];end=phase['end_ascent']
        peak=row['peak_COM_vertical_power_W'];mean=row['mean_ascent_COM_vertical_power_W'];duration=row['prescribed_timing_s']['ascent']
        title=f"BOX JUMP / {row['box_height_m']*100:.0f} CM" if box else f"HIGH-BAR SQUAT / {row['bar_mass_kg']} KG"
        evidence='Prescribed model / conditional sharing'
    selected=(times>=0)&(times<=T)
    low,high=-3.,10.
    assert power[selected].min()/1000>=low and power[selected].max()/1000<=high,'Power plot range must include all data'
    means=np.trapezoid(rows,ct,axis=1)/(ct[-1]-ct[0]);peaks=rows.max(1)
    display_times=np.minimum(np.arange(round(T*30)+1)/30,T);cache=[]
    peak_time=row['peak_power_source_time_s']-offset if cmj else float(times[np.argmax(np.where((times>=up)&(times<=end),power,-np.inf))])
    poster_index=int(np.argmin(abs(display_times-peak_time)))
    iterator=[(poster_index,display_times[poster_index])] if poster_only else enumerate(display_times)
    for k,t in iterator:
        i=int(np.argmin(abs(source_t-t)))
        body=cmj_body(source,i,contour,t) if cmj else Image.open(source/f'render/frame_{i:04d}.png').convert('RGBA')
        # Remove transparent headroom; all anatomical surfaces remain in frame.
        a=np.asarray(body);assert not (a[:90,:,3]>8).any(),'Do not crop visible anatomy'
        body=body.crop((0,90,1080,900)).resize((972,729),Image.Resampling.LANCZOS)
        im=Image.new('RGB',(1080,1920),BG);im.paste(body,(54,205),body);d=ImageDraw.Draw(im)
        text(d,(54,28),'JUMP / SQUAT MECHANICS',39,b=True);text(d,(54,86),title,28,b=True)
        text(d,(54,132),'180 cm / 88 kg  |  '+evidence,20,MUTED)
        phase=('COUNTERMOVEMENT' if t<up else 'PROPULSION' if t<end else 'FLIGHT') if cmj or box else ('READY' if t<.25 else 'DESCENT' if t<bottom else 'BOTTOM' if t<up else 'ASCENT' if t<end else 'LOCKOUT')
        text(d,(54,177),phase,24,GREEN,True);text(d,(842,177),f'{t:.2f} s',24)
        text(d,(188,923),'FRONT OBLIQUE',17,MUTED);text(d,(805,923),'POSTERIOR' if cmj else 'SIDE',17,MUTED)
        text(d,(54,956),'Muscle N',18,MUTED)
        for x,v,label in [(175,0,'0'),(270,100,'100'),(385,1000,'1,000'),(515,20000,'20,000+')]:
            d.rectangle((x,959,x+17,974),fill=tuple(rgb(v)));text(d,(x+24,953),label,17,MUTED)
        text(d,(735,956),'Joint N',18,MUTED)
        for x,v,label in [(817,0,'0'),(904,20000,'20,000+')]:
            d.rectangle((x,959,x+17,974),fill=tuple(joint_rgb(v)),outline=(70,80,95));text(d,(x+24,953),label,17,MUTED)
        p=float(np.interp(t,times,power));F=float(np.interp(t,times,force))
        text(d,(54,1001),'VERTICAL COM POWER',23,b=True);text(d,(828,998),f'{p/1000:+.2f} kW',27,GREEN,True)
        x0,x1,y0,y1=290,1000,1041,1119;X=lambda s:x0+s/T*(x1-x0);Y=lambda v:y1-(v/1000-low)/(high-low)*(y1-y0)
        d.rectangle((round(X(up)),y0,round(X(end)),y1),fill=(34,49,56))
        for value in [-3,0,5,10]:
            y=round(Y(value*1000));d.line((x0,y,x1,y),fill=(55,74,92));text(d,(245,y-8),value,15,MUTED)
        d.line([(X(float(tt)),Y(float(pp))) for tt,pp in zip(times[selected],power[selected])],fill=GREEN,width=3)
        cursor=X(t);d.line((round(cursor),y0,round(cursor),y1),fill=FG,width=2)
        text(d,(54,1132),f'Peak {peak/1000:.2f} kW     Positive-phase mean {mean/1000:.2f} kW     Duration {duration:.3f} s',20,MUTED)
        text(d,(54,1170),'MUSCLE TENSION OVER TIME',24,b=True);text(d,(802,1174),'MEAN / PEAK [N]',18,MUTED)
        x0,width,top,rh=290,470,1206,28
        for j,(label,_) in enumerate(ROWS):
            y=top+j*rh;text(d,(54,y),label,24)
            heat_values=np.interp(np.linspace(0,T,width),ct,rows[j])
            heat=Image.fromarray(rgb(heat_values)[None]).resize((width,22),Image.Resampling.NEAREST);im.paste(heat,(x0,y))
            text(d,(804,y),f'{means[j]:,.0f} / {peaks[j]:,.0f}',22,MUTED)
        cursor=x0+width*t/T;d.line((round(cursor),top-6,round(cursor),top+len(ROWS)*rh-4),fill=FG,width=3)
        text(d,(x0,1466),'0.0 s',17,MUTED);text(d,(x0+width-56,1466),f'{T:.2f} s',17,MUTED);text(d,(804,1466),'Mean per side',17,MUTED)
        text(d,(54,1503),f'GRF {F:,.0f} N  |  1x  |  3 replays  |  Model tension, not EMG',18,MUTED)
        assert np.all(np.asarray(im)[1536:]==BG);cache.append(im)
    if poster_only:
        cache[0].save(OUT/f'{key}_tension_comparison.png');return dict(poster=f'{key}_tension_comparison.png')
    credit_image,credit=make_card(key)
    path=OUT/f'{key}_tension_credited.mp4';writer=imageio_ffmpeg.write_frames(str(path),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-movflags','+faststart']);writer.send(None);count=0
    try:
        for replay in range(3):
            for im in cache:writer.send(np.asarray(im));count+=1
            for _ in range(12):writer.send(np.asarray(cache[-1]));count+=1
        motion_frames=count
        for _ in range(credit['frames']):writer.send(np.asarray(credit_image));count+=1
    finally:writer.close()
    actual,seconds=imageio_ffmpeg.count_frames_and_secs(str(path));assert actual==count
    cache[poster_index].save(OUT/f'{key}_tension_comparison.png')
    return dict(file=path.name,poster=f'{key}_tension_comparison.png',title=title,frames=count,duration_s=seconds,fps=30,display_speed=1,replays=3,independent_clock=True,bottom_reserved_px=384,
                muscle_contour_rows=[r[0] for r in ROWS],muscle_values='conditional group tension N, mean per side',mean_scope='one full source motion; display replays excluded',muscle_and_joint_palette_upper_N=20000,landing_included=False if cmj or box else None,source_motion_video_frames=motion_frames,credits=credit)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--key',default='squat_100');p.add_argument('--poster',action='store_true');a=p.parse_args()
    row=json.loads((OUT/'summary.json').read_text())['trials'][0] if a.key=='cmj' else json.loads((OUT/a.key/'summary.json').read_text())
    data=np.load(OUT/'s10_CMJ_t1_power.npz') if a.key=='cmj' else np.load(OUT/a.key/'motion.npz')
    print(video(a.key,row,data,a.poster))
