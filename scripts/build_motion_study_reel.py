"""Three slow, continuous multiview replays with actual rectus femoris traces."""
from pathlib import Path
import json,html,math,hashlib,csv,sys,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from case_contours import overview
from launch_sources import load,analysis_path,render_root
from launch_relative_render_palette import color

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/instagram_launch'
BG=(13,23,38);FG=(237,243,251);MUTED=(166,188,209);ACC=(239,149,88)
FPS=30;CYCLE=6.;REPEATS=3;MOTION=18.;DURATION=26
GROUPS=[('QUADS','Front thigh',1),('GLUTES','Hip / buttock',0),('HAMSTRINGS','Back thigh',2),('CALVES','Lower leg',3)]

def font(s,b=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if b else 'C:/Windows/Fonts/arial.ttf',s)
def text(d,xy,s,size=24,c=FG,b=False):
    assert d.textbbox((0,0),s,font=font(size,b))[2]<=1080-xy[0]-25,(xy,s)
    d.text(xy,s,font=font(size,b),fill=c)

def anatomy_icon(d,x,y,kind,scale=.8):
    """Schematic muscle locations; separate from the force-solving anatomy."""
    def box(a):return tuple(v*scale+(x if i%2==0 else y) for i,v in enumerate(a))
    def poly(points,fill):d.polygon([(x+a*scale,y+b*scale) for a,b in points],fill=fill)
    grey=(103,124,141);dark=(48,66,82);orange=ACC
    d.ellipse(box((22,0,36,14)),fill=grey)
    poly([(20,18),(38,18),(40,39),(36,47),(22,47),(18,39)],grey)
    for xx in [18,40]:d.line((x+xx*scale,y+21*scale,x+(xx-8 if xx<29 else xx+8)*scale,y+47*scale),fill=grey,width=max(2,round(6*scale)))
    for xx in [22,36]:
        poly([(xx-6,43),(xx+6,43),(xx+4,72),(xx-3,72)],grey)
        poly([(xx-3,73),(xx+4,73),(xx+3,100),(xx-2,100)],grey)
        d.line((x+(xx-2)*scale,y+102*scale,x+(xx+7)*scale,y+102*scale),fill=grey,width=max(2,round(4*scale)))
    if kind==1:
        for xx in [22,36]:
            poly([(xx-5,46),(xx+5,46),(xx+3,68),(xx-2,71),(xx-5,59)],orange)
            d.line((x+xx*scale,y+49*scale,x+xx*scale,y+67*scale),fill=dark,width=max(1,round(scale)))
    elif kind==0:
        for xx in [22,36]:d.ellipse(box((xx-6,35,xx+6,49)),fill=orange)
    elif kind==2:
        for xx in [22,36]:
            poly([(xx-5,48),(xx+5,48),(xx+3,70),(xx-2,72)],orange)
            d.line((x+xx*scale,y+50*scale,x+xx*scale,y+69*scale),fill=dark,width=max(1,round(scale)))
    elif kind==3:
        for xx in [22,36]:poly([(xx-3,75),(xx+4,75),(xx+5,83),(xx+2,94),(xx-2,91),(xx-5,83)],orange)
    else:
        for xx in [22,36]:poly([(xx-2,45),(xx+2,47),(xx+2,65),(xx,72),(xx-2,66)],orange)

def main():
    preview='--preview' in sys.argv
    cases={};bounds=[[],[]]
    for key in ['cmj','squat_100']:
        p=analysis_path(ROOT,key)
        v=load(ROOT,key);z=np.load(p/'tensions.npz');ids=[i for i,n in enumerate(v['names']) if n in ['recfem_r','recfem_l']]
        assert len(ids)==2
        times=z['times'];rf=z['force_N'][:,ids].sum(1)/2
        group=overview(v)[[1,0,2,3]]
        folder=render_root(ROOT)/key;files=sorted((folder/'study_render').glob('frame_*.png'))
        expected=round(times[-1]*60)+1
        assert preview or len(files)==expected,(key,len(files),expected)
        # One crop per view, shared between exercises and across all frames.
        for file in files:
            a=np.asarray(Image.open(file))[:,:,3]
            for view,(lo,hi) in enumerate([(0,800),(800,1600)]):
                yy,xx=np.where(a[:,lo:hi]>8);xx+=lo
                assert len(xx)>0 and xx.min()>lo+2 and xx.max()<hi-3 and yy.min()>2 and yy.max()<857,('Clipped view',file,view)
                bounds[view].append((xx.min(),yy.min(),xx.max()+1,yy.max()+1))
        available=files
        if preview:
            indices=np.array([int(f.stem.split('_')[-1]) for f in files])
            files=[available[int(np.argmin(abs(indices-i)))] for i in range(expected)]
        cases[key]=dict(times=times,rf=rf,groups=group,group_times=np.array(v['times']),files=files,T=float(times[-1]),
            phase_interval=tuple(json.loads((p/'setup.json').read_text())[k] for k in ['countermovement_end_s','takeoff_s']) if key=='cmj' else (1.65,2.55),key=key,
            source_tension_sha256=hashlib.sha256((p/'tensions.npz').read_bytes()).hexdigest(),
            source_motion_sha256=hashlib.sha256((p/'motion.npz').read_bytes()).hexdigest())
    crops=[];sizes=[]
    for view in range(2):
        b=np.array(bounds[view]);lo=view*800;hi=lo+800
        crop=(max(lo,int(b[:,0].min())-9),max(0,int(b[:,1].min())-9),min(hi,int(b[:,2].max())+9),min(860,int(b[:,3].max())+9))
        scale=min((475 if view==0 else 225)/(crop[2]-crop[0]),420/(crop[3]-crop[1]))
        crops.append(crop);sizes.append((round((crop[2]-crop[0])*scale),round((crop[3]-crop[1])*scale)))
    for case in cases.values():
        case['images']=[[],[]]
        for file in case['files']:
            im=Image.open(file).convert('RGBA')
            for view in range(2):case['images'][view].append(im.crop(crops[view]).resize(sizes[view],Image.Resampling.LANCZOS))
    group_ref=np.maximum(*[x['groups'].max(1) for x in cases.values()])
    ymax=math.ceil(max(x['rf'].max() for x in cases.values())/200)*200
    for case in cases.values():
        case['plot_y']=case['rf']/ymax
        case['peaks']=case['groups'].max(1)
    muscle_upper=json.loads((render_root(ROOT)/'cmj/study_muscle_mapping.json').read_text())['common_upper_N']
    jump_motion=np.load(render_root(ROOT)/'cmj/motion.npz')
    jt=jump_motion['times'];jz=jump_motion['com'][:,2]
    takeoff=cases['cmj']['phase_interval'][1]
    jump_height_m=float(jz[jt>=takeoff].max()-np.interp(takeoff,jt,jz))
    standing_height_m=float(jz.max()-jz[jt<=.25].mean())
    height_cm=round(100*jump_height_m)
    comparisons=[]
    for j,(name,region,index) in enumerate(GROUPS):
        jf=float(cases['cmj']['peaks'][j]);sf=float(cases['squat_100']['peaks'][j])
        comparisons.append(dict(name=name,region=region,icon=index,jump_N=jf,squat_N=sf,
            higher='JUMP' if jf>sf else 'SQUAT',higher_to_lower_ratio=max(jf,sf)/min(jf,sf)))
    jf=float(cases['cmj']['rf'].max());sf=float(cases['squat_100']['rf'].max())
    comparisons.append(dict(name='RECTUS FEMORIS',region='Part of the quadriceps',icon=4,jump_N=jf,squat_N=sf,
        higher='JUMP' if jf>sf else 'SQUAT',higher_to_lower_ratio=max(jf,sf)/min(jf,sf)))

    def graph(im,d,case,t,base):
        x0=765;y0=base+100;w=260;h=235
        anatomy_icon(d,698,base+18,4,.52)
        text(d,(735,base+20),'RECTUS FEMORIS',25,FG,True)
        a,b=case['phase_interval']
        d.rectangle((x0+w*a/case['T'],y0,x0+w*b/case['T'],y0+h),fill=(29,45,58))
        for u in [0,.5,1]:
            yy=y0+h*(1-u);d.line((x0,yy,x0+w,yy),fill=(47,65,82),width=1)
            text(d,(724,yy-10),f'{ymax*u/1000:.1f}',18,MUTED)
        text(d,(730,y0-27),'kN',18,MUTED)
        points=[(x0+w*tt/case['T'],y0+h*(1-f)) for tt,f in zip(case['times'],case['plot_y'])]
        d.line(points,fill=(105,126,143),width=3)
        past=[p for p,tt in zip(points,case['times']) if tt<=t]
        cur=float(np.interp(t,case['times'],case['rf']));cx=x0+w*t/case['T'];cy=y0+h*(1-cur/ymax)
        if len(past)>1:d.line([*past,(cx,cy)],fill=ACC,width=4)
        d.line((cx,y0,cx,y0+h),fill=(223,234,244),width=2);d.ellipse((cx-5,cy-5,cx+5,cy+5),fill=FG)
        text(d,(x0,y0+h+12),'0',18,MUTED);text(d,(x0+w-48,y0+h+12),f'{case["T"]:.1f}s',18,MUTED)
        phase=('PREPARATION' if t<a else 'PUSH-OFF' if t<b else 'FLIGHT') if case['key']=='cmj' else ('LOWERING' if t<1.5 else 'PAUSE' if t<a else 'ASCENT' if t<b else 'LOCKOUT')
        text(d,(745,base+365),phase,20,MUTED)
        text(d,(745,base+405),f'{cur/1000:.2f} kN',33,ACC,True)
        text(d,(745,base+451),f'Peak {case["rf"].max()/1000:.2f} kN',22,MUTED)

    def frame(t):
        im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
        text(d,(45,28),'@biomechanics_ksm',23,MUTED)
        text(d,(45,72),f'{height_cm} CM JUMP vs 100 KG SQUAT',40,FG,True)
        text(d,(45,129),'180 CM / 88 KG',24,MUTED)
        p=(t%CYCLE)/CYCLE
        for key,base,title in [('cmj',202,'JUMP'),('squat_100',820,'SQUAT')]:
            case=cases[key];physical=p*case['T']
            text(d,(45,base),title,30,FG,True)
            text(d,(735,base-15),f'{case["T"]/CYCLE:.2f}x speed',22,MUTED)
            fi=physical*60;f0=int(fi);f1=min(f0+1,len(case['files'])-1);mix=fi-f0
            for view in range(2):
                a=case['images'][view][f0];b=case['images'][view][f1]
                body=Image.blend(a,b,mix) if mix>1e-6 else a
                # Native anatomy frames are resampled only for slower display.
                # The underlying force and motion clocks remain unchanged.
                x=(20+(475-sizes[view][0])//2) if view==0 else (490+(225-sizes[view][0])//2)
                y=base+80+(420-sizes[view][1])
                im.paste(body,(x,y),body)
            graph(im,d,case,physical,base)
            for j,(name,region,icon) in enumerate(GROUPS):
                x=45+j*250;y=base+513;force=float(np.interp(physical,case['group_times'],case['groups'][j]));u=force/group_ref[j]
                anatomy_icon(d,x,y-5,icon,.72)
                text(d,(x+50,y),name,19,FG,True)
                d.rounded_rectangle((x+50,y+56,x+191,y+65),radius=4,fill=(41,60,77))
                if u>0:d.rounded_rectangle((x+50,y+56,x+50+max(1,141*u),y+65),radius=4,fill=ACC)
                text(d,(x+50,y+27),f'{force/1000:.2f} kN',21,MUTED)
        d.line((45,800,1035,800),fill=(47,65,82),width=1)
        text(d,(45,1430),'MUSCLE TENSION',20,MUTED)
        ramp=np.array([color(x) for x in np.linspace(0,1,280)],dtype=np.uint8)[None]
        im.paste(Image.fromarray(ramp).resize((280,10)),(45,1469))
        text(d,(340,1453),f'0-{muscle_upper/1000:.0f} kN / same scale',20,MUTED)
        text(d,(45,1500),'Joint reaction: black to red, 0-20 kN',19,MUTED)
        return im

    def notes():
        im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
        text(d,(50,55),'WHICH HAD HIGHER PEAK TENSION?',36,FG,True)
        text(d,(50,117),f'{height_cm} cm jump / 100 kg squat   |   180 cm / 88 kg',28,MUTED)
        text(d,(50,165),'Percentages: increase from the lower to the higher peak.',22,MUTED)
        for j,c in enumerate(comparisons):
            y=222+j*165;anatomy_icon(d,52,y,c['icon'],1.05)
            text(d,(145,y+2),c['name'],29,FG,True)
            text(d,(145,y+47),c['region'],21,MUTED)
            percent=100*(c['higher_to_lower_ratio']-1)
            text(d,(660,y+2),f'{c["higher"]} +{percent:.0f}%',30,ACC,True)
            text(d,(145,y+89),f'Jump {c["jump_N"]/1000:.2f} kN   /   Squat {c["squat_N"]/1000:.2f} kN',24,MUTED)
        text(d,(50,1100),'Model estimates / mean per side / one condition each',22,MUTED)
        text(d,(50,1139),f'Jump height: takeoff-to-apex COM rise ({100*jump_height_m:.1f} cm)',22,MUTED)
        text(d,(50,1178),'Same original movement timing in calculations. No landing analysis.',22,MUTED)
        d.line((50,1252,1030,1252),fill=(47,65,82),width=1)
        sources=['MODEL / REFERENCES',
            'Adapted MyoSim myofullbody / MuJoCo; conditional muscle-force sharing.',
            'Seth et al. (2019), via RMR: 10.3389/fnbot.2019.00090',
            'Calisti et al. (2025): 10.6084/m9.figshare.28890545',
            'Goodman et al. (2023): 10.1519/JSC.0000000000004506; Wade et al. (2018): 10.1038/s41598-018-20387-0',
            'BodyParts3D / DBCLS via MyoHub: CC BY 4.0. Z-Anatomy / Kervyn, Zielinski & Vinent: CC BY-SA 4.0.',
            'Supplementary BP3D / DBCLS: CC BY-SA 2.1 JP. Anatomy registered, posed and recoloured.',
            'AI agent assisted analysis. Full conditions and references in the caption.']
        for j,s in enumerate(sources):text(d,(50,1280+j*29),s,17 if j else 19,MUTED)
        return im

    def credits():
        im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
        text(d,(50,65),'MODEL / REFERENCES',39,FG,True)
        lines=['Adapted MyoSim myofullbody / MuJoCo',
            'Conditional muscle-force sharing; no strength ceiling.',
            'Shoulder paths: Seth et al. (2019), via RMR',
            '10.3389/fnbot.2019.00090',
            'CMJ data: Calisti, Mohr & Federolf (2025)',
            '10.6084/m9.figshare.28890545',
            'Comparison: Goodman et al. (2023); Wade et al. (2018)',
            '10.1519/JSC.0000000000004506',
            '10.1038/s41598-018-20387-0',
            'BodyParts3D / DBCLS via MyoHub: CC BY 4.0',
            'Z-Anatomy: Kervyn, Zielinski & Vinent: CC BY-SA 4.0',
            'Supplementary BP3D / DBCLS: CC BY-SA 2.1 JP',
            'Anatomy registered, posed and recoloured.',
            'AI agent assisted analysis and visualization.']
        for j,s in enumerate(lines):text(d,(50,180+j*76),s,24,MUTED)
        return im
    def exposure_summary():
        exposure=json.loads((OUT/'tension_exposure.json').read_text());ec=exposure['cases']
        images={k:Image.open(render_root(ROOT)/k/'exposure_render/frame_0000.png').convert('RGBA') for k in cases}
        bb=[[],[]];upper=[]
        for key,im in images.items():
            for view,(lo,hi) in enumerate([(0,800),(800,1600)]):
                yy,xx=np.where(np.array(im)[:,lo:hi,3]>8);xx+=lo
                assert len(xx) and xx.min()>lo+2 and xx.max()<hi-3 and yy.min()>2 and yy.max()<857
                bb[view].append((xx.min(),yy.min(),xx.max()+1,yy.max()+1))
            upper.append(json.loads((render_root(ROOT)/key/'exposure_render_mapping.json').read_text())['common_upper_Ns'])
        assert upper[0]==upper[1]
        allb=np.array(bb).reshape(-1,4);top=int(allb[:,1].min())-10;bottom=int(allb[:,3].max())+10
        scale=540/(bottom-top)
        im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
        text(d,(45,48),'TOTAL',52,FG,True)
        text(d,(45,115),'TENSION x TIME',25,MUTED)
        text(d,(45,156),'180 CM / 88 KG',22,MUTED)
        for key,offset,title in [('cmj',20,f'{height_cm} CM JUMP'),('squat_100',550,'100 KG SQUAT')]:
            text(d,(offset+25,205),f'{title} / {ec[key]["duration_s"]:.2f} s',29,FG,True)
            for view in range(2):
                b=np.array(bb[view]);crop=(int(b[:,0].min())-10,top,int(b[:,2].max())+10,bottom)
                body=images[key].crop(crop);body=body.resize((round(body.width*scale),540),Image.Resampling.LANCZOS)
                cx=offset+125+view*230
                im.paste(body,(round(cx-body.width/2),285),body)
        text(d,(45,855),'TOTAL / kN s',22,MUTED)
        ramp=np.array([color(x) for x in np.linspace(0,1,350)],dtype=np.uint8)[None]
        im.paste(Image.fromarray(ramp).resize((350,12)),(45,903))
        text(d,(45,924),'0',18,MUTED);text(d,(330,924),f'{upper[0]/1000:.0f} kN s',18,MUTED)
        text(d,(430,889),'Same scale / one repetition',21,MUTED)
        text(d,(425,968),'PEAK',28,FG,True);text(d,(760,968),'TOTAL',28,FG,True)
        for j,(name,region,icon) in enumerate(GROUPS):
            y=1025+j*78;anatomy_icon(d,44,y-5,icon,.48);text(d,(92,y),name,23,FG,True)
            jf=ec['cmj']['group_integral_mean_per_side_Ns'][j]/1000;sf=ec['squat_100']['group_integral_mean_per_side_Ns'][j]/1000
            c=comparisons[j];text(d,(425,y),f'{c["higher"]} {c["higher_to_lower_ratio"]:.2f}x',27,ACC if c['higher']=='JUMP' else FG,True)
            winner='JUMP' if jf>sf else 'SQUAT';text(d,(760,y),f'{winner} {max(jf,sf)/min(jf,sf):.2f}x',27,ACC if winner=='JUMP' else FG,True)
        text(d,(45,1360),'Model estimates / per side / no landing',19,MUTED)
        refs=['MyoSim / MuJoCo. Seth 2019; Calisti 2025; Kjaer 2016. Methods and sources in caption.',
            'BodyParts3D / DBCLS via MyoHub: CC BY 4.0. Z-Anatomy / Kervyn et al.: CC BY-SA 4.0.',
            'Supplementary BP3D / DBCLS: CC BY-SA 2.1 JP. Anatomy registered, posed and recoloured.']
        for j,s in enumerate(refs):text(d,(45,1430+j*23),s,15,MUTED)
        return im
    info=exposure_summary();credit=info
    if preview:
        frame(45/14).save(OUT/'study_layout_preview.png');info.save(OUT/'study_summary_preview.png')
        print('STUDY_LAYOUT_PREVIEW',flush=True);return
    info.save(OUT/'study_summary.png');credit.save(OUT/'study_sources.png')
    frame(3.75).save(OUT/'study_cover.png')
    path=OUT/'jump_squat_30cm_v2.mp4'
    writer=imageio_ffmpeg.write_frames(str(path),(1080,1920),fps=FPS,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,
        output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None)
    try:
        for k in range(FPS*DURATION):
            t=k/FPS;im=frame(t) if t<MOTION else info
            assert np.all(np.asarray(im)[1536:]==BG)
            writer.send(np.asarray(im))
            if k in [45,100,145,220]:im.save(OUT/f'study_preview_{k:03d}.png')
    finally:writer.close()
    shutil.copy2(path,OUT/'jump_squat_motion_study.mp4')
    meta=dict(file=path.name,width=1080,height=1920,fps=FPS,duration_s=DURATION,repeats_per_motion=3,
        no_additional_profile_images=True,bottom_reserved_px=384,rectus_femoris='Mean of recfem_r and recfem_l element forces; not quadriceps sum.',
        display_retiming_only=True,motion_and_force_amplitudes_changed=True,new_jump_heights_solved=True,landing_included=False,
        colors='One common absolute linear N scale across muscles and motions; not activation or capacity.',
        slow_display_interpolation='Linear blending between adjacent 60 Hz source renders; curves use source force times.',
        views=['front three-quarter','side'],crops=crops,cases={key:dict(duration_s=c['T'],playback_rate=c['T']/CYCLE,
            rectus_peak_N=float(c['rf'].max()),rectus_peak_time_s=float(c['times'][c['rf'].argmax()]),
            force_sha256=c['source_tension_sha256'],motion_sha256=c['source_motion_sha256']) for key,c in cases.items()})
    meta.update(jump_height_m=jump_height_m,jump_height_definition='Prescribed scenario COM rise from takeoff to apex; not jump-and-reach.',
        standing_to_apex_COM_rise_m=standing_height_m,not_a_Sargent_reach_measure=True,
        tension_exposure=json.loads((OUT/'tension_exposure.json').read_text()),
        peak_tension_comparisons=comparisons,conclusion_duration_s=8,source_credits='compact footer on force-time standing-pose conclusion card')
    (OUT/'study_metrics.json').write_text(json.dumps(meta,indent=2))
    sources=(OUT/'source_links.txt').read_text(encoding='utf-8').split('VIDEO SCOPE')[0]
    sources+='HEIGHT REFERENCE\nKjaer et al. (2016), https://doi.org/10.1186/s13102-016-0059-4\n\n'
    sources+='''VIDEO SCOPE
Revised multiview motion study. One prescribed 30cm jump based on a measured posture path, and one prescribed 100kg squat.
180cm / 88kg generic model. Three slow display replays per motion.
Rectus femoris is the mean of recfem_r and recfem_l forces, not the quadriceps group sum.
Each graph uses original physical time and the same absolute kN axis.
Moving surface colours use one absolute linear N scale across all anatomical muscles and both source motions.
Group bars use bilateral group force divided by two and its peak across both source motions.
Normalization is neither muscle activation nor maximum strength usage.
Jump propulsion, ballistic flight and muscle tensions were recomputed. Display frames are additionally interpolated for slow playback.
The prescribed jump is an illustrative inverse-dynamics scenario, not a newly measured or fully validated human jump. No landing, fatigue or injury predictions.
Projected foot support regions; remaining floating-root rotational residual is disclosed in the mechanics audit. No invented support torque is applied.
Height definitions: quiet-standing-to-apex COM rise and takeoff-to-apex COM rise are separate. Neither is a Sargent jump-and-reach score.
Summary: scalar force-time integral (N s), CMJ onset-to-takeoff; squat descent-start to ascent-end. Exact windows are in study_metrics.json.
Standing contours show one movement's integrals on a common display pose, with a single absolute scale for all anatomical muscles.
Three visual replays do not triple the integrals. Group table values are bilateral sums divided by two.
'''
    (OUT/'study_source_links.txt').write_text(sources,encoding='utf-8')
    for key,c in cases.items():
        with (OUT/f'{key}_study_forces.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['physical_time_s','rectus_femoris_mean_per_side_N',*[name+'_mean_per_side_N' for name,_,_ in GROUPS]])
            for k,tt in enumerate(c['times']):w.writerow([tt,c['rf'][k],*[np.interp(tt,c['group_times'],row) for row in c['groups']]])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Biomechanics motion study</title><style>body{background:#111b29;color:#e7eef7;font:17px/1.6 system-ui;max-width:960px;margin:28px auto;padding:18px}a{color:#dca677}img{max-width:100%;display:block}video{max-height:760px;max-width:100%;display:block}section{margin:28px 0}h1,h2{font-weight:500}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#192737;padding:20px;border-radius:8px}</style><h1>Biomechanics_ksm</h1><section><video controls playsinline preload="metadata" src="jump_squat_30cm_v2.mp4" poster="study_cover.png"></video></section>'''
    for file,title in [('profile_bio_en.txt','Profile bio'),('intro_post_en.txt','Short introduction'),('reel_caption_en.txt','Reel caption')]:
        page+=f'<section><h2>{title}</h2><pre>{html.escape((OUT/file).read_text(encoding="utf-8"))}</pre></section>'
    jump_audit='../../'+str((analysis_path(ROOT,'cmj')/'verification.json').relative_to(ROOT)).replace('\\','/')
    page+=f'<p><a download href="{path.name}">Download video</a> · <a href="study_source_links.txt">Sources</a> · <a href="study_metrics.json">Scope and metrics</a> · <a href="{jump_audit}">New jump mechanics audit</a> · <a href="jump_height_method_review.txt">Height and assumptions</a> · <a href="../../results/power_comparison/comparison.html">Earlier analysis viewer</a></p></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print('MOTION_STUDY_READY',json.dumps(meta['cases']),flush=True)

if __name__=='__main__':main()
