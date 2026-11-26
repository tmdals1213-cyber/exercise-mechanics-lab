"""First portrait reel: current jump versus 100 kg squat, no height equivalence claim."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from case_contours import load,overview,rgb

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/instagram_launch'
BG=(13,23,38);FG=(237,243,251);MUTED=(166,188,209);ACC=(218,158,106)
GROUPS=[('QUADS','FRONT THIGH',1),('GLUTES','HIP / BUTTOCK',0),('HAMSTRINGS','BACK THIGH',2),('CALVES','LOWER LEG',3)]
FPS=30;DURATION=25


def font(size,bold=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size)
def label(draw,xy,value,size=28,color=FG,bold=False):draw.text(xy,value,font=font(size,bold),fill=color)


def location(draw,x,y,index):
    c=(115,135,153);draw.ellipse((x+21,y,x+39,y+18),fill=c)
    draw.rounded_rectangle((x+14,y+22,x+46,y+61),radius=8,fill=c)
    draw.line((x+13,y+27,x+2,y+61),fill=c,width=7);draw.line((x+47,y+27,x+58,y+61),fill=c,width=7)
    for a in [x+20,x+40]:draw.line((a,y+58,a,y+105),fill=c,width=10)
    yy=(y+62,y+50,y+62,y+86)[index];hh=(23,16,23,19)[index]
    for a in [x+20,x+40]:draw.rounded_rectangle((a-5,yy,a+5,yy+hh),radius=4,fill=ACC)


def main():
    cases={};bounds=[];records=[]
    windows={'cmj':(.928,1.22),'squat_100':(1.65,2.55)}
    for key in ['cmj','squat_100']:
        folder=OUT/'render_cases'/key;data=load(ROOT,key);ts=np.array(data['times']);values=overview(data)
        files=sorted((folder/'render').glob('frame_*.png'));expected=round(ts[-1]*FPS)+1
        assert len(files)==expected,(key,len(files),expected)
        images=[]
        for p in files:
            im=Image.open(p).convert('RGBA');a=np.asarray(im);yy,xx=np.where(a[:,:,3]>8)
            assert len(xx) and xx.min()>2 and xx.max()<im.width-3 and yy.min()>2 and yy.max()<im.height-3,('Clipped anatomy',p)
            bounds.append((xx.min(),yy.min(),xx.max()+1,yy.max()+1));images.append(im)
        lo,hi=windows[key];mask=(ts>=lo)&(ts<=hi)
        peaks=np.array([values[i,mask].max() for _,_,i in GROUPS])
        cases[key]=dict(times=ts,values=values,images=images,T=float(ts[-1]),peaks=peaks)
    bounds=np.array(bounds);crop=(max(0,int(bounds[:,0].min())-15),max(0,int(bounds[:,1].min())-15),
        min(1080,int(bounds[:,2].max())+15),min(620,int(bounds[:,3].max())+15))
    factor=min(972/(crop[2]-crop[0]),490/(crop[3]-crop[1]));size=(round((crop[2]-crop[0])*factor),round((crop[3]-crop[1])*factor))
    for case in cases.values():case['images']=[im.crop(crop).resize(size,Image.Resampling.LANCZOS) for im in case['images']]
    ratios=cases['cmj']['peaks']/cases['squat_100']['peaks']
    for j,(name,region,_) in enumerate(GROUPS):records.append(dict(group=name,region=region,
        jump_peak_N=float(cases['cmj']['peaks'][j]),squat_peak_N=float(cases['squat_100']['peaks'][j]),
        jump_to_squat_peak_ratio=float(ratios[j])))

    def motion_frame(t,poster=False):
        j=0 if t<6 else min(3,1+int((t-6)//4));name,region,group=GROUPS[j]
        im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
        label(d,(54,30),'@biomechanics_ksm',24,MUTED)
        label(d,(54,76),'JUMP vs 100 KG SQUAT',43,FG,True)
        label(d,(54,144),name,34,ACC,True);label(d,(310,148),region,27,MUTED)
        location(d,944,119,j)
        for key,y,heat_y,title,ratio in [('cmj',218,746,'BODYWEIGHT JUMP',ratios[j]),('squat_100',847,1375,'BACK SQUAT / 100 KG',1.)]:
            case=cases[key];physical=(1.016 if key=='cmj' else 1.9) if poster else t%case['T']
            index=min(round(physical*FPS),len(case['images'])-1);body=case['images'][index]
            # A short cut between retained replays; no invented landing pose/force.
            phase_opacity=min(1.,physical/.07,(case['T']-physical)/.07) if not poster else 1.
            if phase_opacity<1:
                body=body.copy();body.putalpha(body.getchannel('A').point(lambda v:round(v*max(0,phase_opacity))))
            im.paste(body,((1080-size[0])//2,y),body)
            label(d,(54,y-20),title,27,MUTED)
            heat=np.interp(np.linspace(0,case['T'],740),case['times'],case['values'][group])
            im.paste(Image.fromarray(rgb(heat)[None]).resize((740,18)),(54,heat_y))
            x=round(54+740*physical/case['T']);d.line((x,heat_y-5,x,heat_y+23),fill=FG,width=3)
            label(d,(828,heat_y-18),f'{ratio:.2f}x',36,FG,True)
        d.line((54,800,1026,800),fill=(45,61,77),width=1)
        label(d,(54,1440),'PEAK TENSION / SQUAT = 1.00x',25,MUTED)
        return im

    def summary():
        im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
        label(d,(54,70),'PEAK MUSCLE TENSION',38,FG,True)
        label(d,(54,136),'Jump relative to the 100 kg squat',28,MUTED)
        for j,r in enumerate(records):
            y=245+j*126;label(d,(54,y),r['group'],32,FG,True)
            label(d,(654,y),f"{r['jump_to_squat_peak_ratio']:.2f}x",36,ACC,True)
            label(d,(54,y+48),f"Jump {r['jump_peak_N']/1000:.2f} kN   /   Squat {r['squat_peak_N']/1000:.2f} kN",25,MUTED)
        for y,s in [(850,'180 cm / 88 kg; one condition per exercise.'),(904,'Jump: measured-motion reference, propulsion only.'),
            (958,'Squat: prescribed motion; 0.90 s ascent.'),(1012,'Values: muscle-group sum, mean per side.'),
            (1066,'Model estimates; not activation or percent maximum.'),(1120,'Same motions replayed at 1x; landing not analyzed.'),
            (1225,'Next comparison: change jump height and countermovement.')]:label(d,(54,y),s,25,MUTED)
        return im

    def credits():
        im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im)
        label(d,(54,75),'MODEL / REFERENCES',38,FG,True)
        lines=[('Adapted MyoSim myofullbody / MuJoCo',29),('Custom mechanics and conditional muscle-force sharing',24),
            ('Shoulder paths: Seth et al. (2019), adapted via RMR',24),('10.3389/fnbot.2019.00090',23),
            ('CMJ data: Calisti, Mohr & Federolf (2025)',27),('10.6084/m9.figshare.28890545',23),
            ('Squat comparison: Goodman et al. (2023)',27),('10.1519/JSC.0000000000004506',23),
            ('CMJ tendon comparison: Wade et al. (2018)',27),('10.1038/s41598-018-20387-0',23),
            ('BodyParts3D / DBCLS via MyoHub: CC BY 4.0',23),
            ('Z-Anatomy: Kervyn, Zielinski & Vinent: CC BY-SA 4.0',23),
            ('Supplementary BP3D / DBCLS: CC BY-SA 2.1 JP',23),
            ('Anatomy registration, posing and colours adapted.',23),
            ('AI agent assisted preparation. Profile art is illustrative.',23)]
        y=180
        for text,size_ in lines:
            assert d.textbbox((0,0),text,font=font(size_))[2]<972
            label(d,(54,y),text,size_,MUTED);y+=67
        return im

    info,credit=summary(),credits();info.save(OUT/'reel_summary.png');credit.save(OUT/'reel_sources.png')
    cover=motion_frame(1.016,poster=True);cover.save(OUT/'reel_cover.png')
    path=OUT/'jump_vs_100kg_squat_reel.mp4';writer=imageio_ffmpeg.write_frames(str(path),(1080,1920),fps=FPS,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart'])
    writer.send(None)
    try:
        for frame in range(DURATION*FPS):
            t=frame/FPS;im=motion_frame(t) if t<18 else info if t<22 else credit
            assert np.all(np.asarray(im)[1536:]==BG)
            writer.send(np.asarray(im))
            if frame in [90,240,360,480]:im.save(OUT/f'reel_preview_{frame:03d}.png')
    finally:writer.close()
    count,duration=imageio_ffmpeg.count_frames_and_secs(str(path));assert count==DURATION*FPS
    meta=dict(status='prepared_model_based_reel',file=path.name,width=1080,height=1920,fps=FPS,frames=count,duration_s=duration,
        language='English',bottom_reserved_px=384,motion_duration_s=18,summary_duration_s=4,source_duration_s=3,
        rendering=dict(shared_camera=True,shared_scale=True,common_crop=crop),
        comparison='Peak conditional group tension during jump propulsion / squat ascent; mean per side.',
        ratio_denominator='same muscle group, 100 kg squat peak; not muscle capacity',metrics=records,
        additional_jump_heights_solved=False,landing_analysis_included=False,physical_force_values_changed=False,
        replays='Original physical time, independent exercise clocks; short replay cut, no simulated landing or fatigue.',
        physics_provenance=json.loads((OUT/'physics_provenance.json').read_text()))
    (OUT/'reel_metrics.json').write_text(json.dumps(meta,indent=2))
    shutil.copy2(ROOT/'results/power_comparison/CREDITS.md',OUT/'CREDITS.md')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Biomechanics launch</title><style>body{background:#111b29;color:#e7eef7;font:17px/1.6 system-ui;max-width:960px;margin:28px auto;padding:18px}a{color:#dca677}img{max-width:100%;display:block}video{max-height:760px;max-width:100%;display:block}section{margin:28px 0}.avatar{width:220px;border-radius:50%}h1,h2{font-weight:500}pre{white-space:pre-wrap;background:#192737;padding:20px;border-radius:8px}</style><h1>Biomechanics_ksm</h1><img class="avatar" src="profile_minimal.png" alt="글자 없는 인체 동작 프로필"><section><h2>첫 릴스</h2><video controls playsinline preload="metadata" src="jump_vs_100kg_squat_reel.mp4" poster="reel_cover.png"></video></section>'''
    for filename,title in [('profile_bio_ko.txt','프로필 소개'),('intro_post_ko.txt','짧은 소개 글'),('reel_caption_ko.txt','영상과 함께 올릴 글')]:
        import html
        page+=f'<section><h2>{title}</h2><pre>{html.escape((OUT/filename).read_text(encoding="utf-8-sig"))}</pre></section>'
    page+='<p><a href="source_links.txt">출처</a> · <a href="reel_metrics.json">계산값과 영상 범위</a> · <a href="profile_generation_prompt.txt">프로필 이미지 생성 프롬프트</a></p></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print('FIRST_REEL_READY',json.dumps(records),flush=True)


if __name__=='__main__':main()
