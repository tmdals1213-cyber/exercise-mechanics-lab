"""Portrait anatomy, restricted tension contour and separate joint palette."""
import argparse,json,shutil,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from jump_rig import ROOT

OUT=ROOT/os.environ.get('EXERCISE_JUMP_VISUALS','results/jump_contact_visualization');BG=(13,23,38);FG=(237,243,251);MUTED=(166,188,209)
STOPS=np.log1p([0,100,1000,10000]);COLORS=np.array([[43,97,142],[48,163,169],[240,183,78],[238,82,69]])
def rgb(value):
    x=np.log1p(np.maximum(value,0));return np.stack([np.interp(x,STOPS,COLORS[:,i]) for i in range(3)],axis=-1).astype(np.uint8)
def font(n,bold=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',n)
def text(draw,xy,s,n=24,color=FG,bold=False):draw.text(xy,s,font=font(n,bold),fill=color)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--poster',action='store_true');a=ap.parse_args();d=json.loads((OUT/'data.json').read_text());projections=json.loads((OUT/'projections.json').read_text());ts=np.array(d['times']);N=len(ts);groups=d['groups'];F=np.array(d['group_tension_N']);joint=np.array(d['joint_force_N']);rows=[('Gluteals',['gluteals']),('Quadriceps',['quadriceps']),('Hamstrings',['hamstrings']),('Calf',['gastrocnemius','soleus']),('Abdominal wall',['abdominals']),('Lumbar / back',['spinal_extensors','quadratus_lumborum']),('Latissimus',['latissimus']),('Trapezius',['trapezius'])]
    chart=[]
    for label,keys in rows:chart.append((label,F[[i for i,g in enumerate(groups) if g['key'] in keys]].sum(0)/2))
    out=OUT/'portrait_frames';out.mkdir(exist_ok=True);frames=d['preview_indices'] if a.poster else range(N);poster=d['poster_index']
    for frame in frames:
        base=np.array(Image.open(OUT/f'anatomy/base/frame_{frame:04d}.png').convert('RGBA'));mask=np.array(Image.open(OUT/f'anatomy/mask/frame_{frame:04d}.png').convert('RGB')).astype(int);ids=np.rint(mask[:,:,0]/4).astype(int);valid=(abs(mask[:,:,1]-(255-mask[:,:,0]))<5)&(abs(mask[:,:,2]-128)<5)&(ids>0)&(ids<=len(groups));lut=np.vstack([np.zeros(3),rgb(F[:,frame])]);wanted=lut[np.clip(ids,0,len(groups))];lum=base[:,:,:3].mean(2)/255;shade=np.clip(.35+.85*lum,.35,1.15);tinted=np.clip(wanted*shade[:,:,None],0,255);base[:,:,:3][valid]=(.15*base[:,:,:3]+.85*tinted)[valid].astype(np.uint8)
        jvalid=(abs(mask[:,:,1]-(255-mask[:,:,0]))<5)&(abs(mask[:,:,2]-32)<5);jid=np.clip(np.rint(mask[:,:,0]/16).astype(int),0,len(d['joints']));jvalues=np.r_[0,joint[:,frame]];u=np.clip(jvalues[jid]/6000,0,1);jr=np.zeros_like(wanted);jr[:,:,0]=8+237*u;jr[:,:,1]=8+22*u;jr[:,:,2]=10+20*u;jt=np.clip(jr*shade[:,:,None],0,255).astype(np.uint8);base[:,:,:3][jvalid]=jt[jvalid]
        im=Image.new('RGB',(1080,1920),BG);layer=Image.fromarray(base).resize((960,800),Image.Resampling.LANCZOS);im.paste(layer,(60,150),layer);draw=ImageDraw.Draw(im)
        text(draw,(54,30),'VERTICAL JUMP',44,bold=True);text(draw,(54,88),'Conditional tension  /  joint reaction',25);text(draw,(54,124),f'{d.get("height_m",1.8)*100:.0f} cm  ·  {d.get("mass_kg",90):.0f} kg  ·  '+('measured-motion replay' if d.get('measured_replay') else 'muscle sensitivity pending'),21,MUTED)
        phase='READY' if ts[frame]<.25 else 'COUNTERMOVEMENT' if ts[frame]<d['countermovement_end_s'] else 'PROPULSION' if ts[frame]<d['takeoff_s'] else 'FLIGHT';text(draw,(54,176),phase,27,(140,226,213),True);text(draw,(760,176),f'{ts[frame]:.2f} s',27);text(draw,(760,90),f'GRF {np.array(d["GRF_N"])[frame,2]:,.0f} N',23,(100,205,224))
        text(draw,(190,935),'FRONT OBLIQUE',19,MUTED);text(draw,(752,935),'POSTERIOR',19,MUTED)
        text(draw,(54,966),'Muscle-group tension [N]',23,bold=True)
        for j,(value,label) in enumerate([(0,'0'),(100,'100'),(1000,'1,000'),(10000,'10,000')]):
            x=54+j*129;draw.rectangle((x,1000,x+24,1015),fill=tuple(rgb(value)));text(draw,(x+31,994),label,19,MUTED)
        text(draw,(638,966),'Joint region reaction [N]',23,bold=True);draw.rectangle((638,1000,665,1015),fill=(8,8,10),outline=(80,90,105));text(draw,(674,994),'0',19,MUTED);draw.rectangle((754,1000,781,1015),fill=(245,30,30));text(draw,(792,994),'6,000+',19,MUTED)
        text(draw,(54,1060),'TENSION OVER TIME',25,bold=True);text(draw,(320,1064),'Mean per side',18,MUTED);text(draw,(867,1064),'MEAN / PEAK',18,MUTED)
        x0=320;w=492;top=1111;rh=36
        for k,(label,series) in enumerate(chart):
            y=top+k*rh;text(draw,(54,y+1),label,22);heat=Image.fromarray(rgb(series)[None,:,:]).resize((w,25),Image.Resampling.NEAREST);im.paste(heat,(x0,y));mean=np.trapezoid(series,ts)/(ts[-1]-ts[0]);text(draw,(850,y+2),f'{mean:,.0f} / {series.max():,.0f}',20,MUTED)
        cursor=x0+w*(ts[frame]-ts[0])/(ts[-1]-ts[0]);draw.line((cursor,top-9,cursor,top+len(chart)*rh-9),fill=FG,width=3);text(draw,(x0,1404),f'{ts[0]:.1f} s',18,MUTED);text(draw,(x0+w-50,1404),f'{ts[-1]:.2f} s',18,MUTED)
        text(draw,(54,1440),'JOINT REACTION  ·  MEAN / PEAK [N]',20,MUTED,True)
        for k,key in enumerate(['shoulder','elbow','wrist','hip','knee','ankle']):
            values=joint[[i for i,j in enumerate(d['joints']) if j['key'].startswith(key+'_')]].mean(0);x=54+(k%3)*332;y=1472+(k//3)*28;mean=np.trapezoid(values,ts)/(ts[-1]-ts[0]);text(draw,(x,y),f'{key.title()} {mean:,.0f} / {values.max():,.0f}',19,(231,151,147))
        if not np.all(np.array(im)[1536:]==BG):raise RuntimeError('Unsafe bottom overlay')
        im.save(out/f'frame_{frame:04d}.png')
        if frame==poster:im.save(OUT/'poster.png');im.crop((0,0,1080,1536)).save(OUT/'preview.png')
        print('CONTACT_PORTRAIT',frame,flush=True)
    if a.poster:return
    writer=imageio_ffmpeg.write_frames(str(OUT/'jump_tension_portrait.mp4'),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=None,macro_block_size=1,output_params=['-crf','18','-preset','medium','-movflags','+faststart']);writer.send(None);count=0
    try:
        # 2x slower display, three clearly separated replays of one calculation.
        for replay in range(3):
            for frame in range(N):
                pixels=np.array(Image.open(out/f'frame_{frame:04d}.png').convert('RGB'))
                for _ in range(2):writer.send(pixels);count+=1
            end=np.array(Image.open(out/f'frame_{N-1:04d}.png').convert('RGB'))
            for _ in range(12):writer.send(end);count+=1
    finally:writer.close()
    actual,seconds=imageio_ffmpeg.count_frames_and_secs(str(OUT/'jump_tension_portrait.mp4'));assert actual==count
    shutil.copyfile(OUT/'jump_tension_portrait.mp4',OUT/'jump_tension_upright_v3.mp4')
    (OUT/'video_checks.json').write_text(json.dumps(dict(status='passed',size=[1080,1920],fps=30,frames=count,duration_s=seconds,bottom_reserved_px=384,unique_physics_samples=N,display_speed=.5,replays=3,landing_included=False,muscle_palette='logarithmic absolute group tension N',joint_palette='visible articular bone surfaces, black-to-red absolute reaction N; no point markers'),indent=2))
    print('CONTACT_VIDEO_COMPLETE',count,seconds)

if __name__=='__main__':main()
