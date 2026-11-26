"""Portrait review of the checked 240 kg squat; no athlete validation claim."""
from pathlib import Path
import json,hashlib,math,argparse
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from launch_relative_render_palette import color
from case_contours import joint_rgb
from build_motion_study_reel import anatomy_icon

ROOT=Path(__file__).resolve().parents[1];CASE=ROOT/'results/power_comparison/yun_sungbin/squat_240';OUT=ROOT/'deliverables/yun_sungbin_study'
BG=(13,23,38);FG=(237,243,251);MUTED=(166,188,209);COL=[(244,157,89),(80,198,186),(139,158,245)]
def font(n,b=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf'if b else'C:/Windows/Fonts/arial.ttf',n)
def txt(d,xy,s,n=28,c=FG,b=False):d.text(xy,s,font=font(n,b),fill=c)
def main():
    global CASE
    parser=argparse.ArgumentParser();parser.add_argument('--folder');parser.add_argument('--output-name',default='squat_240_preview');parser.add_argument('--speed',type=float,default=.75);args=parser.parse_args()
    if args.folder:CASE=ROOT/args.folder
    OUT.mkdir(exist_ok=True);meta=json.loads((CASE/'summary.json').read_text());assert meta['all_mechanics_gates_passed']
    jump=meta.get('movement')=='cmj_replay';motion=np.load(CASE/'motion.npz');audit=json.loads((CASE/'verification.json').read_text())if jump else None
    v=json.loads((CASE/'analysis/visualization/data.json').read_text());z=np.load(CASE/'analysis/tensions.npz');times=z['times'];forces=z['force_N']
    keys=[('quadriceps','QUADS'),('gluteus','GLUTES'),('calf','CALVES')]
    # Use the explicitly released group definitions rather than mesh labels.
    unique=sorted(set(g['key']for g in v['groups']));print('AVAILABLE_GROUPS',unique,flush=True)
    aliases=[(['quadriceps'],'QUADS'),(['gluteals'],'GLUTES'),(['gastrocnemius','soleus'],'CALVES')]
    traces=[]
    for options,name in aliases:
        gs=[g for g in v['groups']if g['key']in options]
        assert len(gs)==2*len(options),(name,unique)
        ids=sorted({i for g in gs for i in g['indices']});traces.append(forces[:,ids].sum(1)/2/1000)
    traces=np.array(traces);upper=max(1.,math.ceil(traces.max()/5)*5)
    files=sorted((CASE/'study_render').glob('frame_*.png'));expected=round(meta['phase_times_s']['end']*30)+1;assert len(files)==expected,(len(files),expected)
    bounds=[];view_bounds=[[],[]]
    for file in files:
        im=Image.open(file).convert('RGBA');alpha=np.asarray(im)[:,:,3];yy,xx=np.where(alpha>8)
        assert xx.min()>2 and xx.max()<1597 and yy.min()>2 and yy.max()<857,('Clipped',file)
        bounds.append((xx.min(),yy.min(),xx.max()+1,yy.max()+1))
        for view,(left,right)in enumerate([(0,800),(800,1600)]):
            vy,vx=np.where(alpha[:,left:right]>8);assert len(vx)>0
            view_bounds[view].append((vx.min()+left,vy.min(),vx.max()+left+1,vy.max()+1))
    crops=[];sizes=[]
    for view in range(2):
        b=np.array(view_bounds[view]);left=view*800;right=left+800
        crop=(max(left,int(b[:,0].min())-5),max(0,int(b[:,1].min())-5),min(right,int(b[:,2].max())+5),min(860,int(b[:,3].max())+5));crops.append(crop)
        ratio=min(([630,300][view])/(crop[2]-crop[0]),([650,530][view])/(crop[3]-crop[1]));sizes.append((round((crop[2]-crop[0])*ratio),round((crop[3]-crop[1])*ratio)))
    images=[]
    for file in files:
        image=Image.open(file).convert('RGBA');images.append([image.crop(crop).resize(size,Image.Resampling.LANCZOS)for crop,size in zip(crops,sizes)])
    fps=30;speed=args.speed;cycle=meta['phase_times_s']['end']/speed;duration=cycle*3+3;nframes=round(duration*fps)
    writer=imageio_ffmpeg.write_frames(str(OUT/(args.output_name+'.mp4')),(1080,1920),fps=fps,codec='libx264',quality=8,macro_block_size=2,output_params=['-pix_fmt','yuv420p','-movflags','+faststart']);writer.send(None)
    start,end=meta['phase_times_s']['start_ascent'],meta['phase_times_s']['end_ascent'];sel=(times>=start)&(times<=end)
    metrics={}
    for (_,name),trace in zip(aliases,traces):metrics[name]=dict(peak_whole_motion_N=float(trace.max()*1000),peak_ascent_N=float(trace[sel].max()*1000),ascent_integral_Ns=float(np.trapezoid(trace[sel]*1000,times[sel])),per_leg=True)
    for k in range(nframes):
        t=k/fps;summary=t>=3*cycle;source=min(meta['phase_times_s']['end'],(t%cycle)*speed)if not summary else meta['phase_times_s']['end']
        im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);txt(d,(55,70),'JUMP VALIDATION'if jump else'240 kg SQUAT',64,b=True);txt(d,(58,157),'178 cm / 87 kg',30,c=MUTED)
        txt(d,(840,165),f'{speed:g}x',28,c=MUTED)
        idx=min(len(images)-1,round(source*30))
        for view,(image,size)in enumerate(zip(images[idx],sizes)):
            centre=[350,875][view];im.paste(image,(centre-size[0]//2,260+(650-size[1])//2),image)
        d=ImageDraw.Draw(im)
        if not summary:
            txt(d,(60,965),'MUSCLE TENSION',34,b=True);txt(d,(755,974),'kN / leg',25,c=MUTED)
            x0,y0,w,h=95,1060,880,240
            for level in range(3):
                y=y0+h-level*h/2;d.line((x0,y,x0+w,y),fill=(49,65,84),width=2);txt(d,(33,y-13),f'{upper*level/2:g}',22,c=MUTED)
            for i,trace in enumerate(traces):
                pts=[(x0+float(tt/times[-1])*w,y0+h-float(f/upper)*h)for tt,f in zip(times,trace)];d.line(pts,fill=COL[i],width=4)
                y=y0+h-float(np.interp(source,times,trace)/upper)*h;x=x0+source/times[-1]*w;d.ellipse((x-7,y-7,x+7,y+7),fill=COL[i])
            x=x0+source/times[-1]*w;d.line((x,y0-8,x,y0+h+8),fill=FG,width=2)
            for i,(_,name)in enumerate(aliases):
                anatomy_icon(d,60+i*305,1334,[1,0,3][i],scale=.42)
                txt(d,(95+i*305,1345),name,28,c=COL[i],b=True)
        else:
            if jump:
                txt(d,(60,970),'COM RISE',34,b=True);txt(d,(60,1050),f"{audit['checks']['airborne_COM_rise_m']*100:.0f} cm",72,b=True)
                txt(d,(65,1180),'GROUND + FREE FLIGHT',30,c=MUTED);txt(d,(65,1250),'Landing not simulated',28,c=MUTED)
            else:
                txt(d,(60,970),'ASCENT',34,b=True);txt(d,(60,1050),f"{meta['peak_COM_vertical_power_W']/1000:.2f} kW",72,b=True)
                txt(d,(65,1140),'PEAK COM POWER',28,c=MUTED);txt(d,(65,1240),'1s down / 2s up',36)
        for x in range(240):d.line((60+x,1455,60+x,1470),fill=tuple(int(round(c))for c in color(x/239)),width=1)
        txt(d,(320,1450),'Muscle: 0-30 kN',22,c=MUTED)
        for x in range(120):d.line((650+x,1455,650+x,1470),fill=tuple(int(c)for c in joint_rgb(x/119*20000)),width=1)
        txt(d,(790,1450),'Joint: 0-20 kN',22,c=MUTED)
        txt(d,(60,1503),'MyoSim / MuJoCo / Seth et al. 2019'if summary else'Synthetic motion / Force accuracy unvalidated'if jump else'Prescribed motion / Conditional model estimate',21,c=MUTED)
        if k in [0,round(cycle*.5*fps),nframes-1]:im.save(OUT/f'{args.output_name}_{k:04d}.png')
        writer.send(np.asarray(im))
    writer.close()
    report=dict(status='validation_jump_review_only'if jump else'squat_review_only_jump_comparison_pending',fps=fps,frames=nframes,width=1080,height=1920,duration_s=nframes/fps,source_motion_sha256=hashlib.sha256((CASE/'motion.npz').read_bytes()).hexdigest(),source_tension_sha256=hashlib.sha256((CASE/'analysis/tensions.npz').read_bytes()).hexdigest(),source_render_frame_count=len(files),metrics=metrics,all_mechanics_gates_passed=True,human_muscle_force_validated=False)
    (OUT/(args.output_name+'_checks.json')).write_text(json.dumps(report,indent=2));print('PREVIEW_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
