"""Portrait draft from real anatomy renders, checkpoints, failures and traces."""
from pathlib import Path
from functools import lru_cache
import argparse,json,math
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
from jump_monitor_data import ROOT,read,atomic
from build_motion_study_reel import font,BG,FG,MUTED,ACC

OUT=ROOT/'deliverables/jump_rl_study_draft_v1';MINT=(119,218,199);RED=(238,135,150);BLUE=(113,185,220);DIM=(35,55,74);GREY=(107,131,151)

def txt(d,x,y,s,size=24,c=FG,b=False,anchor=None):d.text((x,y),s,font=font(size,b),fill=c,anchor=anchor)
def ease(u):u=float(np.clip(u,0,1));return u*u*(3-2*u)

class Draft:
    def __init__(self):
        self.m=read(OUT/'source_manifest.json');self.metrics=read(OUT/'metrics.json');self.cases={c['key']:c for c in self.m['cases']};self.run=Path(self.m['run']);self.config=read(self.run/'definition.json');self.hist=self.m['training_history'];self.hero=self.cases['best_checked'];self.data={};self.render_counts={}
        for key,case in self.cases.items():
            folder=Path(case['folder']);self.data[key]=dict(z=np.load(folder/'motion.npz'),r=read(folder/'control_summary.json'),cam=read(folder/'fixed_side_camera_v2.json'));directory=folder/case.get('render_dir','fixed_side_render_v2');files=sorted(directory.glob('frame_*.png'));assert files,(key,directory);self.render_counts[key]=(min(int(p.stem.split('_')[-1])for p in files),max(int(p.stem.split('_')[-1])for p in files));expected=round(case['duration_s']*case['render_fps']);assert self.render_counts[key][1]==expected,(key,expected,self.render_counts[key])
        self.series={key:np.load(OUT/f'{key}_metrics.npz')for key in ['initial','best_checked']};self.metric_keys=['ground','hip','knee','ankle'];self.names=['GROUND','HIP','KNEE','ANKLE'];self.units=['BW','Nm','Nm','Nm'];self.curve_limits={key:max(self.metrics['cases'][c]['peak_push'][key]for c in self.series)*1.05 for key in self.metric_keys}
        self.nodes_x=[610,715,820,925];self.nodes_y=[np.linspace(183,323,n)for n in [9,8,8,3]];self.segments=[];clock=0.
        def add(kind,duration,**details):
            nonlocal clock
            self.segments.append(dict(kind=kind,start_s=clock,end_s=clock+duration,**details));clock+=duration
        first=self.cases['first_fall'];J=first['duration_s']-first['render_start_s'];heroJ=self.hero['duration_s']-max(0.,self.data['best_checked']['r']['parameters']['ready_s']-.1)
        add('hook',3.);add('setup',2.6);add('cycle',.4+J+.65+.65,case='first_fall')
        montage=[('short_fall',1.3),('middle_fall',1.3),('other_fall',1.3),('late_fall',1.75),('long_fall',1.3),('improve_230',1.4),('improve_246',1.7)];montage.sort(key=lambda pair:self.cases[pair[0]]['record']['training_trial']);reseed_episode=(self.m.get('reseed_stage')or{}).get('episode',768)
        for key,duration in montage:
            if self.cases[key]['record']['training_trial']<=reseed_episode:add('montage',duration,case=key)
        add('reseed',1.35)
        for key,duration in montage:
            if self.cases[key]['record']['training_trial']>reseed_episode:add('montage',duration,case=key)
        add('hero',heroJ+.35,speed=1.);add('hero',heroJ/.65+.35,speed=.65);add('comparison',4.5);add('open_ending',1.6);self.duration=clock
        atomic(OUT/'timeline.json',dict(duration_s=clock,fps=30,segments=self.segments,hook='CAN AI FIND / THE BEST JUMP?',end='SEARCH CONTINUES',final_conclusion_available=False,baseline_m=self.m['baseline_retained_m'],best_checked_m=self.m['best_checked_min_retained_m'],four_comparison_signals=self.metric_keys))
    @lru_cache(maxsize=30)
    def body_image(self,key,fi):
        case=self.cases[key];file=Path(case['folder'])/case.get('render_dir','fixed_side_render_v2')/f'frame_{fi:04d}.png'
        with Image.open(file)as source:return source.convert('RGBA').crop((0,190,1080,1015))
    @lru_cache(maxsize=12)
    def iso_image(self,fi):
        with Image.open(Path(self.hero['folder'])/'study_render'/f'frame_{fi:04d}.png')as source:return source.convert('RGBA').crop((200,60,650,810)).resize((756,1260),Image.Resampling.LANCZOS)
    def base(self):
        im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);txt(d,45,24,'@biomechanics_ksm',20,MUTED,True);return im,d
    @lru_cache(maxsize=40)
    def network(self,batch,context,after=False):
        before=self.run/f'network_update_{batch:04d}_before.npz'
        path=self.run/f'network_update_{batch:04d}_after.npz'if after else before if before.exists()else self.run/'network_update_0000_before.npz'if batch==1 else self.run/f'network_update_{batch-1:04d}_after.npz'
        state=np.load(path);obs=np.array(self.config['physical_observation']+[context,2.85/3.]);w=[state[f'mlp_extractor.policy_net.{i}.weight']for i in [0,2]]+[state['action_net.weight']];b=[state[f'mlp_extractor.policy_net.{i}.bias']for i in [0,2]]+[state['action_net.bias']];h1=np.tanh(w[0]@obs+b[0]);h2=np.tanh(w[1]@h1+b[1]);mean=w[2]@h2+b[2]
        items=self.config['actions'];groups=[[i for i,x in enumerate(items)if x['kind']=='posture_timing'and x['unit']in ['rad','m']],[i for i,x in enumerate(items)if x['kind']=='posture_timing'and x['unit']not in ['rad','m']],list(range(37,52))]
        levels=[obs,np.mean(h1.reshape(8,8),axis=1),np.mean(h2.reshape(8,8),axis=1),np.array([mean[g].mean()for g in groups])];edges=[w[0].reshape(8,8,9).mean(1),w[1].reshape(8,8,8,8).mean((1,3)),np.array([w[2][g].reshape(len(g),8,8).mean((0,2))for g in groups])]
        return levels,edges
    def graph(self,d,batch=1,context=1.,pulse=-1.,back=False,after=False):
        batch=min(batch,self.m['actual_completed_PPO_updates']);levels,edges=self.network(batch,round(context,5),after)
        for k,matrix in enumerate(edges):
            for j in range(matrix.shape[0]):
                for h in range(matrix.shape[1]):
                    value=float(matrix[j,h]);a=(self.nodes_x[k],self.nodes_y[k][h]);b=(self.nodes_x[k+1],self.nodes_y[k+1][j]);color=(48,79,96)if abs(value)>.025 else(29,46,63);d.line((*a,*b),fill=color,width=1)
                    p=pulse*3-(2-k if back else k)
                    if 0<=p<=1 and(h+j)%3==0:
                        if back:p=1-p
                        x=a[0]+(b[0]-a[0])*p;y=a[1]+(b[1]-a[1])*p;d.ellipse((x-2,y-2,x+2,y+2),fill=ACC if back else MINT)
        for k,values in enumerate(levels):
            for j,value in enumerate(values):
                x=self.nodes_x[k];y=self.nodes_y[k][j];v=float(np.clip(abs(value),0,1));a=(48,74,92);b=MINT if value>=0 else ACC;color=tuple(round(u+(v0-u)*v)for u,v0 in zip(a,b));r=5 if k<3 else 7;d.ellipse((x-r,y-r,x+r,y+r),fill=color)
        for x,s in zip(self.nodes_x,['9','64','64','58']):txt(d,x,345,s,15,MUTED,anchor='mt')
        for y,s in zip(self.nodes_y[-1],['POSE','TIME','EFFORT']):txt(d,940,y-7,s,14,MUTED,True)
    def world(self,im,key,t,show_distance=True):
        case=self.cases[key];D=self.data[key];z=D['z'];r=D['r'];cam=D['cam'];T=case['duration_s'];n=round(T*case['render_fps']);fi=int(np.clip(round(t/T*n),*self.render_counts[key]));native_time=fi/n*T;body=self.body_image(key,fi);im.paste(body,(0,350),body);d=ImageDraw.Draw(im);ppm=1080/cam['ortho_width_m'];zero=540-cam['camera_center_forward_m']*ppm;ground=350+525+cam['camera_center_z_m']*ppm-190;d.line((35,ground,1045,ground),fill=(96,118,138),width=2)
        for meter in np.arange(0,4.01,.5):
            x=zero+meter*ppm
            if 35<x<1045:d.line((x,ground-5,x,ground+12),fill=MUTED,width=2);txt(d,x,ground+22,f'{meter:.1f}',18,MUTED,anchor='mt')
        d.line((zero,ground-28,zero,ground+15),fill=FG,width=3)
        contact=native_time>=r['touchdown_s'];body_contact=r.get('first_body_floor_contact_s');fell=body_contact is not None and native_time>=body_contact
        if contact:
            x=zero+r['first_contact_heel_distance_m']*ppm;d.line((zero,ground,x,ground),fill=RED if fell else ACC,width=4)
            ids=np.flatnonzero((z['times']>=r['takeoff_s'])&(z['times']<=r['touchdown_s']));length=max(1,round(ease((native_time-r['touchdown_s'])/.55)*len(ids)));points=[]
            for v in z['com'][ids[:length]]:points.append((540+(-v[1]+cam['initial_foremost_sole_y_m']-cam['camera_center_forward_m'])*ppm,350+525-(v[2]-cam['camera_center_z_m'])*ppm-190))
            if len(points)>1:d.line(points,fill=ACC,width=2)
        if show_distance:
            if fell:outcome='FALL';c=RED
            elif case['record']['stable']and native_time>=T-.045:outcome='STABLE';c=MINT
            else:outcome='LANDING'if contact else'FLIGHT'if native_time>=r['takeoff_s']else'PUSH-OFF';c=MUTED
            txt(d,45,1025,outcome,24,c,True)
            if contact:
                value=case['record']['retained_distance_m']if outcome=='STABLE'else r['first_contact_heel_distance_m'];txt(d,1035,1020,f'{value:.2f} m',35,c,True,anchor='rt')
        return native_time
    def phase(self,key,t):return np.interp(t,self.metrics['cases'][key]['phase_times'],[0,.10,.32,.50,.74,1.])
    def traces(self,d,t):
        times=self.metrics['cases']['best_checked']['phase_times'];progress=float(np.clip((t-times[2])/(times[3]-times[2]),0,1))
        for i,key in enumerate(self.metric_keys):
            x=45+(i%2)*510;y=1110+(i//2)*188;w=480;d.rounded_rectangle((x,y,x+w,y+166),radius=11,fill=(18,33,49));txt(d,x+15,y+12,self.names[i],20,MUTED,True);value=self.metrics['cases']['best_checked']['peak_push'][key];txt(d,x+w-130,y+22,'PEAK',12,MUTED,anchor='rt');txt(d,x+w-52,y+6,f'{value:.1f}'if i==0 else f'{value:.0f}',29,MINT,True,anchor='rt');txt(d,x+w-14,y+18,self.units[i],15,MUTED,anchor='rt')
            left=x+14;right=x+w-14;top=y+66;bottom=y+146;limit=self.curve_limits[key]
            for source,color in [('initial',GREY),('best_checked',MINT)]:
                data=self.series[source];a,b=self.metrics['cases'][source]['phase_times'][2:4];grid=np.linspace(0,1,170);vals=np.interp(a+grid*(b-a),data['times'],data[key]);points=[(left+u*(right-left),bottom-v/limit*(bottom-top))for u,v in zip(grid,vals)];d.line(points,fill=color if source=='initial'else(44,83,88),width=2)
                if source=='best_checked':
                    n=max(2,round(progress*(len(points)-1))+1);d.line(points[:n],fill=MINT,width=3)
            cursor=left+progress*(right-left);d.line((cursor,top,cursor,bottom+3),fill=ACC,width=2)
    def progress(self,d,trial,box=(45,1170,1035,1435)):
        left,top,right,bottom=box;points=[];best=self.m['baseline_retained_m'];limit=min(trial,len(self.hist));lo=1.8;hi=3.65
        d.line((left,bottom,right,bottom),fill=DIM,width=2)
        for i,row in enumerate(self.hist[:limit]):
            if not row['eligible']:continue
            distance=row['retained_distance_m']if row['stable']else row['distance_m'];x=left+(right-left)*i/max(1,len(self.hist)-1);y=bottom-(distance-lo)/(hi-lo)*(bottom-top)
            if top<=y<=bottom:d.ellipse((x-2,y-2,x+2,y+2),fill=MINT if row['stable']else RED)
            if row['stable']:best=max(best,row['retained_distance_m'])
            points.append((x,bottom-(best-lo)/(hi-lo)*(bottom-top)))
        if len(points)>1:d.line(points,fill=ACC,width=3)
        txt(d,left,bottom+16,f'{trial} TRIALS',21,MUTED);txt(d,right,bottom+16,f'SAMPLED {best:.2f} m',22,ACC,True,anchor='rt')
    def render(self,segment,local):
        kind=segment['kind'];im,d=self.base();duration=segment['end_s']-segment['start_s']
        if kind=='hook':
            t=min(3.,max(0.,self.data['best_checked']['r']['parameters']['ready_s']-.1)+local*.8);fi=min(90,round(t*30));body=self.iso_image(fi);im.paste(body,(160,300),body);d=ImageDraw.Draw(im);txt(d,45,82,'CAN AI FIND',70,FG,True);txt(d,45,163,'THE BEST JUMP?',70,FG,True);txt(d,45,278,'BIOMECHANICS × REINFORCEMENT LEARNING',24,MINT,True);txt(d,45,1430,'STANDING LONG JUMP',27,MUTED,True)
        elif kind=='setup':
            self.world(im,'initial',0.,False);d=ImageDraw.Draw(im);txt(d,45,78,'ONE BODY.',48,FG,True);txt(d,45,137,'FIXED LIMITS.',48,FG,True);self.graph(d,1,0.,pulse=(local/.8)%1 if local<.8 else-1)
            for x,w,label in [(45,280,'178 CM / 87 KG'),(340,315,'FIXED TORQUE'),(670,365,'58 PARAMETERS')]:d.rounded_rectangle((x,1120,x+w,1190),radius=12,fill=DIM);txt(d,x+w/2,1141,label,23,FG,True,anchor='mt')
            txt(d,45,1300,'DISTANCE + LANDING',31,MINT,True);txt(d,45,1420,'MODEL  →  SIMULATE  →  PPO',28,MUTED,True)
        elif kind=='cycle':
            key='first_fall';case=self.cases[key];start=case['render_start_s'];J=case['duration_s']-start;t=min(case['duration_s'],start+max(0.,local-.4));self.world(im,key,t);d=ImageDraw.Draw(im);stage='TRY'if local<.4+J else'SCORE'if local<.4+J+.65 else'UPDATE';txt(d,45,80,'TRY. SCORE. UPDATE.',42,FG,True);txt(d,45,145,stage,30,ACC,True);after=stage=='UPDATE'and local>=duration-.18;self.graph(d,1,float(case['record'].get('context')or 0.),pulse=local/.4 if local<.4 else(local-(.4+J+.65))/.65 if stage=='UPDATE'else-1,back=stage=='UPDATE',after=after);self.progress(d,48 if stage!='TRY'else int(case['record']['training_trial']))
            if stage=='UPDATE':txt(d,45,1110,'48 TRIALS → 1 UPDATE',27,MINT,True)
        elif kind=='montage':
            case=self.cases[segment['case']];start=case['render_start_s'];u=min(1.,local/max(.1,duration-.15));t=start+u*(case['duration_s']-start);self.world(im,case['key'],t);d=ImageDraw.Draw(im);trial=int(case['record']['training_trial']);batch=(trial-1)//48+1;txt(d,45,80,'FAIL. LEARN. REPEAT.',42,FG,True);txt(d,45,145,'BATCH 01'if trial<=48 else f'TRIAL {trial}',28,MUTED,True);self.graph(d,batch,float(case['record'].get('context')or 0.));self.progress(d,max(48,trial))
        elif kind=='reseed':
            self.world(im,'improve_246',self.cases['improve_246']['duration_s']);d=ImageDraw.Draw(im);txt(d,45,80,'RESEED',58,ACC,True);txt(d,45,158,'SAME LIMITS',27,MUTED,True);self.graph(d,17,1.,pulse=ease(local/duration));self.progress(d,min(768,len(self.hist)));txt(d,45,1108,'SEARCH RESTART',26,MINT,True)
        elif kind=='hero':
            speed=segment['speed'];start=max(0.,self.data['best_checked']['r']['parameters']['ready_s']-.1);t=min(self.hero['duration_s'],start+local*speed);native_t=self.world(im,'best_checked',t);d=ImageDraw.Draw(im);txt(d,45,75,'BEST CHECKED',42,FG,True);txt(d,45,135,'SO FAR',25,MUTED,True);txt(d,45,181,f'{self.m["best_checked_min_retained_m"]:.2f} m',49,MINT,True);txt(d,1035,1050,f'{speed:g}×',19,MUTED,anchor='rt');self.graph(d,self.m['actual_completed_PPO_updates'],1.);self.traces(d,native_t);txt(d,45,1493,'PUSH-OFF PHASE',17,MUTED);txt(d,1035,1493,'Initial  /  Checked',17,MUTED,anchor='rt')
        elif kind=='comparison':
            txt(d,45,80,'PUSH-OFF',49,FG,True);txt(d,45,149,'FORCE + TORQUE',27,MUTED,True);old=self.metrics['cases']['initial']['peak_push'];new=self.metrics['cases']['best_checked']['peak_push'];u=ease(local/.65)
            for i,key in enumerate(self.metric_keys):
                y=280+i*250;txt(d,45,y,self.names[i],29,FG,True);txt(d,1035,y,f'{old[key]:.1f} → {new[key]:.1f} {self.units[i]}'if i==0 else f'{old[key]:.0f} → {new[key]:.0f} {self.units[i]}',35,MINT,True,anchor='rt');maximum=max(old[key],new[key])*1.12;left=45;width=990
                d.rounded_rectangle((left,y+64,left+width*old[key]/maximum*u,y+90),radius=8,fill=GREY);d.rounded_rectangle((left,y+108,left+width*new[key]/maximum*u,y+140),radius=8,fill=MINT);txt(d,1035,y+163,f'{(new[key]/old[key]-1)*100:+.0f}%',24,ACC,True,anchor='rt')
            txt(d,45,1320,'INITIAL  /  CHECKED',22,MUTED,True);txt(d,45,1390,'Right-side net torque magnitudes',20,MUTED);txt(d,45,1475,'MuJoCo · PPO | Anatomy: BodyParts3D/DBCLS · MyoHub',14,MUTED);txt(d,45,1501,'Z-Anatomy · BodyExplorer | Full credits supplied with draft',14,MUTED)
        else:
            self.world(im,'best_checked',self.hero['duration_s'],False);d=ImageDraw.Draw(im);txt(d,45,80,'SEARCH',65,FG,True);txt(d,45,157,'CONTINUES.',65,MINT,True);txt(d,45,1170,f'{self.m["actual_training_rollouts_used"]} TRIALS',32,MUTED,True);txt(d,45,1234,f'{self.m["actual_completed_PPO_updates"]} PPO UPDATES',32,MUTED,True);txt(d,45,1350,'STUDY IN PROGRESS',29,ACC,True);txt(d,45,1490,'@biomechanics_ksm',21,MUTED,True)
        assert np.all(np.asarray(im)[1536:]==BG),(kind,'footer overwritten');return im
    def at(self,clock):
        segment=next((s for s in self.segments if clock<s['end_s']),self.segments[-1]);return self.render(segment,max(0.,clock-segment['start_s']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args();draft=Draft();targets=[('hook',1.6),('study',4.),('learning',6.5),('failure',next(s['start_s']+.9 for s in draft.segments if s.get('case')=='long_fall')),('checked',next(s['start_s']+2 for s in draft.segments if s['kind']=='hero')),('comparison',next(s['start_s']+1.5 for s in draft.segments if s['kind']=='comparison'))]
    for name,t in targets:draft.at(t).save(OUT/f'preview_{name}.png')
    if args.preview:return
    video=OUT/'standing_long_jump_rl_draft.mp4';writer=imageio_ffmpeg.write_frames(str(video),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None);count=round(draft.duration*30)
    try:
        for k in range(count):writer.send(np.asarray(draft.at(k/30)))
    finally:writer.close()
    atomic(OUT/'draft_delivery.json',dict(status='draft_composed',video=str(video),frames=count,fps=30,duration_s=count/30,resolution=[1080,1920],reserved_footer_px=384,actual_failure_scenes=6,all_body_motions_from_distinct_native_trials=True,network_checkpoints_are_actual=True,network_graph_compressed=True,pulses_are_computation_schematic=True,individual_muscle_tensions_shown=False,actual_PPO_updates=draft.m['actual_completed_PPO_updates'],CEM_reseed_separate=True,final_conclusion='Not yet available; SEARCH CONTINUES ending.',force_comparison_scope=draft.metrics['peak_comparison_phase']))
    caption='Can reinforcement learning find a better standing long jump?\nOne virtual athlete. Fixed joint-torque limits.\nReal simulated trials, failed landings, and guided search restarts.\nWork in progress — showing simulated forces and net joint torques.\n';(OUT/'caption_en.txt').write_text(caption,encoding='utf-8');print('JUMP_RL_DRAFT_VIDEO_READY',video,count/30,flush=True)

if __name__=='__main__':main()


