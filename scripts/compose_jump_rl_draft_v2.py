"""Case-specific RL explanation, anatomical torque arrows, ghosts and four stages."""
from pathlib import Path
from functools import lru_cache
import argparse,math,json
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFilter
from scipy.interpolate import CubicSpline
from jump_monitor_data import ROOT,read,atomic
from compose_jump_rl_draft import Draft as Base,txt,ease,MINT,RED,BLUE,DIM,GREY
from build_motion_study_reel import BG,FG,MUTED,ACC
OUT=ROOT/'deliverables/jump_rl_study_draft_v2'
STAGE_COLORS=[GREY,BLUE,(239,192,108),MINT]

class Draft(Base):
    def __init__(self,partial=False):
        self.m=read(OUT/'source_manifest.json');self.metrics=read(OUT/'metrics.json');self.cases={c['key']:c for c in self.m['cases']};self.run=Path(self.m['run']);self.config=read(OUT/'policy_definition.json');self.hist=self.m['training_history'];self.hero=self.cases['best_checked'];self.data={};self.render_counts={};self.partial=partial
        for key,c in self.cases.items():
            f=Path(c['folder']);cam=read(f/'fixed_side_camera_v2.json',dict(ortho_width_m=4.6,camera_center_forward_m=c['camera_center_forward_m'],camera_center_z_m=1.15,initial_foremost_sole_y_m=-.0306183966));self.data[key]=dict(z=np.load(f/'motion.npz'),r=read(f/'control_summary.json'),cam=cam,j=np.load(f/'joint_overlay.npz'),names=read(f/'joint_names.json'));files=list((f/c['render_dir']).glob('frame_*.png'));expected=round(c['duration_s']*c['render_fps']);self.render_counts[key]=(round(c['render_start_s']*c['render_fps']),expected)
            if not partial:assert files and max(int(p.stem.split('_')[-1])for p in files)==expected,(key,len(files),expected)
        self.keys=self.m['final_four_keys'];self.series={k:np.load(OUT/f'{k}_metrics.npz')for k in self.keys};self.metric_keys=['ground','hip','knee','ankle'];self.names=['GRF','HIP','KNEE','ANKLE'];self.units=['BW','Nm','Nm','Nm'];self.limits={k:max(self.metrics['cases'][c]['full_cycle_peak'][k]for c in self.keys)*1.07 for k in self.metric_keys};self.nodes_x=[610,715,820,925];self.nodes_y=[np.linspace(183,323,n)for n in[9,8,8,3]];self.m['baseline_retained_m']=self.m['final_four_distances_m'][0];self.m['best_checked_min_retained_m']=self.m['final_four_distances_m'][-1]
        self.segments=[];clock=0.
        def add(kind,duration,**kw):
            nonlocal clock
            self.segments.append(dict(kind=kind,start_s=clock,end_s=clock+duration,**kw));clock+=duration
        add('hook',2.8);add('control',3.1);add('conditions',1.6);add('rl_example',7.2)
        clips=[c for c in self.m['cases']if c['role']=='montage']+[self.cases[k]for k in self.keys[1:]];clips.sort(key=lambda c:c['record']['training_trial']);reseed=False
        for c in clips:
            if c['record']['training_trial']>768 and not reseed:add('reseed',.7);reseed=True
            add('montage',.85 if c['record']['fell']else 1.05,case=c['key'])
        for index,key in enumerate(self.keys):add('final_stage',self.cases[key]['duration_s']+.2,case=key,index=index)
        add('four_stage_summary',3.2);self.duration=clock
        atomic(OUT/'timeline.json',dict(duration_s=clock,fps=30,segments=self.segments,final_four_keys=self.keys,final_four_distances_m=self.m['final_four_distances_m'],com_marker=False,postlanding_trajectory=True,ghosts=3,policy_example_updates=self.m['policy_example_updates'],final_slot_replaceable=True,search_continues=True))

    @lru_cache(maxsize=42)
    def body_image(self,key,fi):
        c=self.cases[key];folder=Path(c['folder'])/c['render_dir'];p=folder/f'frame_{fi:04d}.png'
        if not p.exists()and self.partial:
            options=list(folder.glob('frame_*.png'))
            if not options:return Image.new('RGBA',(1080,825))
            p=min(options,key=lambda p:abs(int(p.stem.split('_')[-1])-fi))
        with Image.open(p)as im:return im.convert('RGBA').crop((0,190,1080,1015))

    @lru_cache(maxsize=16)
    def iso_image(self,fi):
        folder=Path(self.hero['folder'])/'study_render';p=folder/f'frame_{fi:04d}.png'
        if not p.exists()and self.partial:
            options=list(folder.glob('frame_*.png'))
            if not options:return Image.new('RGBA',(756,1260))
            p=min(options,key=lambda p:abs(int(p.stem.split('_')[-1])-fi))
        with Image.open(p)as im:return im.convert('RGBA').crop((200,60,650,810)).resize((756,1260),Image.Resampling.LANCZOS)

    @lru_cache(maxsize=1)
    def brain_layers(self):
        im=Image.new('RGBA',(240,230));d=ImageDraw.Draw(im)
        def curve(points,color,width=3,closed=False):
            a=np.array(points,float)*np.array([200,178])+[20,10];t=np.arange(len(a));s=CubicSpline(t,a,bc_type='periodic'if closed else'natural');q=s(np.linspace(0,len(a)-1,160));d.line([tuple(p)for p in q],fill=color,width=width,joint='curve')
        outline=[(.5,.12),(.38,.03),(.23,.08),(.12,.23),(.08,.43),(.12,.66),(.28,.85),(.43,.84),(.50,.74),(.57,.84),(.72,.85),(.88,.66),(.92,.43),(.88,.23),(.77,.08),(.62,.03),(.5,.12)]
        curve(outline,(*MINT,230),4,True)
        for mirror in[False,True]:
            folds=[[(.42,.17),(.30,.17),(.24,.29),(.34,.33),(.43,.28)],[(.17,.33),(.20,.43),(.34,.42),(.42,.48),(.36,.56)],[(.14,.53),(.23,.53),(.26,.64),(.38,.66),(.43,.61)],[(.27,.25),(.20,.25),(.17,.35)],[(.21,.72),(.33,.74),(.40,.72)],[(.45,.39),(.44,.51),(.47,.63)]]
            for line in folds:curve([(1-x if mirror else x,y)for x,y in line],(*FG,170),3)
        curve([(.5,.15),(.49,.34),(.51,.55),(.5,.75),(.52,.97)],(*BLUE,190),3)
        return im,im.filter(ImageFilter.GaussianBlur(9))

    def brain(self,im,clock,back=False,strength=1.):
        outline,glow=self.brain_layers();pulse=(.5+.5*math.sin(clock*15))**2;alpha=.25+.70*pulse*strength;g=glow.copy();g.putalpha(g.getchannel('A').point(lambda v:round(v*alpha)));im.paste(g.resize((186,178)),(407,166),g.resize((186,178)));im.paste(outline.resize((186,178)),(407,166),outline.resize((186,178)));d=ImageDraw.Draw(im);color=ACC if back else MINT
        for k in range(3):
            x=583+((clock*130+k*11)%30);y=239+(k-1)*12;d.ellipse((x-2,y-2,x+2,y+2),fill=color)

    def project(self,key,p):
        c=self.data[key]['cam'];ppm=1080/c['ortho_width_m'];return(540+(-p[1]+c['initial_foremost_sole_y_m']-c['camera_center_forward_m'])*ppm,350+525-(p[2]-c['camera_center_z_m'])*ppm-190)

    def ghosts(self,im,key,t,previous=None,fade=0.):
        c=self.cases[key];n=round(c['duration_s']*c['render_fps'])
        for lag,opacity in[(.60,.065),(.40,.11),(.20,.18)]:
            if t-lag<c['render_start_s']:continue
            fi=int(np.clip(round((t-lag)/c['duration_s']*n),*self.render_counts[key]));body=self.body_image(key,fi).copy();body.putalpha(body.getchannel('A').point(lambda v:round(v*opacity)));im.paste(body,(0,350),body)
        if previous and fade>0:
            prev=self.cases[previous];fi=round(prev['duration_s']*prev['render_fps']);body=self.body_image(previous,fi).copy();body.putalpha(body.getchannel('A').point(lambda v:round(v*.12*fade)));shift=round((prev['camera_center_forward_m']-c['camera_center_forward_m'])*1080/4.6);im.paste(body,(shift,350),body)

    def arrow(self,d,a,b,color,width=4):
        if np.linalg.norm(np.array(b)-a)<4:return
        d.line((*a,*b),fill=color,width=width);theta=math.atan2(b[1]-a[1],b[0]-a[0]);r=10;d.polygon([b,(b[0]-r*math.cos(theta-.45),b[1]-r*math.sin(theta-.45)),(b[0]-r*math.cos(theta+.45),b[1]-r*math.sin(theta+.45))],fill=color)

    def control_links(self,im,key,t,clock):
        D=self.data[key];i=int(np.argmin(abs(D['z']['times']-t)));d=ImageDraw.Draw(im)
        for n,name in enumerate(['hip_flexion_r','knee_angle_r','ankle_angle_r']):
            end=self.project(key,D['j']['anchors'][i,D['names'].index(name)]);nodes=np.array([(925,323),(1008,425+n*35),(max(end[0]+60,875),end[1]-90),end]);s=CubicSpline(np.arange(4),nodes);p=s(np.linspace(0,3,70));d.line([tuple(v)for v in p],fill=(33,67,77),width=1);q=s(((clock*1.8+n*.65)%3));d.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=MINT)

    def joint_control(self,im,key,t,angles=False):
        D=self.data[key];z=D['z'];i=int(np.argmin(abs(z['times']-t)));d=ImageDraw.Draw(im);p=D['r']['parameters'];names=D['names'];out=[]
        for j,name in enumerate(['hip_flexion_r','knee_angle_r','ankle_angle_r']):
            k=names.index(name);tau=float(z['joint_net_torque_Nm'][i,k]);ratio=min(1.,abs(tau)/p['joint_torque_limits_Nm'][name]);x,y=self.project(key,D['j']['anchors'][i,k]);r=19;co=tuple(round(a+(b-a)*ratio)for a,b in zip((84,103,120),ACC));box=(x-r,y-r,x+r,y+r);d.arc(box,30,330,fill=(37,54,70),width=3);end=30+ratio*300;d.arc(box,30,end,fill=co,width=5);theta=math.radians(end if tau>0 else 30);point=(x+r*math.cos(theta),y+r*math.sin(theta));direction=theta+(math.pi/2 if tau>0 else-math.pi/2);self.arrow(d,(point[0]-7*math.cos(direction),point[1]-7*math.sin(direction)),point,co,3);angle=float(np.degrees(D['j']['angles_rad'][i,k]));out.append(angle)
            if angles:txt(d,x+27,y-9,f'{angle:.0f}°',18,FG,True)
        F=z['force_world_N'][i];cop=z['actual_COP_world_m'][i]
        if len(cop)==3 and np.isfinite(cop).all()and np.linalg.norm(F)>50:
            a=self.project(key,cop);b=(a[0]-F[1]/(87*9.80665)*14,a[1]-F[2]/(87*9.80665)*14);self.arrow(d,a,b,BLUE,4)
        return out

    def phase_x(self,key,t):
        return np.interp(t,self.metrics['cases'][key]['phase_times'],[0,.12,.34,.52,.72,1.])

    @lru_cache(maxsize=64)
    def curve(self,key,metric,left,right,top,bottom):
        # Keep narrow native contact peaks while reducing to screen columns.
        data=self.series[key];phase=np.interp(data['times'],self.metrics['cases'][key]['phase_times'],[0,.12,.34,.52,.72,1.]);values=data[metric];bins=np.clip((phase*(right-left)).astype(int),0,int(right-left));starts=np.r_[0,np.flatnonzero(np.diff(bins))+1];ends=np.r_[starts[1:],len(bins)];ids=[]
        for a,b in zip(starts,ends):
            part=values[a:b];ids.extend(sorted(set([int(a),int(a+np.argmin(part)),int(a+np.argmax(part)),int(b-1)])))
        return[(left+phase[i]*(right-left),bottom-values[i]/self.limits[metric]*(bottom-top))for i in ids]

    def traces(self,d,key,t):
        progress=self.phase_x(key,t);D=self.data[key];idx=int(np.argmin(abs(D['z']['times']-t)));angles=[]
        for name in['hip_flexion_r','knee_angle_r','ankle_angle_r']:angles.append(float(np.degrees(D['j']['angles_rad'][idx,D['names'].index(name)])))
        for j,metric in enumerate(self.metric_keys):
            x=45+(j%2)*510;y=1110+(j//2)*188;w=480;d.rounded_rectangle((x,y,x+w,y+166),radius=11,fill=(18,33,49));txt(d,x+14,y+11,self.names[j],20,MUTED,True);val=float(np.interp(t,self.series[key]['times'],self.series[key][metric]));txt(d,x+w-62,y+5,f'{val:.1f}'if j==0 else f'{val:.0f}',31,STAGE_COLORS[self.keys.index(key)],True,anchor='rt');txt(d,x+w-13,y+20,self.units[j],15,MUTED,anchor='rt')
            if j:txt(d,x+115,y+17,f'{angles[j-1]:.0f}°',17,FG)
            left=x+15;right=x+w-15;top=y+64;bottom=y+146;limit=self.limits[metric]
            for stage,color in zip(self.keys,STAGE_COLORS):
                pts=self.curve(stage,metric,left,right,top,bottom);d.line(pts,fill=color if stage==key else tuple(int(v*.43)for v in color),width=3 if stage==key else 1)
            cursor=left+progress*(right-left);d.line((cursor,top,cursor,bottom),fill=FG,width=1)
        for i,(dist,color)in enumerate(zip(self.m['final_four_distances_m'],STAGE_COLORS)):
            x=45+i*250;d.line((x,1505,x+20,1505),fill=color,width=4);txt(d,x+30,1494,f'{dist:.2f} m',18,color,True)

    def policy_controls(self,d,blend):
        a=self.data['policy_before']['r']['parameters'];b=self.data['policy_after']['r']['parameters'];items=[('LAND PREP',a['prepare_transition_s']*1000,b['prepare_transition_s']*1000,'ms'),('HIP DRIVE',np.mean(a['propulsion_effort_knots_fraction']['hip_flexion'])*100,np.mean(b['propulsion_effort_knots_fraction']['hip_flexion'])*100,'% CAP'),('FLIGHT TRUNK',math.degrees(a['flight_trunk']),math.degrees(b['flight_trunk']),'°')]
        for j,(label,old,new,unit)in enumerate(items):
            y=1120+j*120;d.rounded_rectangle((45,y,1035,y+98),radius=10,fill=(18,33,49));txt(d,65,y+15,label,24,MUTED,True);s=f'{old:.1f} → {new:.1f} {unit}'if j==0 else f'{old:+.1f} → {new:+.1f} {unit}'if j==1 else f'{old:.1f} → {new:.1f}{unit}';txt(d,1010,y+44,s,31,MINT if blend else GREY,True,anchor='rt')

    def render(self,seg,local):
        kind=seg['kind'];duration=seg['end_s']-seg['start_s'];im,d=self.base()
        if kind=='hook':
            t=.5+local*.75;body=self.iso_image(min(90,round(t*30)));im.paste(body,(160,300),body);d=ImageDraw.Draw(im);txt(d,45,82,'CAN AI FIND',70,FG,True);txt(d,45,163,'THE BEST JUMP?',70,FG,True);txt(d,45,278,'BIOMECHANICS × REINFORCEMENT LEARNING',24,MINT,True);txt(d,45,1430,'STANDING LONG JUMP',27,MUTED,True)
        elif kind=='control':
            r=self.data['initial']['r'];p=r['parameters'];start=p['ready_s'];end=r['takeoff_s']+.1;t=start+(end-start)*ease(local/duration);native=self.world(im,'initial',t,False);self.joint_control(im,'initial',native,True);d=ImageDraw.Draw(im);txt(d,45,80,'TORQUE → MOTION',49,FG,True);txt(d,45,151,'SAME BODY. FIXED LIMITS.',27,MUTED,True);self.traces(d,'initial',native)
        elif kind=='conditions':
            self.world(im,'initial',0.,False);self.joint_control(im,'initial',0.);d=ImageDraw.Draw(im);txt(d,45,80,'ONE VIRTUAL ATHLETE',43,FG,True)
            for x,w,s in[(45,280,'178 CM / 87 KG'),(340,315,'FIXED TORQUE'),(670,365,'58 PARAMETERS')]:d.rounded_rectangle((x,1120,x+w,1190),radius=12,fill=DIM);txt(d,x+w/2,1140,s,23,FG,True,anchor='mt')
            txt(d,45,1300,'DISTANCE + LANDING',31,MINT,True);txt(d,45,1420,'POSE  ·  TIMING  ·  DRIVE',27,MUTED,True)
        elif kind=='rl_example':
            before,after=self.m['policy_example_updates'];update_start=2.4;update_end=4.1;afterpart=local>=update_end;key='policy_after'if afterpart else'policy_before';c=self.cases[key];u=np.clip((local-update_end)/(duration-update_end)if afterpart else local/update_start,0,1);t=c['render_start_s']+u*(c['duration_s']-c['render_start_s']);native=self.world(im,key,t)
            if afterpart:self.control_links(im,key,native,local)
            self.joint_control(im,key,native,True);self.brain(im,local,back=update_start<=local<update_end);d=ImageDraw.Draw(im);txt(d,45,78,'LEARN FROM',43,FG,True);txt(d,45,134,'THE LANDING',43,FG,True);updating=update_start<=local<update_end;stage='UPDATE'if updating else'AFTER'if afterpart else'BEFORE';txt(d,45,219,stage,29,ACC if updating else MINT if afterpart else MUTED,True);batch=after if afterpart else before;self.graph(d,batch,1.,pulse=((local-update_start)/.8)%1 if updating else(local/.65)%1,back=updating,after=True);self.policy_controls(d,afterpart or updating)
            txt(d,45,1477,f'UPDATES {before} → {after}  ·  96 TRIALS',21,MUTED,True)
            if updating:
                d.rectangle((40,1018,600,1056),fill=BG)
                txt(d,45,1025,'DISTANCE + LANDING → REWARD',23,MINT,True);d.rounded_rectangle((45,1058,360,1097),radius=8,fill=DIM);txt(d,60,1063,f'{self.cases["policy_before"]["record"]["reward"]:.2f} → {self.cases["policy_after"]["record"]["reward"]:.2f}',25,MINT,True)
        elif kind=='montage':
            c=self.cases[seg['case']];start=c['render_start_s'];u=min(1.,local/max(.1,duration-.12));t=start+u*(c['duration_s']-start);ordered=[s for s in self.segments if s['kind']=='montage'];i=ordered.index(seg);previous=ordered[i-1]['case']if i else None;self.ghosts(im,c['key'],t,previous,max(0.,1-local/.16));self.world(im,c['key'],t);self.brain(im,local+seg['start_s']);d=ImageDraw.Draw(im);txt(d,45,80,'TRY. FALL. ADAPT.',43,FG,True);trial=c['record']['training_trial'];txt(d,45,145,f'TRIAL {trial}',27,MUTED,True);self.graph(d,max(1,min(50,(max(1,trial)-1)//48+1)),float(c['record'].get('context')or 0.),pulse=(local/.3)%1,after=True);self.progress(d,trial)
        elif kind=='reseed':
            self.brain(im,local,True);self.graph(d,17,1.,pulse=local/duration,after=True);txt(d,45,80,'RESEED',53,ACC,True);txt(d,45,150,'SAME LIMITS',27,MUTED,True);self.world(im,'middle_one',self.cases['middle_one']['duration_s']);self.progress(d,768)
        elif kind=='final_stage':
            key=seg['case'];c=self.cases[key];t=min(c['duration_s'],local);native=self.world(im,key,t,False);self.joint_control(im,key,native,True);d=ImageDraw.Draw(im);i=seg['index'];col=STAGE_COLORS[i];txt(d,45,78,f'{i+1:02d} / 04',30,MUTED,True);txt(d,45,127,'INITIAL SUCCESS'if i==0 else'LONGEST CHECKED'if i==3 else'IMPROVEMENT',43,FG,True);txt(d,45,191,f'{c["checked_distance_m"]:.2f} m',49,col,True);r=self.data[key]['r'];p=r['parameters'];phase=0 if native<p['ready_s']+p['crouch_s']+p.get('prepush_lean_s',0)else 1 if native<r['takeoff_s']else 2 if native<r['touchdown_s']else 3
            for j,s in enumerate(['LOAD','PUSH','FLIGHT','LAND']):
                x=45+j*252;d.rounded_rectangle((x,280,x+234,324),radius=9,fill=col if j==phase else DIM);txt(d,x+117,289,s,20,BG if j==phase else MUTED,True,anchor='mt')
            self.traces(d,key,native)
        else:
            txt(d,45,80,'FOUR STAGES',49,FG,True);txt(d,45,149,'SAME TORQUE LIMITS',27,MUTED,True)
            for i,(key,col)in enumerate(zip(self.keys,STAGE_COLORS)):
                c=self.cases[key];x=45+i*252;txt(d,x+116,235,f'{c["checked_distance_m"]:.2f} m',34,col,True,anchor='mt');t=self.metrics['cases'][key]['phase_times'][2];fi=round(t/c['duration_s']*round(c['duration_s']*c['render_fps']));body=self.body_image(key,fi);bbox=body.getbbox()
                if bbox:
                    body=body.crop(bbox);body=body.resize((round(body.width*.60),round(body.height*.60)),Image.Resampling.LANCZOS);im.paste(body,(round(x+116-body.width/2),round(630-body.height)),body)
            d=ImageDraw.Draw(im)
            for j,metric in enumerate(self.metric_keys):
                y=710+j*188;txt(d,45,y,self.names[j],24,MUTED,True);maximum=max(self.metrics['cases'][k]['peak_push'][metric]for k in self.keys)
                for i,(key,col)in enumerate(zip(self.keys,STAGE_COLORS)):
                    x=45+i*252;v=self.metrics['cases'][key]['peak_push'][metric];txt(d,x+116,y+32,f'{v:.1f}'if j==0 else f'{v:.0f}',34,col,True,anchor='mt');d.rounded_rectangle((x,y+89,x+230*v/maximum,y+107),radius=7,fill=col)
                txt(d,1035,y,self.units[j],20,MUTED,anchor='rt')
            txt(d,45,1462,'PUSH-OFF PEAKS  ·  SEARCH CONTINUES',20,ACC,True);txt(d,45,1500,'MuJoCo · PPO | BodyParts3D/DBCLS · MyoHub · Z-Anatomy · BodyExplorer',13,MUTED)
        assert np.all(np.asarray(im)[1536:]==BG),(kind,'footer overwritten');return im

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');ap.add_argument('--partial',action='store_true');args=ap.parse_args();draft=Draft(args.partial);targets=[('hook',1.5),('control',4.2),('brain',10.9),('update',11.4),('ghosts',next(s['start_s']+.55 for s in draft.segments if s['kind']=='montage')),('four_stages',draft.duration-1.),('final_motion',next(s['start_s']+1.5 for s in draft.segments if s['kind']=='final_stage'and s['index']==3))]
    for name,t in targets:draft.at(t).save(OUT/f'preview_{name}.png')
    if args.preview:return
    video=OUT/'standing_long_jump_rl_draft_v2.mp4';writer=imageio_ffmpeg.write_frames(str(video),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None);n=round(draft.duration*30)
    try:
        for k in range(n):writer.send(np.asarray(draft.at(k/30)))
    finally:writer.close()
    atomic(OUT/'draft_delivery.json',dict(status='encoded',video=str(video),frames=n,fps=30,duration_s=n/30,resolution=[1080,1920],footer_reserved_px=384,com_sphere=False,trajectory_after_landing=True,brain_is_symbolic=True,network_is_actual_checkpoint=True,policy_example_updates=draft.m['policy_example_updates'],native_distinct_falls=12,ghosts_per_trial=3,final_four_distances_m=draft.m['final_four_distances_m'],individual_muscle_forces=False,forces_shown='Resultant GRF and right hip/knee/ankle net torque.',last_result_replaceable=True));print('V2_VIDEO_READY',video,n/30,flush=True)

if __name__=='__main__':main()



