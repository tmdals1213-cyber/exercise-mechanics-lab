"""Visible real controls, persistent pre-fall memories and readable muscle views."""
from pathlib import Path
from functools import lru_cache
import math,argparse
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
from scipy.interpolate import CubicSpline
import compose_jump_rl_draft_v2 as previous
from compose_jump_rl_draft_v2 import txt,ease,MINT,RED,BLUE,DIM,GREY,STAGE_COLORS,FG,MUTED,ACC,BG
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v3';previous.OUT=OUT

class Draft(previous.Draft):
    def __init__(self,partial=False):
        super().__init__(partial);self.muscle_keys=self.m['display_muscle_keys'];self.muscle_names=['GLUTES','QUADS','HAMSTRINGS','CALVES'];self.muscles={k:np.load(OUT/f'{k}_muscles.npz')for k in self.keys};self.muscle_report=read(OUT/'muscle_analysis.json');self.reference=self.m['relative_muscle_reference_N'];self.muscle_limits={g:max(float(self.muscles[k][g].max()/self.reference[g])for k in self.keys)*1.08 for g in self.muscle_keys};self.parameter_key='initial';self.parameter_time=0.;self.segments=[];clock=0.
        def add(kind,duration,**kw):
            nonlocal clock
            self.segments.append(dict(kind=kind,start_s=clock,end_s=clock+duration,**kw));clock+=duration
        add('hook',2.8);add('control',5.8);add('conditions',1.6);add('rl_example',8.2)
        clips=[c for c in self.m['cases']if c['role']=='montage']+[self.cases[k]for k in self.keys[1:]];clips.sort(key=lambda c:c['record']['training_trial']);reseed=False
        for c in clips:
            if c['record']['training_trial']>768 and not reseed:add('reseed',.7);reseed=True
            add('montage',1.0 if c['record']['fell']else 1.15,case=c['key'])
        for i,key in enumerate(self.keys):add('final_stage',self.cases[key]['duration_s']+.2,case=key,index=i)
        add('four_stage_summary',3.8);self.duration=clock;atomic(OUT/'timeline.json',dict(duration_s=clock,fps=30,segments=self.segments,final_four_keys=self.keys,final_four_distances_m=self.m['final_four_distances_m'],persistent_preimpact_ghosts=4,visible_parameter_families=['arm swing and elbow pose','load/push/landing timing','five families of three-knot equivalent drive','air tuck and landing preparation'],final_views=self.m['final_views'],tensions='Conditional muscle group tensions / initial successful push-off peak',com_sphere=False))

    def graph(self,d,*a,**kw):
        super().graph(d,*a,**kw);d.rectangle((598,341,1079,386),fill=BG);d.rectangle((935,163,1079,338),fill=BG)
        for y,s in zip(self.nodes_y[-1],['POSE','TIME','DRIVE']):txt(d,942,y-12,s,25,MUTED,True)
        txt(d,610,346,'58 PARAMETERS',29,MINT,True)

    def progress(self,d,trial,box=(45,1170,1035,1420)):
        left,top,right,bottom=box;best=self.m['baseline_retained_m'];points=[]
        for i,r in enumerate(self.hist[:min(trial,len(self.hist))]):
            if not r['eligible']:continue
            value=r['retained_distance_m']if r['stable']else r['distance_m'];x=left+(right-left)*i/max(1,len(self.hist)-1);y=bottom-(value-1.8)/1.85*(bottom-top)
            if top<=y<=bottom:d.ellipse((x-3,y-3,x+3,y+3),fill=MINT if r['stable']else RED)
            if r['stable']:best=max(best,r['retained_distance_m'])
            points.append((x,bottom-(best-1.8)/1.85*(bottom-top)))
        if len(points)>1:d.line(points,fill=ACC,width=7)
        txt(d,left,1452,f'{trial} TRIALS',34,MUTED,True);txt(d,right,1452,f'SAMPLED {best:.2f} m',34,ACC,True,anchor='rt')

    def phase(self,key,t):
        r=self.data[key]['r'];p=r['parameters'];return 0 if t<p['ready_s']+p['crouch_s']+p.get('prepush_lean_s',0)else 1 if t<r['takeoff_s']else 2 if t<r['touchdown_s']else 3

    def parameter_cards(self,d,key,t):
        p=self.data[key]['r']['parameters'];phase=self.phase(key,t);arm=(p['shoulder_bottom'],p['shoulder_top'])if phase<2 else(p['flight_shoulder'],p['elbow'])if phase==2 else(p['landing_shoulder'],.35);labels=['ARM / ELBOW'if phase>=2 else'ARM SWING','TIMING','HIP DRIVE','TUCK / LAND'];vals=[f'{math.degrees(arm[0]):.0f}° / {math.degrees(arm[1]):.0f}°',f'{p["crouch_s"]:.2f} / {p["push_s"]:.2f} s',None,f'{math.degrees(p["flight_hip"]):.0f}° / {math.degrees(p["flight_knee"]):.0f}°'if phase<3 else f'{math.degrees(p["landing_knee"]):.0f}°'];subs=['Shoulder targets'if phase<2 else'Shoulder / elbow','Load / push',None,'Hip / knee'if phase<3 else'Landing knee']
        if phase<2:subs[0]=f'Arm delay {p["arm_delay_s"]:+.2f} s'
        if phase==2:vals[1]=f'{p["prepare_landing_s"]:.2f} s';subs[1]='Landing preparation'
        if phase==3:vals[1]=f'{p["landing_transition_s"]:.2f} s';subs[1]='Landing transition'
        for i,(label,value)in enumerate(zip(labels,vals)):
            x=45+(i%2)*510;y=1110+(i//2)*190;d.rounded_rectangle((x,y,x+480,y+174),radius=12,fill=(18,33,49));txt(d,x+17,y+14,label,30,MUTED,True)
            if getattr(self,'parameter_updating',False):d.rounded_rectangle((x,y,x+480,y+174),radius=12,outline=ACC,width=3)
            if value is not None:txt(d,x+17,y+59,value,43,MINT,True);txt(d,x+17,y+121,subs[i],27,MUTED)
            else:
                u=np.linspace(0,1,120);from jump_effort_control import effort_curve
                v=np.array([effort_curve(q,p['propulsion_effort_knots_fraction']['hip_flexion'])for q in u]);left=x+22;right=x+458;middle=y+108;pts=[(left+q*(right-left),middle-w/.4*48)for q,w in zip(u,v)];d.line((left,middle,right,middle),fill=DIM,width=3);d.line(pts,fill=MINT,width=7)
                for q,w in zip([.25,.5,.75],p['propulsion_effort_knots_fraction']['hip_flexion']):cx=left+q*(right-left);cy=middle-w/.4*48;d.ellipse((cx-7,cy-7,cx+7,cy+7),fill=FG)
                txt(d,x+17,y+135,'3 CONTROL POINTS',25,MUTED,True)
        txt(d,45,1500,'POSE  ·  TIMING  ·  DRIVE',27,MUTED,True)

    def policy_controls(self,d,blend):self.parameter_cards(d,self.parameter_key,self.parameter_time)

    def joint_control(self,im,key,t,angles=False):
        D=self.data[key];i=int(np.argmin(abs(D['z']['times']-t)));d=ImageDraw.Draw(im);p=D['r']['parameters'];joints=['shoulder_elv_r','elbow_flexion_r','flex_extension','hip_flexion_r','knee_angle_r','ankle_angle_r']
        for name in joints:
            k=D['names'].index(name);v=float(D['z']['joint_net_torque_Nm'][i,k]);ratio=min(1.,abs(v)/p['joint_torque_limits_Nm'][name]);x,y=self.project(key,D['j']['anchors'][i,k]);rad=16 if name.startswith(('elbow','flex_'))else 19;co=tuple(round(a+(b-a)*ratio)for a,b in zip((74,91,106),(238,95,81)));box=(x-rad,y-rad,x+rad,y+rad);d.arc(box,30,330,fill=(31,42,55),width=3);d.arc(box,30,30+300*ratio,fill=co,width=5)

    def control_links(self,im,key,t,clock):
        D=self.data[key];i=int(np.argmin(abs(D['z']['times']-t)));d=ImageDraw.Draw(im)
        for n,name in enumerate(['shoulder_elv_r','elbow_flexion_r','flex_extension','hip_flexion_r','knee_angle_r','ankle_angle_r']):
            end=self.project(key,D['j']['anchors'][i,D['names'].index(name)]);start=(925,183 if n<2 else 323);nodes=np.array([start,(1008,420+n*15),(max(end[0]+60,875),end[1]-80),end]);s=CubicSpline(np.arange(4),nodes);points=s(np.linspace(0,3,60));d.line([tuple(v)for v in points],fill=(33,67,77),width=2);q=s((clock*1.8+n*.47)%3);d.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=MINT)

    @lru_cache(maxsize=12)
    def pre_fall_times(self,key):
        D=self.data[key];r=D['r'];z=D['z'];head=read(Path(self.cases[key]['folder'])/'body_names.json').index('head');upper=r['first_body_floor_contact_s']-.12;eligible=np.flatnonzero((z['times']>r['takeoff_s']+.03)&(z['times']<upper)&(z['body_xpos'][:,head,2]>.65));assert len(eligible),key;targets=[min(r['touchdown_s']+.08,upper-.08),upper-.02];return[float(z['times'][eligible[np.argmin(abs(z['times'][eligible]-t))]])for t in targets]

    def persistent_ghosts(self,im,seg):
        earlier=[s for s in self.segments if s['kind']=='montage'and s['start_s']<seg['start_s']and self.cases[s['case']]['record']['fell']][-3:];frames=[]
        for age,s in enumerate(earlier):
            key=s['case'];times=self.pre_fall_times(key)if s==earlier[-1]else self.pre_fall_times(key)[:1]
            for t in times:frames.append((key,t,.07+.035*age))
        current=self.cases[seg['case']]
        for key,t,opacity in frames[-4:]:
            c=self.cases[key];n=round(c['duration_s']*c['render_fps']);fi=int(np.clip(round(t/c['duration_s']*n),*self.render_counts[key]));body=self.body_image(key,fi).copy();body.putalpha(body.getchannel('A').point(lambda a:round(a*opacity)));shift=round((c['camera_center_forward_m']-current['camera_center_forward_m'])*1080/4.6);im.paste(body,(shift,350),body)

    @lru_cache(maxsize=20)
    def multi_image(self,key,fi):
        c=self.cases[key];folder=Path(c['multiview_folder'])/c['multiview_dir'];p=folder/f'frame_{fi:04d}.png'
        if not p.exists()and self.partial:
            files=list(folder.glob('frame_*.png'))
            if not files:return Image.new('RGBA',(1600,860))
            p=min(files,key=lambda x:abs(int(x.stem.split('_')[-1])-fi))
        with Image.open(p)as im:return im.convert('RGBA')

    @lru_cache(maxsize=4)
    def rear_image(self,key):
        c=self.cases[key];fi=c.get('rear_poster_frame',round(self.metrics['cases'][key]['phase_times'][2]/c['duration_s']*round(c['duration_s']*c['multiview_fps'])));p=Path(c['multiview_folder'])/'rear_posters'/f'frame_{fi:04d}.png'
        if not p.exists()and self.partial:return self.multi_image(key,fi).crop((800,0,1600,860))
        with Image.open(p)as im:return im.convert('RGBA').crop((800,0,1600,860))

    def multiview(self,im,key,t):
        c=self.cases[key];n=round(c['duration_s']*c['multiview_fps']);fi=int(np.clip(round(t/c['duration_s']*n),0,n));native=fi/n*c['duration_s'];raw=self.multi_image(key,fi)
        for i,center in enumerate([270,810]):
            panel=raw.crop((i*800,0,(i+1)*800,860));bbox=panel.getchannel('A').getbbox()
            if bbox:
                part=panel.crop((max(0,bbox[0]-5),0,min(800,bbox[2]+5),860));im.paste(part,(round(center-part.width/2),200),part)
        d=ImageDraw.Draw(im);txt(d,55,350,'FRONT ¾',27,MUTED,True);txt(d,595,350,'SIDE',27,MUTED,True);return native

    @lru_cache(maxsize=32)
    def muscle_curve(self,key,group,left,right,top,bottom):
        z=self.muscles[key];phase=np.interp(z['times'],self.metrics['cases'][key]['phase_times'],[0,.12,.34,.52,.72,1]);return[(left+x*(right-left),bottom-v/self.reference[group]/self.muscle_limits[group]*(bottom-top))for x,v in zip(phase,z[group])]

    def muscle_cards(self,d,key,t):
        progress=self.phase_x(key,t)
        for j,group in enumerate(self.muscle_keys):
            x=45+(j%2)*510;y=1110+(j//2)*192;d.rounded_rectangle((x,y,x+480,y+176),radius=12,fill=(18,33,49));txt(d,x+16,y+12,self.muscle_names[j],30,MUTED,True);v=float(np.interp(t,self.muscles[key]['times'],self.muscles[key][group]))/self.reference[group];txt(d,x+460,y+47,f'{v:.2f}×',42,MINT,True,anchor='rt');left=x+16;right=x+464;top=y+72;bottom=y+157
            d.line(self.muscle_curve('initial',group,left,right,top,bottom),fill=GREY,width=5);d.line(self.muscle_curve(key,group,left,right,top,bottom),fill=MINT,width=8);cx=left+progress*(right-left);d.line((cx,top,cx,bottom),fill=FG,width=3)
        txt(d,45,1504,'MODEL TENSION  ·  INITIAL = 1.0×',27,MUTED,True)

    def render(self,seg,local):
        kind=seg['kind'];duration=seg['end_s']-seg['start_s']
        if kind=='rl_example':
            after=local>=4.1;self.parameter_updating=2.4<=local<4.1;self.parameter_key='policy_after'if after else'policy_before';c=self.cases[self.parameter_key];u=np.clip((local-4.1)/(duration-4.1)if after else local/2.4,0,1);self.parameter_time=c['render_start_s']+u*(c['duration_s']-c['render_start_s']);im=super().render(seg,local);d=ImageDraw.Draw(im);d.rectangle((40,1477,1060,1535),fill=BG);txt(d,45,1493,'POLICY → PARAMETERS → CONTROLLER',27,MUTED,True);self.parameter_updating=False;return im
        if kind not in['control','montage','final_stage','four_stage_summary']:return super().render(seg,local)
        im,d=self.base()
        if kind=='control':
            key='initial';t=min(self.cases[key]['duration_s'],local/duration*self.cases[key]['duration_s']);native=self.world(im,key,t,False);self.joint_control(im,key,native);d=ImageDraw.Draw(im);txt(d,45,78,'WHAT CAN AI CHANGE?',49,FG,True);txt(d,45,146,'ARMS · TIMING · DRIVE · TUCK',30,MUTED,True);phase=self.phase(key,native)
            for i,s in enumerate(['LOAD','PUSH','AIR','LAND']):x=45+i*252;d.rounded_rectangle((x,265,x+234,314),radius=9,fill=MINT if phase==i else DIM);txt(d,x+117,274,s,26,BG if phase==i else MUTED,True,anchor='mt')
            txt(d,45,1030,'AIR: POSE CONTROL'if phase==2 else'GROUND: SET LAUNCH'if phase<2 else'LAND: ABSORB + BALANCE',31,MINT,True);self.parameter_cards(d,key,native)
        elif kind=='montage':
            c=self.cases[seg['case']];start=c['render_start_s'];t=start+min(1,local/max(.1,duration-.16))*(c['duration_s']-start);self.persistent_ghosts(im,seg);self.world(im,c['key'],t);self.brain(im,local+seg['start_s']);d=ImageDraw.Draw(im);txt(d,45,78,'TRY. FALL. ADAPT.',43,FG,True);trial=c['record']['training_trial'];txt(d,45,143,f'TRIAL {trial}',31,MUTED,True);self.graph(d,max(1,min(50,(max(1,trial)-1)//48+1)),float(c['record'].get('context')or 0.),pulse=(local/.3)%1,after=True);self.progress(d,trial)
        elif kind=='final_stage':
            key=seg['case'];c=self.cases[key];t=min(c['duration_s'],local);native=self.multiview(im,key,t);d=ImageDraw.Draw(im);i=seg['index'];txt(d,45,78,f'{i+1:02d} / 04',32,MUTED,True);txt(d,45,127,'INITIAL SUCCESS'if i==0 else'LONGEST CHECKED'if i==3 else'IMPROVEMENT',43,FG,True);txt(d,45,192,f'{c["checked_distance_m"]:.2f} m',49,STAGE_COLORS[i],True);grf=float(np.interp(native,self.series[key]['times'],self.series[key]['ground']));txt(d,1035,205,f'GRF {grf:.1f} BW',32,BLUE,True,anchor='rt');phase=self.phase(key,native)
            for j,s in enumerate(['LOAD','PUSH','AIR','LAND']):x=45+j*252;d.rounded_rectangle((x,275,x+234,320),radius=9,fill=STAGE_COLORS[i]if phase==j else DIM);txt(d,x+117,283,s,25,BG if phase==j else MUTED,True,anchor='mt')
            self.muscle_cards(d,key,native)
        else:
            txt(d,45,76,'FOUR STAGES',49,FG,True);txt(d,45,146,'MODEL TENSION · × INITIAL',30,MUTED,True);txt(d,1035,151,'REAR VIEW',27,MUTED,True,anchor='rt')
            for i,key in enumerate(self.keys):
                c=self.cases[key];x=45+i*252;txt(d,x+116,235,f'{c["checked_distance_m"]:.2f} m',40,STAGE_COLORS[i],True,anchor='mt');body=self.rear_image(key);bbox=body.getchannel('A').getbbox()
                if bbox:body=body.crop(bbox);body=body.resize((round(body.width*.5),round(body.height*.5)),Image.Resampling.LANCZOS);im.paste(body,(round(x+116-body.width/2),round(630-body.height)),body)
            d=ImageDraw.Draw(im)
            for j,group in enumerate(self.muscle_keys):
                y=690+j*188;txt(d,45,y,self.muscle_names[j],33,MUTED,True);values=[self.muscle_report['cases'][key]['primary_peaks_N'][group]/self.reference[group]for key in self.keys];maximum=max(values)
                for i,(v,col)in enumerate(zip(values,STAGE_COLORS)):x=45+i*252;txt(d,x+116,y+44,f'{v:.2f}×',46,col,True,anchor='mt');d.rounded_rectangle((x,y+110,x+230*v/maximum,y+129),radius=7,fill=col)
            txt(d,45,1470,'PUSH-OFF PEAKS · SEARCH CONTINUES',24,ACC,True);txt(d,45,1510,'MuJoCo · PPO | MyoSuite · BodyParts3D/DBCLS · Z-Anatomy',15,MUTED)
        assert np.all(np.asarray(im)[1536:]==BG),(kind,'footer overwritten');return im

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');ap.add_argument('--partial',action='store_true');args=ap.parse_args();d=Draft(args.partial);times=[('control',5.),('parameters_air',7.1),('learning',12.),('persistent_ghosts',next(s['start_s']+.25 for s in d.segments if s['kind']=='montage'and s['case']=='fall_05')),('multiview',next(s['start_s']+1.5 for s in d.segments if s['kind']=='final_stage'and s['index']==0)),('summary',d.duration-1.)]
    for name,t in times:d.at(t).save(OUT/f'preview_{name}.png')
    if args.preview:return
    video=OUT/'standing_long_jump_rl_draft_v3.mp4';w=imageio_ffmpeg.write_frames(str(video),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);w.send(None);count=round(d.duration*30)
    try:
        for k in range(count):w.send(np.asarray(d.at(k/30)))
    finally:w.close()
    atomic(OUT/'draft_delivery.json',dict(video=str(video),frames=count,fps=30,duration_s=count/30,com_sphere=False,parameter_families_visible=True,persistent_failure_ghosts=4,final_views=d.m['final_views'],muscle_force_conditional=True,normalization='Initial successful model push-off peak per right-side group; not muscle activation or strength.',final_four_distances_m=d.m['final_four_distances_m'],minimum_primary_text_px=25,main_force_curve_width_px=8,last_result_replaceable=True));print('V3_VIDEO_READY',video,count/30,flush=True)

if __name__=='__main__':main()



