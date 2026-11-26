"""Preserve the approved33s; front/rear + world-side path; four simultaneous jumps."""
from pathlib import Path
from functools import lru_cache
import copy,argparse
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
import compose_jump_rl_draft_v3 as base
from compose_jump_rl_draft_v3 import txt,ease,FG,MUTED,ACC,BG,MINT,BLUE,GREY,DIM,STAGE_COLORS
from jump_monitor_data import ROOT,read,atomic
from fullbody_muscle_contour_v4 import COLORS,STOPS
OLD=ROOT/'deliverables/jump_rl_study_draft_v3';OUT=ROOT/'deliverables/jump_rl_study_draft_v4';base.OUT=OUT;base.previous.OUT=OUT

class Draft(base.Draft):
    def __init__(self,partial=False):
        super().__init__(partial);old=read(OLD/'timeline.json');self.segments=[]
        for segment in old['segments']:
            if segment['start_s']>=33.:break
            s=copy.deepcopy(segment)
            if s['end_s']>33.:s['source_end_s']=s['end_s'];s['end_s']=33.
            self.segments.append(s)
        clock=33.
        for i,key in enumerate(self.keys):
            duration=self.cases[key]['duration_s']+.2;self.segments.append(dict(kind='final_stage',case=key,index=i,start_s=clock,end_s=clock+duration));clock+=duration
        self.segments.append(dict(kind='simultaneous_four',start_s=clock,end_s=clock+7.));self.duration=clock+7.
        atomic(OUT/'timeline.json',dict(duration_s=self.duration,fps=30,preserved_prefix_s=33.,segments=self.segments,views=['front_oblique','rear_oblique','world_fixed_side'],COM_path_after_touchdown_only=True,simultaneous_final_models=4,final_four_distances_m=self.m['final_four_distances_m'],fullbody_contours='All identified surfaces, common absoluteN logarithmic palette; explicitly unidentified surfaces grey.'))
        if not partial:
            for key in self.keys:
                c=self.cases[key];folder=Path(c['multiview_folder']);n=round(c['duration_s']*24)
                for name in[c['multiview_dir'],c['full_side_dir']]:assert len(list((folder/name).glob('frame_*.png')))==n+1,(key,name)

    @lru_cache(maxsize=18)
    def side_image(self,key,fi):
        c=self.cases[key];folder=Path(c['multiview_folder'])/c['full_side_dir'];p=folder/f'frame_{fi:04d}.png'
        if not p.exists()and self.partial:
            files=list(folder.glob('frame_*.png'))
            if not files:return Image.new('RGBA',(594,454))
            p=min(files,key=lambda p:abs(int(p.stem.split('_')[-1])-fi))
        with Image.open(p)as im:return im.convert('RGBA').crop((0,190,1080,1015)).resize((594,454),Image.Resampling.LANCZOS)

    def place_person(self,im,raw,panel,center,top,scale):
        source=raw.crop((panel*800,0,(panel+1)*800,860));box=source.getchannel('A').getbbox()
        if not box:return
        part=source.crop((max(0,box[0]-5),0,min(800,box[2]+5),860));part=part.resize((round(part.width*scale),round(part.height*scale)),Image.Resampling.LANCZOS);im.paste(part,(round(center-part.width/2),top),part)

    def legend(self,d):
        txt(d,45,882,'TENSION',25,MUTED,True)
        for y in range(925,1061):
            v=1-(y-925)/135;color=tuple(int(np.interp(v*STOPS[-1],STOPS,COLORS[:,j]))for j in range(3));d.line((47,y,65,y),fill=color,width=1)
        for force,label in[(30000.,'30 kN'),(1000.,'1'),(100.,'0.1'),(0.,'0')]:y=1060-np.log1p(force)/STOPS[-1]*135;txt(d,79,y-14,label,24,MUTED)
        d.rectangle((165,1035,184,1054),fill=(109,116,125));txt(d,192,1032,'N/A',24,MUTED,True)

    def triptych(self,im,key,t):
        c=self.cases[key];D=self.data[key];r=D['r'];cam=D['cam'];z=D['z'];n=round(c['duration_s']*24);fi=int(np.clip(round(t/c['duration_s']*n),0,n));native=fi/n*c['duration_s'];raw=self.multi_image(key,fi)
        self.place_person(im,raw,0,270,230,.85);self.place_person(im,raw,1,810,230,.85);small=self.side_image(key,fi);im.paste(small,(243,720),small);d=ImageDraw.Draw(im);txt(d,55,335,'FRONT ¾',27,MUTED,True);txt(d,595,335,'REAR ¾',27,MUTED,True);txt(d,851,889,'SIDE',31,MUTED,True)
        ppm=1080/cam['ortho_width_m'];zero=243+.55*(540-cam['camera_center_forward_m']*ppm);floor=720+.55*(525+cam['camera_center_z_m']*ppm-190);d.line((249,floor,831,floor),fill=(93,116,135),width=3);d.line((zero,floor-9,zero,floor+9),fill=FG,width=3)
        if native>=r['touchdown_s']:
            end=zero+.55*r['first_contact_heel_distance_m']*ppm;d.line((zero,floor,end,floor),fill=ACC,width=4);ids=np.flatnonzero((z['times']>=r['takeoff_s'])&(z['times']<=r['touchdown_s']));count=max(1,round(ease((native-r['touchdown_s'])/.55)*len(ids)));pts=[]
            for p in z['com'][ids[:count]]:pts.append((243+.55*(540+(-p[1]+cam['initial_foremost_sole_y_m']-cam['camera_center_forward_m'])*ppm),720+.55*(525-(p[2]-cam['camera_center_z_m'])*ppm-190)))
            if len(pts)>1:d.line(pts,fill=ACC,width=4)
        grf=float(np.interp(native,self.series[key]['times'],self.series[key]['ground']));txt(d,851,954,'GRF',27,BLUE,True);txt(d,851,990,f'{grf:.1f} BW',33,BLUE,True);self.legend(d);return native

    def chips(self,d,phase):
        for j,s in enumerate(['LOAD','PUSH','AIR','LAND']):x=45+j*252;d.rounded_rectangle((x,275,x+234,320),radius=9,fill=MINT if phase==j else DIM);txt(d,x+117,283,s,25,BG if phase==j else MUTED,True,anchor='mt')

    def render(self,seg,local):
        if seg['start_s']<33.:
            source=copy.deepcopy(seg);source['end_s']=source.pop('source_end_s',source['end_s']);return super().render(source,local)
        im,d=self.base();kind=seg['kind'];duration=seg['end_s']-seg['start_s']
        if kind=='final_stage':
            key=seg['case'];c=self.cases[key];native=self.triptych(im,key,min(c['duration_s'],local));d=ImageDraw.Draw(im);i=seg['index'];txt(d,45,78,f'{i+1:02d} / 04',32,MUTED,True);txt(d,45,127,'INITIAL SUCCESS'if i==0 else'LONGEST CHECKED'if i==3 else'IMPROVEMENT',43,FG,True);txt(d,45,192,f'{c["checked_distance_m"]:.2f} m',49,STAGE_COLORS[i],True);self.chips(d,self.phase(key,native));self.muscle_cards(d,key,native)
        else:
            txt(d,45,77,'SAME TORQUE LIMITS',46,FG,True);phase_progress=min(1.,local/max(.1,duration-.45));phase=0 if phase_progress<.34 else 1 if phase_progress<.52 else 2 if phase_progress<.72 else 3
            for j,s in enumerate(['LOAD','PUSH','AIR','LAND']):x=45+j*252;d.rounded_rectangle((x,160,x+234,205),radius=9,fill=MINT if phase==j else DIM);txt(d,x+117,168,s,25,BG if phase==j else MUTED,True,anchor='mt')
            for i,key in enumerate(self.keys):
                c=self.cases[key];t=float(np.interp(phase_progress,[0,.12,.34,.52,.72,1.],self.metrics['cases'][key]['phase_times']));n=round(c['duration_s']*24);raw=self.multi_image(key,int(np.clip(round(t/c['duration_s']*n),0,n)));col=i%2;row=i//2;center=270+540*col;top=290+610*row;self.place_person(im,raw,0,center,top,.65);label='INITIAL SUCCESS'if i==0 else'BEST CHECKED'if i==3 else f'IMPROVEMENT {i}';txt(d,center,242+610*row,label,28,MUTED,True,anchor='mt');txt(d,center,282+610*row,f'{c["checked_distance_m"]:.2f} m',45,STAGE_COLORS[i],True,anchor='mt')
            txt(d,45,1450,'SEARCH CONTINUES',33,ACC,True);txt(d,45,1508,'MuJoCo · PPO | MyoSuite · BodyParts3D/DBCLS · Z-Anatomy',15,MUTED)
        assert np.all(np.asarray(im)[1536:]==BG),(kind,'footer overwritten');return im

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');ap.add_argument('--partial',action='store_true');args=ap.parse_args();d=Draft(args.partial)
    for name,t in[('three_views',34.5),('trajectory',36.2),('fullbody',next(s['start_s']+1.7 for s in d.segments if s['kind']=='final_stage'and s['index']==3)),('four_simultaneous',d.duration-3.5)]:d.at(t).save(OUT/f'preview_{name}.png')
    if args.preview:return
    video=OUT/'standing_long_jump_rl_draft_v4.mp4';w=imageio_ffmpeg.write_frames(str(video),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);w.send(None);count=round(d.duration*30)
    try:
        for k in range(count):w.send(np.asarray(d.at(k/30)))
    finally:w.close()
    atomic(OUT/'draft_delivery.json',dict(video=str(video),frames=count,fps=30,duration_s=count/30,preserved_prefix_s=33,tail_views=d.m['final_views'],simultaneous_final_models=4,final_view='Matched front-oblique from above, fixed pixels/metre; phase synchronized.',fullbody_muscle_force_model=True,unidentified_surfaces_explicit_NA=True,body_color_scale_N=[0,100,1000,30000],force_cards='Right-side muscle group tension / initial successful push-off peak',search_goal_retained_m=3.,physical_torque_caps_unchanged=True));print('V4_VIDEO_READY',video,count/30,flush=True)

if __name__=='__main__':main()


