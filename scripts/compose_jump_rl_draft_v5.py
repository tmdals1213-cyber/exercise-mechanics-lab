"""Preserve the approved opening, then add four simultaneous fixed-side jumps."""
from pathlib import Path
import argparse,copy,hashlib
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
import compose_jump_rl_draft_v4 as previous
from compose_jump_rl_draft_v4 import txt,ease,BG,FG,MUTED,MINT,ACC,DIM,STAGE_COLORS
from jump_monitor_data import ROOT,read,atomic
from jump_trial_provenance import collect
OUT=ROOT/'deliverables/jump_rl_study_draft_v5'

class Draft:
    def __init__(self,layout_preview=False):
        self.prefix=previous.Draft();self.preview=layout_preview;OUT.mkdir(parents=True,exist_ok=True)
        prefix_times=[0.,1.,8.,16.,24.,32.9]
        prefix_hashes=[hashlib.sha256(np.asarray(self.prefix.at(t)).tobytes()).hexdigest()for t in prefix_times]
        if layout_preview:self.core=self.prefix
        else:
            previous.OUT=OUT;previous.base.OUT=OUT;previous.base.previous.OUT=OUT;self.core=previous.Draft()
        assert prefix_hashes==[hashlib.sha256(np.asarray(self.prefix.at(t)).tobytes()).hexdigest()for t in prefix_times],'Approved opening changed after final source switch'
        atomic(OUT/'opening_preservation_check.json',dict(passed=True,sample_times_s=prefix_times,raw_frame_sha256=prefix_hashes))
        self.provenance=collect(self.core.m);atomic(OUT/'trial_provenance.json',self.provenance);self.complete=bool(self.core.m.get('stable_three_metre_complete'));self.segments=copy.deepcopy(self.core.segments[:-1]);start=self.core.segments[-1]['start_s'];self.segments.extend([dict(kind='side_four',start_s=start,end_s=start+8.),dict(kind='simultaneous_four',start_s=start+8.,end_s=start+15.)]);self.duration=start+15.
        atomic(OUT/'timeline_v5.json',dict(duration_s=self.duration,fps=30,preserved_prefix_s=33.,segments=self.segments,side_four_fixed_world=True,side_four_same_body_scale=True,phase_alignment='Corresponding LOAD/PUSH/AIR/LAND events aligned for comparison; original simulated time and pose unchanged.',trial_count_definition=self.provenance['count_definition'],complete=self.complete))

    def trial(self,key):return self.provenance['cases'][key]['display_label']

    def side_four(self,segment,local):
        im,d=self.core.base();txt(d,45,78,'SAME LIMITS · SIDE VIEW',44,FG,True);u=min(1.,local/7.5);phase=0 if u<.34 else 1 if u<.52 else 2 if u<.72 else 3
        for j,name in enumerate(['LOAD','PUSH','AIR','LAND']):
            x=45+j*252;d.rounded_rectangle((x,160,x+234,205),radius=9,fill=MINT if phase==j else DIM);txt(d,x+117,168,name,25,BG if phase==j else MUTED,True,anchor='mt')
        for i,key in enumerate(self.core.keys):
            c=self.core.cases[key];D=self.core.data[key];r=D['r'];cam=D['cam'];z=D['z'];t=float(np.interp(u,[0,.12,.34,.52,.72,1.],self.core.metrics['cases'][key]['phase_times']));n=round(c['duration_s']*24);fi=int(np.clip(round(t/c['duration_s']*n),0,n));native=fi/n*c['duration_s'];scale=.5;top=260+i*305;baseline=top+295;ppm=1080/cam['ortho_width_m'];paste_y=baseline-scale*(525+cam['camera_center_z_m']*ppm);zero=520.;paste_x=zero-scale*(540-cam['camera_center_forward_m']*ppm);folder=Path(c['multiview_folder'])/c['full_side_dir']
            with Image.open(folder/f'frame_{fi:04d}.png')as raw:small=raw.convert('RGBA').resize((540,525),Image.Resampling.LANCZOS)
            im.paste(small,(round(paste_x),round(paste_y)),small);d=ImageDraw.Draw(im);txt(d,45,top+60,'INITIAL'if i==0 else'FINAL'if i==3 else f'IMPROVEMENT {i}',27,MUTED,True);txt(d,45,top+105,f'{c["checked_distance_m"]:.2f} m',45,STAGE_COLORS[i],True);txt(d,45,top+164,self.trial(key),28,MUTED,True);d.line((450,baseline,1000,baseline),fill=(55,77,95),width=2);d.line((zero,baseline-7,zero,baseline+7),fill=FG,width=3)
            if native>=r['touchdown_s']:
                end=zero+scale*r['first_contact_heel_distance_m']*ppm;d.line((zero,baseline,end,baseline),fill=STAGE_COLORS[i],width=4);ids=np.flatnonzero((z['times']>=r['takeoff_s'])&(z['times']<=r['touchdown_s']));count=max(1,round(ease((native-r['touchdown_s'])/.55)*len(ids)));points=[(paste_x+scale*(540+(-p[1]+cam['initial_foremost_sole_y_m']-cam['camera_center_forward_m'])*ppm),paste_y+scale*(525-(p[2]-cam['camera_center_z_m'])*ppm))for p in z['com'][ids[:count]]]
                if len(points)>1:d.line(points,fill=STAGE_COLORS[i],width=4)
        return im

    def at(self,t):
        if t<33:return self.prefix.at(t)
        seg=next((s for s in self.segments if s['start_s']<=t<s['end_s']),self.segments[-1]);local=max(0.,t-seg['start_s'])
        if seg['kind']=='side_four':im=self.side_four(seg,local)
        else:
            im=self.core.render(seg,local);d=ImageDraw.Draw(im)
            if seg['kind']=='final_stage':txt(d,1035,204,self.trial(seg['case']),29,MUTED,True,anchor='rt')
            elif seg['kind']=='simultaneous_four':
                for i,key in enumerate(self.core.keys):txt(d,270+(i%2)*540,338+(i//2)*610,self.trial(key),26,MUTED,True,anchor='mt')
                d.rectangle((0,1430,1079,1535),fill=BG);txt(d,45,1450,'3 m+ · STABLE LANDING'if self.complete else'FINAL COMPARISON LAYOUT',33,ACC,True);txt(d,45,1508,'MuJoCo · PPO | MyoSuite · BodyParts3D/DBCLS · Z-Anatomy',15,MUTED)
        assert np.all(np.asarray(im)[1536:]==BG),'Footer overwritten';return im

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layout-preview',action='store_true');ap.add_argument('--preview',action='store_true');ap.add_argument('--duration',type=float,default=59.8,help='Presentation duration; first33s remain at their approved timing.');ap.add_argument('--output');args=ap.parse_args();draft=Draft(args.layout_preview)
    side=next(s for s in draft.segments if s['kind']=='side_four');draft.at(side['start_s']+6.2).save(OUT/'preview_four_side.png');draft.at(draft.duration-3.5).save(OUT/'preview_four_oblique.png')
    if args.preview or args.layout_preview:return
    count=round(args.duration*30);duration=count/30;assert duration>33.;speed=(draft.duration-33.)/(duration-33.);video=Path(args.output)if args.output else OUT/'standing_long_jump_rl_final.mp4';video.parent.mkdir(parents=True,exist_ok=True)
    timeline=read(OUT/'timeline_v5.json');timeline['source_duration_s']=draft.duration;timeline['duration_s']=duration;timeline['presentation_speed_after_33s']=speed
    for segment in timeline['segments']:
        for field in ['start_s','end_s']:
            if segment[field]>33.:segment[field]=33.+(segment[field]-33.)/speed
    atomic(OUT/'timeline_v5.json',timeline)
    writer=imageio_ffmpeg.write_frames(str(video),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);writer.send(None)
    try:
        for i in range(count):
            t=i/30;source_t=t if t<33. else 33.+(t-33.)*speed;writer.send(np.asarray(draft.at(min(source_t,draft.duration-1e-6))))
    finally:writer.close()
    atomic(OUT/'draft_delivery.json',dict(video=str(video),frames=count,fps=30,duration_s=duration,source_duration_s=draft.duration,presentation_speed_after_33s=speed,presentation_timing_only=True,physical_simulation_unchanged=True,preserved_prefix_s=33,side_four_simultaneous=True,oblique_four_simultaneous=True,phase_aligned=True,stable_three_metre_complete=draft.complete,trial_provenance=draft.provenance));print('V5_VIDEO_READY',str(video),flush=True)

if __name__=='__main__':main()
