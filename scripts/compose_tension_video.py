"""Encode the verified portrait frames; repeated display is not new simulation."""
import json,argparse,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/tension_visualization'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repetitions',type=int,default=3);a=ap.parse_args();files=sorted((OUT/'portrait_frames').glob('frame_*.png'));data=json.loads((OUT/'data.json').read_text(encoding='utf-8'))
    if len(files)!=len(data['times']):raise RuntimeError('Incomplete frame capture')
    expected=np.array([13,23,38],dtype=np.uint8)
    for path in files:
        image=np.array(Image.open(path).convert('RGB'))
        if image.shape!=(1920,1080,3):raise RuntimeError('Wrong frame size')
        if abs(image[1536:].astype(int)-expected).max()>0:raise RuntimeError('Bottom safe region contains content')
    output=OUT/'jump_wholebody_muscle_portrait.mp4';writer=imageio_ffmpeg.write_frames(str(output),(1080,1920),fps=data['fps'],codec='libx264',quality=None,macro_block_size=1,pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','18','-preset','medium','-movflags','+faststart']);writer.send(None)
    try:
        for repeat in range(a.repetitions):
            for path in files:writer.send(np.array(Image.open(path).convert('RGB')))
    finally:writer.close()
    reader=imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24');meta=next(reader);first=np.frombuffer(next(reader),dtype=np.uint8).reshape(1920,1080,3);reader.close();count,seconds=imageio_ffmpeg.count_frames_and_secs(str(output))
    if tuple(meta['size'])!=(1080,1920) or count!=len(files)*a.repetitions:raise RuntimeError('Encoded video dimensions/frame count mismatch')
    footer=first[1544:,8:-8].reshape(-1,3)
    if np.ptp(footer,axis=0).max()>4:raise RuntimeError('Encoded safe area is not uniform')
    shutil.copy2(output,OUT/'tension_dual_view_portrait.mp4')
    for name,source in [('anatomy_preview.png','poster.png'),('humanoid_preview.png','humanoid_poster.png')]:
        poster=Image.open(OUT/source).convert('RGB');preview=Image.new('RGB',(1080,720),(13,23,38));preview.paste(poster.crop((0,135,1080,182)),(0,0));preview.paste(poster.crop((0,335,1080,975)),(0,60));preview.save(OUT/name)
        if name=='humanoid_preview.png':preview.save(OUT/'smooth_preview.png')
    Image.open(OUT/'wholebody_phase_27.png').crop((0,0,1080,1536)).save(OUT/'wholebody_preview.png')
    neutral=Image.open(OUT/'neutral_pose.png').convert('RGBA');box=neutral.getbbox();neutral=neutral.crop((max(0,box[0]-35),max(0,box[1]-25),min(neutral.width,box[2]+35),min(neutral.height,box[3]+25)));neutral.thumbnail((972,780))
    card=Image.new('RGB',(1080,1020),(13,23,38));card.paste(neutral,((1080-neutral.width)//2,160),neutral)
    draw=ImageDraw.Draw(card);font=lambda size:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    draw.text((54,40),'RELAXED STANDING REFERENCE',font=font(38),fill=(237,243,251));draw.text((54,96),'Same skeleton / illustrative pose / no tension assigned',font=font(25),fill=(168,190,209));draw.text((54,975),'Separate from the prescribed jump-ready pose',font=font(24),fill=(168,190,209));card.save(OUT/'neutral_pose_card.png')
    report={'status':'passed','video':output.name,'width':1080,'height':1920,'fps':meta['fps'],'unique_frames':len(files),'display_repetitions':a.repetitions,'output_frames':count,'duration_s':seconds,'bottom_reserved_px':384,'safe_region_exactly_blank':True,'encoded_footer_uniform':True,'repetitions_are_replays_of_one_reference':True};(OUT/'video_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
