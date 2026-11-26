"""Decode the delivered video, including the reserved portrait footer."""
import json,numpy as np,imageio_ffmpeg
from jump_rig import ROOT

def main():
    out=ROOT/'results/jump_contact_visualization';reader=imageio_ffmpeg.read_frames(str(out/'jump_tension_upright_v3.mp4'),pix_fmt='rgb24');meta=next(reader);width,height=meta['size'];assert (width,height)==(1080,1920)
    count=0;variation=0;displacement=0
    for raw in reader:
        pixels=np.frombuffer(raw,np.uint8).reshape(height,width,3);footer=pixels[1536:].astype(int)
        variation=max(variation,int(np.ptp(footer,axis=(0,1)).max()));displacement=max(displacement,int(abs(footer-np.array([13,23,38])).max()));assert variation<=3 and displacement<=5
        count+=1
    expected=json.loads((out/'video_checks.json').read_text());assert count==expected['frames']
    result=dict(status='passed',decoded_frames=count,size=[width,height],footer_reserved_px=384,footer_max_channel_variation=variation,footer_max_background_deviation=displacement)
    (out/'decoded_video_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
