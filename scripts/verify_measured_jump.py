"""Check the delivered conditional analysis and its portrait media."""
import json,hashlib,os
from pathlib import Path
from html.parser import HTMLParser
import numpy as np
import imageio_ffmpeg
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/os.environ.get('EXERCISE_JUMP_SOURCE','results/measured_jump_analysis/s10_CMJ_t1');out=ROOT/os.environ.get('EXERCISE_JUMP_VISUALS','results/measured_jump_visualization');motion=np.load(source/'motion.npz');tension=np.load(source/'tensions.npz');report=json.loads((source/'tension_summary.json').read_text());data=json.loads((out/'data.json').read_text());summary=json.loads((out/'analysis_summary.json').read_text());check=json.loads((source/'verification.json').read_text());setup=json.loads((source/'setup.json').read_text());raw=np.load(ROOT/'results/measured_cmj/s10_CMJ_t1_measured.npz');C=np.array([[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]])
    assert hashlib.sha256((source/'motion.npz').read_bytes()).hexdigest()==report['source_motion_sha256']==check['motion_sha256'];assert len(data['times'])==len(tension['times']);assert data['mass_kg']==88 and data['muscle_count']==438;assert not summary['root_residual_applied'] and not summary['muscle_force_validated'];assert check['ten_percent_reference_agreement']
    ids=np.array([np.argmin(abs(raw['times']-tt-setup['source_start_s'])) for tt in motion['times']]);assert np.max(abs(motion['contact_forces_N']-raw['applied_ground_force_N'][ids]@C.T))<1e-10;assert np.max(abs(motion['contact_free_moments_Nm']-raw['applied_free_moment_Nm'][ids]@C.T))<1e-10
    solved=[i for i,n in enumerate(report['coordinate_names']) if not n.startswith('mtp_angle_')];residual=np.einsum('tqn,tn->tq',tension['R'][:,solved],tension['force_N'])-tension['target'][:,solved];assert abs(residual).max()<.01;assert np.isfinite(tension['force_N']).all() and tension['force_N'].min()>-1e-6
    mapping=json.loads((out/'render_mapping.json').read_text());assert mapping['same_values_front_back'];assert all(any(m['key']==key and m['group_id']>0 for m in mapping['meshes']) for key in ['latissimus','trapezius','serratus']);regions=json.loads((out/'joint_region_mapping.json').read_text());assert min(regions['rest_face_counts'])>0
    for i in range(len(data['times'])):
        im=np.array(Image.open(out/f'portrait_frames/frame_{i:04d}.png').convert('RGB'));assert im.shape==(1920,1080,3);assert np.all(im[1536:]==[13,23,38])
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            d=dict(attrs);link=d.get('href') or d.get('src')
            if link and not link.startswith(('http','#')):self.links.append(link)
    parser=Links();parser.feed((out/'index.html').read_text(encoding='utf-8'))
    for link in parser.links:assert (out/link).exists(),link
    reader=imageio_ffmpeg.read_frames(str(out/'jump_tension_portrait.mp4'),pix_fmt='rgb24');meta=next(reader);assert tuple(meta['size'])==(1080,1920);count=0;max_footer_error=0
    for raw_frame in reader:
        a=np.frombuffer(raw_frame,np.uint8).reshape(1920,1080,3);max_footer_error=max(max_footer_error,int(abs(a[1536:].astype(int)-[13,23,38]).max()));count+=1
    video=json.loads((out/'video_checks.json').read_text());assert count==video['frames'] and max_footer_error<=5
    result=dict(status='conditional_analysis_and_media_checks_passed',force_samples=len(data['times']),muscle_elements=438,internal_moment_balance_error_Nm=float(abs(residual).max()),measured_force_amplitudes_preserved=True,free_moment_preserved=True,active_vertical_force_RMS_bodyweights=check['checks']['active_vertical_force_RMS_bodyweights'],root_moment_RMS_Nm=check['checks']['root_moment_RMS_Nm'],human_muscle_force_validation=False,video_frames=count,portrait=[1080,1920],bottom_reserved_px=384,decoded_footer_max_error=max_footer_error,back_muscle_color_mapping_passed=True,all_joint_surface_regions_present=True,local_links=len(parser.links))
    (out/'delivery_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
