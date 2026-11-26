"""Verify delivered aggregation, force-time integrals, video format and safe area."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ohp_100kg_2s'
d=json.loads((OUT/'muscle_motion.json').read_text(encoding='utf-8'))
frames=d['frames'];names=d['muscle_names'];t=np.array([f['time_s'] for f in frames]);F=np.array([f['muscle_forces_N'] for f in frames])
assert F.shape==(151,50)
assert np.isfinite(F).all() and (F>=-1e-5).all()
assert d['summary']['max_equilibrium_error_Nm']<.1
assert d['summary']['max_sample_position_error_m']<.004
assert abs(frames[-1]['elbow_flexion_deg'])<.001
for key,g in d['groups'].items():
 ids=[names.index(n) for n in g['muscles']]
 force=F[:,ids].sum(axis=1)
 assert np.max(abs(force-np.array(g['force_N'])))<.002,key
 impulse=np.sum((force[1:]+force[:-1])/2*np.diff(t))
 assert abs(impulse-g['total_impulse_Ns'])<.002,key
poster=np.asarray(Image.open(OUT/'portrait_poster.png').convert('RGB'))
assert poster.shape==(1920,1080,3)
assert np.all(poster[1536:]==np.array([13,18,25])), 'Bottom 20% must be empty.'
reader=imageio.get_reader(OUT/'ohp_100kg_portrait.mp4')
metadata=reader.get_meta_data();reader.close()
assert metadata['size']==(1080,1920)
assert abs(metadata['fps']-30)<.01
report={'status':'passed','muscles':50,'frames':len(frames),'video_size':list(metadata['size']),
        'video_fps':metadata['fps'],'video_duration_s':metadata['duration'],
        'bottom_blank_px':384,'max_equilibrium_error_Nm':d['summary']['max_equilibrium_error_Nm'],
        'max_IK_error_m':d['summary']['max_sample_position_error_m'],
        'balance_scope':'shoulder elevation + elbow flexion ONLY',
        'bounded_reference_feasible_frame_fraction':d['summary']['bounded_feasible_frame_fraction']}
(OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
