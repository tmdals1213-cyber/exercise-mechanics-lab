"""Validate solved full-body loads, anatomical rig, encoded portrait and UI reserve."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ohp_fullbody'
config=json.loads((ROOT/'scenarios/ohp_fullbody.json').read_text(encoding='utf-8'))
data=np.load(OUT/'fullbody_motion.npz');meta=json.loads((OUT/'fullbody_metadata.json').read_text(encoding='utf-8'))
force=data['force_N'];contacts=data['contacts_N'];reserve=data['reserve'];times=data['times'];grips=data['physical_grips']
assert force.shape==(76,416)
assert np.isfinite(force).all() and force.min()>=0 and force.max()<1e6
assert meta['max_balance_error']<.01
assert abs(reserve[:,:6]).max()<1e-4, 'No hidden floating-base support is allowed.'
assert abs(reserve[:,6:]).max()<.1
assert contacts[:,:,2].min()>-.01
assert (abs(contacts[:,:,:2])-.8*contacts[:,:,2,None]).max()<.03
assert abs((grips[:,1,0]-grips[:,0,0])-config['grip_width_m']).max()<.003
assert abs(grips[:,0,1:]-grips[:,1,1:]).max()<.002
assert meta['max_IK_error_m']<.003
assert abs(data['physical_elbow_centres'][0,:,0]-grips[0,:,0]).max()<.01
expected=90*9.81+100*9.80665
assert abs(contacts[times>=2,:,2].sum(axis=1)-expected).max()<.05
rig=json.loads((OUT/'rig_validation.json').read_text());assert rig['max_bone_head_error_m']<1e-4
assert abs(grips[0,:,2].mean()-1.50)<.003
poster=np.asarray(Image.open(OUT/'multiview_poster.png').convert('RGB'))
assert poster.shape==(1920,1080,3)
assert np.all(poster[1536:]==np.array([13,18,25]))
r=imageio.get_reader(OUT/'ohp_multiview_portrait.mp4');info=r.get_meta_data()
assert info['size']==(1080,1920) and abs(info['fps']-30)<.01
for index in [15,45,75,120]:
 image=r.get_data(index);assert image.shape[:2]==(1920,1080)
r.close()
report={'status':'passed','model_actuators':416,'atlas_muscle_surfaces':rig['total_muscle_meshes'],
 'mapped_surfaces':rig['mapped_meshes'],'frames':76,'video_resolution':list(info['size']),'fps':info['fps'],
 'bottom_reserved_px':384,'max_force_balance_error':meta['max_balance_error'],
 'max_root_reserve':float(abs(reserve[:,:6]).max()),'max_other_reserve':float(abs(reserve[:,6:]).max()),
 'max_rig_bone_error_m':rig['max_bone_head_error_m'],'steady_vertical_ground_support_N':float(contacts[-1,:,2].sum()),
 'unmapped_surfaces':'grey-brown, no invented tension','physics_length_scale':meta['physics_length_scale'],
 'physical_bar_start_height_m':float(grips[0,:,2].mean()),'physical_grip_width_m':float(grips[0,1,0]-grips[0,0,0])}
(OUT/'validation_fullbody.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
