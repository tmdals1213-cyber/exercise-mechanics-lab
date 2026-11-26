"""Expose only an audited jump to the existing muscle/render adapters."""
from pathlib import Path
import argparse,json,shutil,hashlib
import numpy as np
from retarget_measured_jump import MeasuredJumpRig

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=ROOT/a.folder
 quality=json.loads((folder/'motion_quality.json').read_text())
 if quality.get('force_analysis_allowed')is False or quality.get('status')=='kinematic_review_only':
  raise RuntimeError('Reference motion is still a kinematic review. Complete timing, geometry, contact and dynamics review before force analysis.')
 v=json.loads((folder/'verification.json').read_text());assert all(v['gates'].values()),'Rejected motion cannot enter force analysis'
 assert hashlib.sha256((folder/'motion.npz').read_bytes()).hexdigest()==v['motion_sha256']
 setup=json.loads((folder/'setup.json').read_text());z=np.load(folder/'motion.npz');out=folder/'analysis';out.mkdir(exist_ok=True)
 for name in ['motion.npz','setup.json','verification.json','motion_quality.json','contact_definition.json']:shutil.copy2(folder/name,out/name)
 rig=MeasuredJumpRig(setup['mass_kg'],setup['height_m'])
 summary=dict(status='audited_conditional_jump_precontact_only',movement='cmj_replay',scale=rig.b.report['global_normalization_scale'],body_names=setup['body_names'],phase_times_s=dict(bottom=setup['countermovement_end_s'],start_ascent=setup['countermovement_end_s'],end_ascent=setup['takeoff_s'],end=float(z['times'][-1])),takeoff_velocity_m_s=v['checks']['actual_takeoff_velocity_m_s'],all_mechanics_gates_passed=True,human_motion_validated=False,landing_simulated=False,source_motion_sha256=v['motion_sha256'])
 (folder/'summary.json').write_text(json.dumps(summary,indent=2));print('FORWARD_ANALYSIS_PREPARED',folder,flush=True)
if __name__=='__main__':main()
