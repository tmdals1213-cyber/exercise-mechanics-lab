"""Render-only metadata for the measured full-cycle jump; never releases loads."""
from pathlib import Path
import json,hashlib
import numpy as np
from retarget_measured_jump import MeasuredJumpRig

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'results/measured_jump_analysis/s10_CMJ_t1_full_cycle'
    z=np.load(folder/'motion.npz');setup=json.loads((folder/'setup.json').read_text());verification=json.loads((folder/'verification.json').read_text())
    digest=hashlib.sha256((folder/'motion.npz').read_bytes()).hexdigest();assert digest==verification['motion_sha256']
    r=MeasuredJumpRig(setup['mass_kg'],setup['height_m']);r.b.pose(r.b.q0)
    pre=np.flatnonzero(z['times']<setup['takeoff_s']);setup['countermovement_end_s']=float(z['times'][pre[np.argmin(z['com'][pre,2])]])
    (folder/'setup.json').write_text(json.dumps(setup,indent=2))
    quality=json.loads((folder/'motion_quality.json').read_text());quality.update(status='kinematic_review_only',force_analysis_allowed=False,landing_force_validated=False)
    (folder/'motion_quality.json').write_text(json.dumps(quality,indent=2))
    np.savez_compressed(folder/'reference_pose.npz',body_xpos=r.d.xpos.copy(),body_xmat=r.d.xmat.copy())
    phases={'bottom':setup['countermovement_end_s'],'start_ascent':setup['countermovement_end_s'],'end_ascent':setup['takeoff_s'],'landing':setup['landing_s'],'end':float(z['times'][-1])}
    summary=dict(status='measured_full_cycle_kinematic_review',movement='cmj_replay',scale=r.b.report['global_normalization_scale'],body_names=setup['body_names'],phase_times_s=phases,
                 measured_replay=True,landing_included=True,landing_simulated=False,force_analysis_allowed=False,human_motion_validated=False,source_motion_sha256=digest)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    stats={}
    regions={'propulsion':(z['times']>=setup['countermovement_end_s'])&(z['times']<setup['takeoff_s']-.03),'flight':(z['times']>setup['takeoff_s']+.03)&(z['times']<setup['landing_s']-.03),'landing_and_recovery':z['times']>=setup['landing_s']}
    for name,mask in regions.items():
        stats[name]=dict(samples=int(mask.sum()),root_force_RMS_N=np.sqrt(np.mean(z['root_force_residual_N'][mask]**2,axis=0)).tolist(),root_moment_RMS_Nm=np.sqrt(np.mean(z['root_moment_residual_Nm'][mask]**2,axis=0)).tolist(),peak_measured_GRF_N=float(z['force_world_N'][mask,2].max()))
    report=dict(status='measured_motion_review_only',motion_sha256=digest,phase_residuals=stats,root_residuals_applied_as_forces=False,force_analysis_allowed=False,
                active_stance_vertical_RMS_bodyweights=verification['checks']['active_vertical_force_RMS_bodyweights'],
                limitation='The reported stance error does not establish landing-force accuracy, individual muscle-force accuracy or generated box-jump validity.',
                source_paper='https://doi.org/10.1038/s41597-025-05934-5',source_data='https://doi.org/10.6084/m9.figshare.28890545')
    report['force_agreement_inherited_from_reference_COM']=setup.get('force_agreement_inherited_from_reference_COM',False)
    report['COM_trajectory_independently_measured']=False
    (folder/'full_cycle_review.json').write_text(json.dumps(report,indent=2));(folder/'tracking_report.json').write_text(json.dumps(dict(source=report['source_data'])))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
