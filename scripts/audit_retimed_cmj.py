"""Audit a prescribed, projected-contact inverse-dynamics illustration.

This audit deliberately differs from the strict sticking-contact forward solver.
Passing permits a conditional visualization, not a validated human prediction.
"""
from pathlib import Path
import json,argparse,hashlib
import numpy as np
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation
from retarget_measured_jump import MeasuredJumpRig
from mechanics_engine_adapter import sample

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',required=True);ap.add_argument('--compare');a=ap.parse_args()
    p=ROOT/a.folder;z=np.load(p/'motion.npz');s=json.loads((p/'setup.json').read_text());r=MeasuredJumpRig()
    t=z['times'];to=s['takeoff_s'];stance=t<to-1e-8;flight=t>=to-1e-8;W=s['mass_kg']*9.80665;L=s['height_m']
    residual=z['root_residual'];F=z['contact_forces_N'];normal=F[:,:,2];gap=z['material_contact_positions_m'][:,:,2]
    rms_force=float(np.sqrt(np.mean(np.sum(residual[stance,:3]**2,axis=1))))
    rms_moment=float(np.sqrt(np.mean(np.sum(residual[stance,3:]**2,axis=1))))
    max_moment=float(np.linalg.norm(residual[stance,3:],axis=1).max())
    weighted_gap=float(np.sqrt(np.sum(normal*np.square(gap))/normal.sum()))
    cone=float(np.max(np.linalg.norm(F[:,:,:2],axis=2)-.8*normal))
    base=np.flatnonzero(flight)[0];air_t=t[flight]-to;com=z['com'][flight];v=z['velocity'][base]
    predicted=z['com'][base]+air_t[:,None]*v+.5*air_t[:,None]**2*np.array([0,0,-9.80665])
    ballistic=float(np.linalg.norm(com-predicted,axis=1).max())
    H=z['angular_momentum'][flight];Herror=float(np.linalg.norm(H-H[0],axis=1).max())
    measured_height=float(com[:,2].max()-z['com'][base,2]);standing=float(com[:,2].max()-z['com'][t<=.25,2].mean())
    ref=ROOT/'results/power_comparison/cmj_analysis/motion.npz'
    source_unchanged=hashlib.sha256(ref.read_bytes()).hexdigest()==json.loads((p/'candidate_summary.json').read_text())['reference_motion_sha256']
    definitions=json.loads((p/'contact_definition.json').read_text())['points'];cross=0.;rom=0.
    for i in np.unique(np.r_[np.linspace(0,len(t)-1,25).astype(int),base]):
        contacts=[(x['body_id'],z['contact_positions_m'][i,j],F[i,j],np.zeros(3))for j,x in enumerate(definitions)]
        mech=sample(r,z['qpos'][i],z['qvel'][i],z['qacc'][i],contacts);cross=max(cross,mech['crosscheck_max'])
    ids=np.array(r.b.independent);angles=z['qpos'][:,r.ids];bounds=r.m.jnt_range[ids]
    rom=float(np.maximum(bounds[:,0]-angles,angles-bounds[:,1]).max())
    # Foot-tip material points are not a Sargent reach measurement.
    apex=int(np.argmax(z['com'][:,2]));tip_ids=[j for j,x in enumerate(definitions)if x['forefoot'] and j%6>=4]
    toe_clearance=float(z['material_contact_positions_m'][apex,tip_ids,2].mean())
    checks=dict(stance_force_RMS_N=rms_force,stance_force_RMS_BW=rms_force/W,
        stance_moment_RMS_Nm=rms_moment,stance_moment_RMS_BW_height=rms_moment/(W*L),
        stance_moment_max_Nm=max_moment,stance_moment_max_BW_height=max_moment/(W*L),
        force_weighted_projected_contact_gap_RMS_m=weighted_gap,contact_normal_min_N=float(normal.min()),
        friction_cone_max_excess_N=cone,independent_NE_max_error_SI=cross,
        flight_ballistic_max_error_m=ballistic,flight_H_max_drift_kg_m2_s=Herror,
        ROM_excess_rad=max(rom,0),airborne_COM_height_m=measured_height,
        standing_to_apex_COM_rise_m=standing,toe_tip_clearance_at_COM_apex_m=toe_clearance,
        peak_GRF_N=float(z['force_world_N'][:,2].max()),takeoff_vertical_velocity_m_s=float(v[2]))
    gates=dict(independent_newton_euler=cross<1e-5,source_preserved=source_unchanged,
        translational_balance_RMS_under_one_percent_BW=rms_force/W<.01,
        rotational_residual_RMS_under_five_percent_BW_height=rms_moment/(W*L)<.05,
        unilateral_contact=float(normal.min())>-1e-5,friction_cone=cone<1e-5,
        airborne_height_within_one_percent=abs(measured_height-.3)<.003,
        flight_ballistic=ballistic<.001,flight_angular_momentum=Herror<.1,anatomical_ranges=rom<.002)
    if a.compare:
        other=json.loads((ROOT/a.compare/'candidate_summary.json').read_text());delta=abs(checks['peak_GRF_N']/other['peak_GRF_N']-1)
        checks['sample_rate_peak_GRF_relative_difference']=delta;gates['sample_rate_convergence_under_one_percent']=delta<.01
    digest=hashlib.sha256((p/'motion.npz').read_bytes()).hexdigest()
    report=dict(status='prescribed_projected_contact_diagnostic',motion_sha256=digest,gates=gates,checks=checks,
        human_muscle_force_validated=False,exact_sticking_contact=False,force_residual_applied_as_external_support=False,
        limitations=['Reference-shaped prescribed motion, not a newly measured jump.',
            'Ground support is the projection of foot material points. Contact slip/gap is not resolved by a forward contact model.',
            'Rotational root residual is reported and is not converted into an invented external support torque.',
            'Numerical checks do not establish 10% accuracy for individual muscle forces.'])
    (p/'verification.json').write_text(json.dumps(report,indent=2))
    quality=dict(status='passed' if all(gates.values()) else 'failed',motion_sha256=digest,
        scope='Conditional inverse-dynamics illustration, projected contacts; no landing.',exact_sticking_contact=False)
    (p/'motion_quality.json').write_text(json.dumps(quality,indent=2));print(json.dumps(report),flush=True)
    assert all(gates.values()),gates

if __name__=='__main__':main()
