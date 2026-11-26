"""Continuous landing feedback. A valid launch keeps its distance after a fall."""
import numpy as np
from jump_phase_objective import phase_metrics

def score(report,z,model):
    phase=phase_metrics(report,z,model);td=report.get('touchdown_s');D=report.get('first_contact_heel_distance_m')
    body=report.get('first_body_floor_contact_s');fell=bool(report['fell'])
    if td is None:return dict(distance_m=D,stability=0.,fell=fell,reward=-3.,eligible=False,phase=phase)
    late=z['times']>=max(td,float(z['times'][-1])-.25)
    speed=float(np.linalg.norm(z['velocity'][late],axis=1).mean())
    height=float(z['com'][late,2].mean())
    H=float(abs(z['angular_momentum'][late,0]).mean())
    survival=float(np.clip(((body if body is not None else z['times'][-1])-td)/2.5,0,1))
    upright=float(np.clip((height-.25)/.60,0,1))
    quiet=float(np.exp(-speed/.6-H/20.))
    S=float(survival*upright*quiet)
    eligible=bool(phase['precontact_dynamics_eligible'])
    reward=float(D+.8*S-.8*float(fell))if eligible and D is not None else -3.
    return dict(distance_m=D,stability=S,fell=fell,reward=reward,eligible=eligible,
                distance_contribution_m=D,stability_contribution_m=.8*S,fall_penalty_m=.8*float(fell),
                survival_fraction=survival,late_COM_speed_m_s=speed,late_COM_height_m=height,
                late_abs_Hx_kg_m2_s=H,body_floor_contact_s=body,stable=bool(report['stable_landing']),phase=phase,
                reward_scope='D + 0.8*S - 0.8*fall, meter-equivalent design weights; S is a model recovery score, not human success probability. Falls keep D. Invalid precontact dynamics receive -3.')
