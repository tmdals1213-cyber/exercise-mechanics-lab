"""A declared conservative arrival-pose domain for an initial solution only."""
import numpy as np
from jump_phase_objective import phase_metrics

INITIAL_CONTACT_ENVELOPE=dict(minimum_COM_height_m=.55,minimum_nonfoot_clearance_proxy_m=.35,maximum_abs_torso_pitch_rad=1.10,
    purpose='Conservative initial-pose search domain, not an empirical human limit or a recovery guarantee.')

def evaluate_launch(report,motion,model):
    metrics=phase_metrics(report,motion,model);td=report.get('touchdown_s')
    if td is None:height=None
    else:height=float(report['touchdown_COM_m'][2])if'touchdown_COM_m'in report else float(motion['com'][int(np.argmin(abs(motion['times']-td))),2])
    pitch=report.get('landing_torso_pitch_rad');domain=bool(height is not None and height>=INITIAL_CONTACT_ENVELOPE['minimum_COM_height_m']and metrics['precontact_nonfoot_clearance_proxy_m']>=INITIAL_CONTACT_ENVELOPE['minimum_nonfoot_clearance_proxy_m']and pitch is not None and abs(pitch)<=INITIAL_CONTACT_ENVELOPE['maximum_abs_torso_pitch_rad'])
    metrics.update(contact_COM_height_m=height,contact_torso_pitch_rad=pitch,initial_contact_envelope_eligible=domain,optimization_domain_eligible=metrics['precontact_dynamics_eligible']and domain)
    return metrics
