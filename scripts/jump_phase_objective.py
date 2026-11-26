"""Separate pre-contact jump performance from the current recovery controller.

This metric never certifies a legal/landable long-jump record. It keeps a later
fall from erasing a reproducible launch or heel position at first foot contact.
Existing supported-recovery training and its historical scores are unchanged.
"""
import numpy as np

def phase_metrics(report, motion, model,continuous_free_flight=False):
    t=motion['times'];to=report.get('takeoff_s');td=report.get('touchdown_s')
    end=td if td is not None else t[-1];pre=t<=end
    limits=model.jnt_range[3:]
    q=motion['qpos'][pre,3:]
    excess=float(np.maximum(q-limits[:,1],limits[:,0]-q).max())
    clear=[]
    for name,radius in [('head',.10),('torso',.14),('pelvis',.10)]:
        bid=model.body(name).id
        xyz=motion['body_xpos'][pre,bid]+np.einsum('nij,j->ni',motion['body_xmat'][pre,bid].reshape(-1,3,3),model.body_ipos[bid])
        clear.append(float((xyz[:,2]-radius).min()))
    ft=motion['physics_times_s'];force=motion['physics_force_world_N'];ground=ft<(to if to is not None else np.inf)
    peak=float(force[ground,2].max()/(87*9.80665))if ground.any()else 0.
    # The native touchdown event uses a 30N threshold. A sample just before
    # that event may already carry a small contact force; it must not be used
    # to demand airborne angular-momentum conservation.
    flight=(t>to+.005)&(t<td-.005 if td is not None else True)&(np.linalg.norm(motion['force_world_N'],axis=1)<1.)if to is not None else np.zeros(len(t),bool)
    audit_scope='Contact-free samples (<1N), excluding 5ms event margins; later recovery is not included.'
    reference_time=to
    if continuous_free_flight:
        # Event detection is evaluated before the current controller/contact
        # solve. Brief residual/re-contacts can follow that event. Conservation
        # must use an interval with no external contact in the PHYSICS ledger,
        # referenced to its own first state, rather than the earlier event H.
        free=(ft>to+.005)&(ft<(td-.005 if td is not None else np.inf))&(np.linalg.norm(force,axis=1)<.1)if to is not None else np.zeros(len(ft),bool)
        edges=np.diff(np.r_[False,free,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
        flight=np.zeros(len(t),bool)
        if len(starts):
            j=int(np.argmax(ft[ends]-ft[starts]));a=ft[starts[j]];b=ft[ends[j]]
            if b-a>=.10:flight=(t>=a+2*report['physics_dt_s'])&(t<=b-2*report['physics_dt_s'])
        audit_scope='Longest continuous native-contact-free physics interval (<0.1N), excluding5ms event margins and2steps at interval edges; conservation referenced to first sampled state. Earlier event/contact impulses excluded.'
    ballistic=None;H_error=None
    if flight.any():
        first=int(np.flatnonzero(flight)[0]);reference_time=float(t[first])if continuous_free_flight else to
        initial=motion['com'][first]if continuous_free_flight else np.array(report['takeoff_COM_m']);velocity=motion['velocity'][first]if continuous_free_flight else np.array(report['takeoff_velocity_m_s']);elapsed=t[flight]-reference_time;expected=initial+elapsed[:,None]*velocity;expected[:,2]-=.5*9.80665*elapsed**2
        ballistic=float(np.linalg.norm(motion['com'][flight]-expected,axis=1).max())
        H0=motion['angular_momentum'][first,0]if continuous_free_flight else report['takeoff_H_kg_m2_s'][0]
        H_error=float(abs(motion['angular_momentum'][flight,0]-H0).max())
    eligible=bool(to is not None and td is not None and flight.any()and report['maximum_root_applied_force_or_torque']==0 and excess<.04 and min(clear)>.10 and peak<5.5 and ballistic is not None and ballistic<.008 and H_error<.05)
    return dict(precontact_dynamics_eligible=eligible,
                heel_projection_at_first_foot_contact_m=report.get('first_contact_heel_distance_m'),
                takeoff_velocity_m_s=report.get('takeoff_velocity_m_s'),
                takeoff_COM_m=report.get('takeoff_COM_m'),
                takeoff_angular_momentum_kg_m2_s=report.get('takeoff_H_kg_m2_s'),
                takeoff_s=to,first_foot_contact_s=td,
                precontact_ROM_excess_rad=excess,precontact_nonfoot_clearance_proxy_m=min(clear),
                propulsion_peak_GRF_BW=peak,flight_ballistic_COM_error_m=ballistic,
                flight_angular_momentum_error_kg_m2_s=H_error,
                flight_audit_samples=int(flight.sum()),flight_audit_scope=audit_scope,flight_audit_reference_time_s=reference_time,
                recovery_evaluated=bool(td is not None and(report.get('fell')or t[-1]>=td+.5)),
                current_controller_recovery_stable=bool(report['stable_landing'])if td is not None and(report.get('fell')or t[-1]>=td+.5)else None,
                current_controller_supported_distance_m=report.get('valid_contact_distance_m'),
                official_jump_record_certified=False,
                human_recoverability_determined=False,
                metric_scope='Heel position at native first foot contact relative to initial toe line; later recovery is separate. Not the nearest whole-body contact or official test score.')

def launch_distance_reward(metrics):
    """Proposed primary metric; recovery flags never enter this phase reward."""
    distance=metrics.get('heel_projection_at_first_foot_contact_m')
    return float(distance)if metrics['precontact_dynamics_eligible']and distance is not None else -1.
