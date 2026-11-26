"""Versioned, additive recovery shaping; preserve all first-contact attempts."""
import numpy as np
from jump_phase_objective import phase_metrics

VERSION='additive_recovery_v2.1_continuous_flight_audit'
TARGET_FIRST=2.85
TARGET_RETAINED=2.80

def score(report,z,model):
    phase=phase_metrics(report,z,model,continuous_free_flight=True);td=report.get('touchdown_s');D=report.get('first_contact_heel_distance_m');body=report.get('first_body_floor_contact_s');fell=bool(report['fell'])
    if td is None:return dict(distance_m=D,retained_distance_m=None,stability=0.,stable=False,fell=fell,reward=-4.,eligible=False,goal_met=False,phase=phase,objective_version=VERSION)
    times=z['times'];end=min(float(times[-1]),body-.005)if body is not None else float(times[-1]);late=(times>=max(td,end-.35))&(times<=end)
    if not late.any():late=np.abs(times-td)==np.abs(times-td).min()
    speed=float(np.linalg.norm(z['velocity'][late],axis=1).mean());height=float(z['com'][late,2].mean());H=float(abs(z['angular_momentum'][late,0]).mean())
    survival=float(np.clip(((body if body is not None else times[-1])-td)/2.5,0,1))
    upright=float(np.clip((height-.25)/.65,0,1));quiet=float(np.exp(-speed/1.2-H/25.))
    points=z['contact_positions_m'][late];capture=z['com'][late,1]+z['velocity'][late,1]/np.sqrt(9.80665/np.maximum(.3,z['com'][late,2]));lower=points[:,:,1].min(1);upper=points[:,:,1].max(1)
    cap_error=float(np.maximum(0.,np.maximum(lower-capture,capture-upper)).mean());alignment=float(np.exp(-cap_error/.25))
    support=float(np.mean((z['force_world_N'][late,2]>.35*87*9.80665)&(z['force_world_N'][late,2]<2.*87*9.80665)))
    S=float(.30*survival+.25*upright+.20*quiet+.15*alignment+.10*support)
    limits=model.jnt_range[3:];rom=float(np.maximum(z['qpos'][:,3:]-limits[:,1],limits[:,0]-z['qpos'][:,3:]).max())
    last=times>=max(td,float(times[-1])-.50);settled=bool(report['stable_landing']and not fell and body is None and times[-1]>=td+2.45 and np.linalg.norm(z['velocity'][last],axis=1).max()<.25 and abs(z['angular_momentum'][last,0]).max()<5. and rom<.04)
    retained=report.get('nearest_landing_foot_contact_distance_m');retained=float(min(D,retained))if D is not None and retained is not None else D
    loss=max(0.,D-retained)if D is not None and retained is not None and not fell else 0.
    deficit=max(0.,TARGET_FIRST-D)if D is not None else TARGET_FIRST
    eligible=bool(phase['precontact_dynamics_eligible'])
    reward=float(D+2*S+.75*float(settled)-.75*float(fell)-.75*deficit-.75*loss)if eligible and D is not None else -4.
    return dict(distance_m=D,retained_distance_m=retained,stability=S,stable=settled,fell=fell,reward=reward,eligible=eligible,goal_met=bool(settled and eligible and D>=TARGET_FIRST and retained>=TARGET_RETAINED),survival_fraction=survival,upright_score=upright,quiet_score=quiet,capture_alignment_score=alignment,foot_support_score=support,late_COM_speed_m_s=speed,late_COM_height_m=height,late_abs_Hx_kg_m2_s=H,body_floor_contact_s=body,postcontact_duration_s=float(times[-1]-td),full_ROM_excess_rad=rom,distance_loss_m=loss,target_shortfall_m=deficit,phase=phase,objective_version=VERSION,reward_components=dict(first_contact=D,recovery=2*S,stable_bonus=.75*float(settled),fall_penalty=-.75*float(fell),target_shortfall=-.75*deficit,retreat_penalty=-.75*loss),stability_scope='Additive model recovery score, not human success probability; pre-body-contact samples for falls. Fixed design weights and fixed torque capacities.')

def motion_preview(report,z,model):
    names=['pelvis','torso','head','humerus_r','ulna_r','lunate_r','humerus_l','ulna_l','lunate_l','femur_r','tibia_r','calcn_r','toes_r','femur_l','tibia_l','calcn_l','toes_l']
    links=[(0,1),(1,2),(1,3),(3,4),(4,5),(1,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16)]
    ids=[model.body(name).id for name in names];t=z['times'];select=np.unique(np.r_[np.arange(0,len(t),max(1,round(1/30/np.median(np.diff(t))))),len(t)-1])
    return dict(t=t[select].tolist(),xyz=z['body_xpos'][select][:,ids].tolist(),com=z['com'][select].tolist(),velocity=z['velocity'][select].tolist(),grf=(z['force_world_N'][select,2]/(87*9.80665)).tolist(),names=names,links=links,takeoff_s=report.get('takeoff_s'),touchdown_s=report.get('touchdown_s'),body_contact_s=report.get('first_body_floor_contact_s'),origin_y=float(z['contact_positions_m'][0,:,1].min()),scope='Computed segment positions, including actual fall follow-through; no prescribed flight curve.')
