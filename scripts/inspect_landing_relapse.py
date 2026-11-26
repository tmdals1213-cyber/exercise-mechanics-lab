"""Find a real failed trial that nearly stood before accelerating again."""
import json
import numpy as np
from jump_monitor_data import ROOT,read,atomic

def main():
    root=ROOT/'results/controlled_broad_jump/landing_stability_v2';ranked=[]
    for file in root.glob('worker_*/episode_*/score.json'):
        row=read(file);td=row.get('phase',{}).get('first_foot_contact_s');body=row.get('body_floor_contact_s')
        if not row.get('fell')or td is None or body is None or body<td+1.:continue
        z=np.load(file.parent/'control_motion.npz');t=z['times'];speed=np.linalg.norm(z['velocity'],axis=1);ids=np.flatnonzero((t>td+.35)&(t<body-.3)&(z['com'][:,2]>.80)&(speed<.25))
        if not len(ids):continue
        # Prefer a sustained near-quiet state, not a single turning point.
        groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1);group=max(groups,key=len);duration=float(t[group[-1]]-t[group[0]])
        if duration<.06:continue
        quiet=int(group[np.argmin(speed[group])]);later=np.flatnonzero((t>t[quiet])&(t<body));j=int(later[np.argmax(speed[later])])
        ranked.append(dict(folder=str(file.parent),id=row['id'],kind=row['kind'],dt_s=row['dt_s'],quiet_time_s=float(t[quiet]),quiet_after_td_s=float(t[quiet]-td),quiet_duration_s=duration,quiet_speed_m_s=float(speed[quiet]),quiet_COM_height_m=float(z['com'][quiet,2]),speed_before_body_contact_m_s=float(speed[j]),body_contact_s=body,first_contact_m=row['distance_m'],quiet_grf_BW=float(z['force_world_N'][quiet,2]/(87*9.80665))))
    ranked.sort(key=lambda q:(q['quiet_duration_s'],-q['quiet_speed_m_s']),reverse=True);assert ranked,'No near-upright relapse case found.'
    out=ROOT/'results/landing_controller_audit_v1';out.mkdir(exist_ok=True);atomic(out/'relapse_cases.json',ranked);atomic(out/'selected_case.json',ranked[0]);print(json.dumps(ranked[:3],indent=2),flush=True)

if __name__=='__main__':main()
