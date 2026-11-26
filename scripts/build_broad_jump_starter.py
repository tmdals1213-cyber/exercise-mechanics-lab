"""Small deterministic posture/timing search for a recognisable broad-jump seed."""
from pathlib import Path
import json,numpy as np,mujoco
from controlled_broad_jump import ROOT,DEFAULT,build,run

def main():
 out=ROOT/'results/controlled_broad_jump/standing_long_jump_seed_v1';out.mkdir(parents=True,exist_ok=True);base=DEFAULT.copy();base.update(json.loads((ROOT/'scenarios/controlled_jump_pilot.json').read_text())['parameters']);base.update(ready_s=.65,crouch_s=.65,ready_shoulder=.85,shoulder_bottom=-.5,shoulder_top=1.65,knee_delay=.04,hip_delay=0.,hip_bottom=1.65,knee_bottom=1.45,ankle_bottom=.68,trunk_bottom=-.12,hip_top=.28,pitch_bottom=.65,pitch_top=.65,lean_m=.32,push_s=.32,ankle_delay=.62,arm_delay_s=-.04,flight_hip=.95,flight_knee=1.2,flight_shoulder=.2,prepare_hip=1.35,prepare_knee=.40,prepare_shoulder=.5,prepare_trunk=.05,prepare_landing_s=.20,seconds=4.8,joint_armature_kg_m2=0.);prepared=build(.001);rows=[]
 variants=[{},dict(lean_m=.38,pitch_top=.75,hip_top=.42),dict(lean_m=.42,pitch_top=.75,hip_top=.50),dict(lean_m=.36,pitch_top=.70,hip_top=.35,push_s=.36),dict(lean_m=.40,pitch_top=.85,hip_top=.50,ankle_delay=.45),dict(lean_m=.32,pitch_top=.65,hip_top=.35,arm_delay_s=-.08),dict(lean_m=.38,pitch_top=.75,hip_top=.45,hip_bottom=1.75,knee_bottom=1.55,ankle_bottom=.76),dict(lean_m=.42,pitch_top=.85,hip_top=.50,hip_bottom=1.8,knee_bottom=1.55,ankle_bottom=.80)]
 for i,vary in enumerate(variants):
  p=base.copy();p.update(vary);label=f'candidate_{i:02d}';rep,z=run(p,out/label,dt=.001,prepared=prepared);vel=rep.get('takeoff_velocity_m_s',[0,0,0]);H=rep.get('takeoff_H_kg_m2_s',[100,0,0])[0];vx=-vel[1];vz=vel[2];pitch=rep.get('takeoff_root_pitch_rad',2);score=3*max(1.25-vx,0)+2*max(1.6-vz,0)+.06*abs(H)+max(abs(pitch-.65)-.4,0)-(1 if rep['stable_landing']else 0);rep.update(search_score=score,candidate=i);rows.append(rep);(out/'candidates.json').write_text(json.dumps(rows,indent=2));print('BROAD_SEED',i,rep['status'],'v',np.round(vel,2),'angle',round(float(np.rad2deg(np.arctan2(vz,max(vx,.001)))),1),'H',round(H,1),'d',rep.get('first_contact_heel_distance_m'),'score',round(score,2),flush=True)
 best=min(rows,key=lambda r:r['search_score']);(out/'selected_candidate.json').write_text(json.dumps(best,indent=2));print('BROAD_SEED_SELECTED',best['candidate'],flush=True)
if __name__=='__main__':main()
