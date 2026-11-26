import json,numpy as np,mujoco
from controlled_broad_jump import ROOT,DEFAULT,build,run

def main():
 out=ROOT/'results/controlled_broad_jump/ground_control_diagnosis_v2';out.mkdir(parents=True,exist_ok=True);base=DEFAULT.copy();base.update(json.loads((ROOT/'results/controlled_broad_jump/broad_jump_seed_refined_v4/candidate_01/control_summary.json').read_text())['parameters']);base.update(seconds=2.5,stop_after_takeoff_s=.045);prepared=build(.001);rows=[]
 variants=[]
 for ff in [0.,.6,1.]:
  for posture in [dict(),dict(hip_bottom=1.4,knee_bottom=1.45,ankle_bottom=.85),dict(hip_bottom=1.25,knee_bottom=1.3,ankle_bottom=.85),dict(hip_bottom=1.4,knee_bottom=1.6,ankle_bottom=.95)]:
   variants.append(dict(target_velocity_feedforward=ff,lean_m=.55,pitch_bottom=.9,pitch_top=1.,hip_top=.05,push_s=.30,ankle_delay=.55,arm_delay_s=-.08,leg_kp=850.,ankle_kp=400.,**posture))
 for i,vary in enumerate(variants):
  p=base.copy();p.update(vary);rep,z=run(p,out/f'candidate_{i:02d}',dt=.001,prepared=prepared);vel=rep.get('takeoff_velocity_m_s',[0,0,0]);H=rep.get('takeoff_H_kg_m2_s',[100])[0];vx=-vel[1];vz=vel[2];cost=(vx-2.9)**2+((vz-2)/.8)**2+.003*H**2+(30 if rep['takeoff_s']is None or rep['fell']else 0);row=dict(candidate=i,cost=cost,vx=vx,vz=vz,Hx=H,peak_BW=rep['peak_GRF_BW'],COM=rep.get('takeoff_COM_m'),parameters=vary);rows.append(row);(out/'search.json').write_text(json.dumps(rows,indent=2));print('GROUND_DIAG',i,round(cost,3),'v',round(vx,2),round(vz,2),'H',round(H,1),'F',round(rep['peak_GRF_BW'],1),flush=True)
 (out/'summary.json').write_text(json.dumps(sorted(rows,key=lambda r:r['cost']),indent=2))
if __name__=='__main__':main()
