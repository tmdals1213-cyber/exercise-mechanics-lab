import json,numpy as np
from controlled_broad_jump import ROOT,DEFAULT,build,run

def main():
 out=ROOT/'results/controlled_broad_jump/effort_diagnosis_v5';out.mkdir(parents=True,exist_ok=True);source=ROOT/'results/controlled_broad_jump/ground_control_diagnosis_v2/candidate_02/control_summary.json';base=DEFAULT.copy();base.update(json.loads(source.read_text())['parameters']);base.update(ankle_bottom=.50,trunk_bottom=-.35,pitch_gain=500,hip_bottom=1.25,knee_bottom=1.3,push_s=.28);prepared=build(.001);rows=[]
 variants=[dict(push_torque_boost_Nm=dict(hip_flexion=h,knee_angle=k,ankle_angle=a),trunk_bottom=t)for h,k in [(0,0),(-100,-100),(-200,-200),(-300,-200)]for a,t in [(-100,-.35),(-250,-.35),(-100,-.5),(-250,-.5)]]
 for i,change in enumerate(variants):
  p=base.copy();p.update(change);rep,z=run(p,out/f'candidate_{i:02d}',dt=.001,prepared=prepared);vel=rep.get('takeoff_velocity_m_s',[0,0,0]);H=rep.get('takeoff_H_kg_m2_s',[100])[0];vx=-vel[1];vz=vel[2];cost=(vx-2.9)**2+((vz-2)/.8)**2+.002*H**2+(30 if rep['takeoff_s']is None or rep['fell']else 0);row=dict(candidate=i,cost=cost,vx=vx,vz=vz,Hx=H,peak_BW=rep['peak_GRF_BW'],COM=rep.get('takeoff_COM_m'),parameters=change);rows.append(row);(out/'search.json').write_text(json.dumps(rows,indent=2));print('EFFORT',i,round(cost,3),'v',round(vx,2),round(vz,2),'H',round(H,1),'F',round(rep['peak_GRF_BW'],1),flush=True)
 (out/'summary.json').write_text(json.dumps(sorted(rows,key=lambda r:r['cost']),indent=2))
if __name__=='__main__':main()
