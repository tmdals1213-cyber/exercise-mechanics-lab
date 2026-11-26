import json,numpy as np
from controlled_broad_jump import ROOT,DEFAULT,build,run

def main():
 out=ROOT/'results/controlled_broad_jump/push_response_diagnosis_v3';out.mkdir(parents=True,exist_ok=True);base=DEFAULT.copy();base.update(json.loads((ROOT/'results/controlled_broad_jump/broad_jump_seed_refined_v4/candidate_01/control_summary.json').read_text())['parameters']);base.update(seconds=2.7,stop_after_takeoff_s=.045,propulsion_force_feedback=True);prepared=build(.001);rows=[]
 for i,(ax,az,push,lean)in enumerate([(6,6,.32,.32),(9,8,.32,.40),(12,8,.32,.50),(10,10,.36,.50),(12,10,.40,.55),(14,12,.40,.55),(10,8,.40,.60),(12,8,.45,.60)]):
  p=base.copy();p.update(push_forward_accel_m_s2=ax,push_up_accel_m_s2=az,push_s=push,lean_m=lean);rep,z=run(p,out/f'candidate_{i:02d}',dt=.001,prepared=prepared);vel=rep.get('takeoff_velocity_m_s',[0,0,0]);H=rep.get('takeoff_H_kg_m2_s',[100])[0];vx=-vel[1];vz=vel[2];cost=(vx-2.9)**2+((vz-2)/.8)**2+.003*H**2+(30 if rep['takeoff_s']is None or rep['fell']else 0);row=dict(candidate=i,cost=cost,vx=vx,vz=vz,Hx=H,peak_BW=rep['peak_GRF_BW'],COM=rep.get('takeoff_COM_m'),parameters=dict(ax=ax,az=az,push=push,lean=lean));rows.append(row);(out/'search.json').write_text(json.dumps(rows,indent=2));print('PUSH_RESPONSE',i,round(cost,3),'v',round(vx,2),round(vz,2),'H',round(H,1),'F',round(rep['peak_GRF_BW'],1),flush=True)
 (out/'summary.json').write_text(json.dumps(sorted(rows,key=lambda r:r['cost']),indent=2))
if __name__=='__main__':main()
