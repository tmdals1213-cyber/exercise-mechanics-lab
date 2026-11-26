import json,numpy as np
from controlled_broad_jump import ROOT,DEFAULT,build,run

def main():
 out=ROOT/'results/controlled_broad_jump/balance_release_diagnosis_v4';out.mkdir(parents=True,exist_ok=True);source=ROOT/'results/controlled_broad_jump/ground_control_diagnosis_v2/candidate_02/control_summary.json';base=DEFAULT.copy();base.update(json.loads(source.read_text())['parameters']);prepared=build(.001);print('ANKLE_NATIVE_RANGE',[(prepared[1].joint(j).name,prepared[1].jnt_range[j].tolist(),bool(prepared[1].jnt_limited[j]))for j in range(prepared[1].njnt)if'ankle'in prepared[1].joint(j).name],flush=True);rows=[]
 variants=[dict(push_balance_velocity_gain=v,push_lean_extra_m=e,pitch_gain=g)for v in [100.,0.,-100.]for e in [0.,.3]for g in [250.,500.]]
 for i,change in enumerate(variants):
  p=base.copy();p.update(change);rep,z=run(p,out/f'candidate_{i:02d}',dt=.001,prepared=prepared);vel=rep.get('takeoff_velocity_m_s',[0,0,0]);H=rep.get('takeoff_H_kg_m2_s',[100])[0];vx=-vel[1];vz=vel[2];cost=(vx-2.9)**2+((vz-2)/.8)**2+.003*H**2+(30 if rep['takeoff_s']is None or rep['fell']else 0);row=dict(candidate=i,cost=cost,vx=vx,vz=vz,Hx=H,peak_BW=rep['peak_GRF_BW'],COM=rep.get('takeoff_COM_m'),parameters=change);rows.append(row);(out/'search.json').write_text(json.dumps(rows,indent=2));print('BALANCE_RELEASE',i,round(cost,3),'v',round(vx,2),round(vz,2),'H',round(H,1),'F',round(rep['peak_GRF_BW'],1),flush=True)
 (out/'summary.json').write_text(json.dumps(sorted(rows,key=lambda r:r['cost']),indent=2))
if __name__=='__main__':main()
