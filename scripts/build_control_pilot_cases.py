from pathlib import Path
import json,numpy as np
from controlled_broad_jump import ROOT,DEFAULT,build,run
def main():
 out=ROOT/'results/controlled_broad_jump/controller_pilot_v1';out.mkdir(parents=True,exist_ok=True);p=DEFAULT.copy();p.update(json.loads((ROOT/'results/controlled_broad_jump/refined_control_v9/control_summary.json').read_text())['parameters']);p.update(seconds=4.5,joint_armature_kg_m2=0.);prepared=build(.0005);cache={};rows=[]
 for name,offset in [('baseline',0.),('earlier_15ms',-.015),('later_15ms',.015),('earlier_30ms',-.030),('later_30ms',.030)]:
  case=p.copy();case['prepare_landing_s']+=offset;report,z=run(case,out/name,dt=.0005,prepared=prepared,flight_cache=cache);rows.append(dict(case=name,offset_s=offset,status=report['status'],valid_distance_m=report.get('valid_distance_m'),first_contact_heel_distance_m=report.get('first_contact_heel_distance_m'),takeoff_velocity_m_s=report.get('takeoff_velocity_m_s'),compute_seconds=report['compute_seconds']));(out/'cases.json').write_text(json.dumps(rows,indent=2));print('CONTROL_CASE',name,report['status'],report.get('valid_distance_m'),flush=True)
 print('CONTROL_PILOT_CASES_READY',out,flush=True)
if __name__=='__main__':main()
