"""A small landing fit with the chosen propulsion and body held fixed."""
import json,numpy as np
from controlled_broad_jump import ROOT,DEFAULT,build,run

def main():
 out=ROOT/'results/controlled_broad_jump/broad_jump_base_completion_v8';out.mkdir(parents=True,exist_ok=True);source=ROOT/'results/controlled_broad_jump/lean_timing_diagnosis_v6/candidate_07/control_summary.json';base=DEFAULT.copy();base.update(json.loads(source.read_text())['parameters']);base.pop('stop_after_takeoff_s',None);base.update(seconds=5.5,flight_transition_s=.20,prepare_transition_s=.16);prepared=build(.0005);cache={};rows=[]
 variants=[dict(prepare_hip=h,prepare_knee=k,prepare_landing_s=t,prepare_trunk=-.1,prepare_shoulder=.9)for h,k,t in [(1.3,.3,.12),(1.5,.3,.12),(1.5,.5,.12),(1.3,.3,.20),(1.5,.3,.20),(1.5,.5,.20)]]
 for i,change in enumerate(variants):
  p=base.copy();p.update(change);rep,z=run(p,out/f'candidate_{i:02d}',dt=.0005,prepared=prepared,flight_cache=cache);row=dict(candidate=i,status=rep['status'],stable=rep['stable_landing'],distance=rep.get('valid_contact_distance_m'),first_distance=rep.get('first_contact_heel_distance_m'),capture=rep.get('landing_capture_error_m'),tilt=rep.get('landing_torso_pitch_rad'),takeoff=rep.get('takeoff_velocity_m_s'),parameters=change);rows.append(row);(out/'cases.json').write_text(json.dumps(rows,indent=2));print('BASE_COMPLETION',i,rep['status'],'d',rep.get('first_contact_heel_distance_m'),'capture',rep.get('landing_capture_error_m'),'tilt',rep.get('landing_torso_pitch_rad'),flush=True)
  if rep['stable_landing'] and rep.get('valid_contact_distance_m',0)>1.7:
   (out/'selected.json').write_text(json.dumps(dict(source=str(out/f'candidate_{i:02d}'),report=rep),indent=2));print('NORMAL_BASE_READY',out/f'candidate_{i:02d}',flush=True);return
 (out/'status.json').write_text(json.dumps(dict(status='landing_needs_work',scope='Propulsion fixed for all six trials; only landing posture/timing fitted.'),indent=2))
if __name__=='__main__':main()
