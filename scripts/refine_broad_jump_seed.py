import json,numpy as np
from controlled_broad_jump import ROOT,DEFAULT,build,run
def main():
 source=ROOT/'results/controlled_broad_jump/broad_jump_seed_flight_v2/best/control_summary.json';out=ROOT/'results/controlled_broad_jump/broad_jump_seed_refined_v4';out.mkdir(parents=True,exist_ok=True);base=DEFAULT.copy();base.update(json.loads(source.read_text())['parameters']);prepared=build(.0005);cache={};rows=[]
 variants=[{},dict(prepare_landing_s=base['prepare_landing_s']-.015),dict(prepare_landing_s=base['prepare_landing_s']+.015),dict(prepare_trunk=base['prepare_trunk']-.05),dict(prepare_trunk=base['prepare_trunk']+.05),dict(prepare_knee=base['prepare_knee']+.08),dict(prepare_knee=base['prepare_knee']-.08),dict(prepare_hip=base['prepare_hip']+.08),dict(prepare_hip=base['prepare_hip']-.08),dict(prepare_shoulder=base['prepare_shoulder']-.12),dict(prepare_shoulder=base['prepare_shoulder']+.12)]
 for i,change in enumerate(variants):
  p=base.copy();p.update(change);rep,z=run(p,out/f'candidate_{i:02d}',dt=.0005,prepared=prepared,flight_cache=cache);rows.append(dict(candidate=i,status=rep['status'],distance_m=rep.get('valid_distance_m'),parameters_changed=change));(out/'cases.json').write_text(json.dumps(rows,indent=2));print('REFINED_BROAD',i,rep['status'],rep.get('valid_distance_m'),flush=True)
  if rep['stable_landing']:(out/'selected.json').write_text(json.dumps(dict(source=str(out/f'candidate_{i:02d}'),report=rep),indent=2));print('BROAD_STARTER_READY',out/f'candidate_{i:02d}',flush=True);return
 print('BROAD_STARTER_NEEDS_MORE_TUNING',flush=True)
if __name__=='__main__':main()
