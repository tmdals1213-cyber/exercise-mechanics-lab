"""Count actual V3 PPO rollouts, not worker episode IDs or validation replays."""
from pathlib import Path
import json
from jump_monitor_data import read

def collect(manifest):
    run=Path(manifest['run']);rows=[json.loads(line)for line in(run/'history.jsonl').read_text(encoding='utf-8').splitlines()if line.strip()];training=[r for r in rows if r['kind']=='ppo_train'];byid={r['id']:i for i,r in enumerate(training,1)};validations=[]
    for path in run.glob('validation_*.json'):
        report=read(path)
        if report.get('passed'):validations.append((path,report))
    frozen=Path(manifest.get('qualified_final_validation',''))
    if frozen.is_file():
        report=read(frozen)
        if report.get('passed'):validations.insert(0,(frozen,report))
    result={}
    for key in manifest['final_four_keys']:
        case=next(c for c in manifest['cases']if c['key']==key);record=case['record'];source=str(Path(case['source'])).replace('\\','/').lower();match=next(((path,report)for path,report in validations if any(str(Path(c['folder'])).replace('\\','/').lower()==source for c in report['cases'])),None)
        origin=match[1]['cases'][0]if match else record;trial=byid.get(origin['id'],sum(r['finished_at']<=origin['finished_at']for r in training));supplemental=match[1].get('discovery_provenance')if match else None
        label=f'LOCAL TRIAL {supplemental["candidate_number"]:,}'if supplemental else f'PPO TRIAL {trial:,}'if trial else'BASELINE'
        result[key]=dict(discovered_at_ppo_trial=trial,discovery_kind=origin['kind'],discovery_id=origin['id'],display_label=label,supplemental_search=supplemental,validation_report=str(match[0])if match else None,render_source_id=record['id'],checked_distance_m=case['checked_distance_m'])
    return dict(cases=result,completed_v3_PPO_trials=len(training),count_definition='Completed PPO rollouts in landing_stability_v3, including failures. Worker-local episode numbers, fine-step verification, CEM and recenter replays are excluded.',inherited_prior_protocol_training_episodes=read(run/'definition.json').get('inherited_training_episodes',0))
