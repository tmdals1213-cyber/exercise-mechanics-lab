"""Prepare the fixed first example, identical inputs except 65 vs 80 cm grip."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'scenarios/ohp_fullbody.json').read_text('utf-8'))
for grip in [65,80]:
 case=json.loads(json.dumps(config));case['grip_width_m']=grip/100
 (ROOT/f'scenarios/ohp_grip_{grip}.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
 (ROOT/f'results/grip_comparison/grip_{grip}').mkdir(parents=True,exist_ok=True)
print('Prepared matched 65 and 80 cm scenarios')
