"""Compare like-named groups, retaining protocol mismatches and source limits."""
from pathlib import Path
import html,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'


def main():
    spec=json.loads((ROOT/'scenarios/muscle_force_literature.json').read_text())
    reference=spec['references'][0];cases={};provenance={}
    for kg in [60,100,140]:
        key=f'squat_{kg}';source=OUT/key/'analysis';z=np.load(source/'tensions.npz');s=json.loads((source/'tension_summary.json').read_text());F=z['force_N'];names=s['names'][:s['model_muscle_elements']]
        values={}
        for row in reference['quantities']:
            ids=[i for i,n in enumerate(names) if n.startswith(tuple(row['prefixes']))]
            assert ids
            per_side=F[:,ids].sum(1)/2
            right=[i for i in ids if not names[i].endswith('_l')]
            left=[i for i in ids if names[i].endswith('_l')]
            values[row['key']]=dict(mean_per_side_peak_N=float(per_side.max()),
                right_peak_N=float(F[:,right].sum(1).max()),left_peak_N=float(F[:,left].sum(1).max()),
                model_peak_divided_by_reference_mean=float(per_side.max()/row['mean_peak_N']),
                ratio_is_prediction_error=False,matched_protocol=False)
        cases[key]=values;provenance[key]=hashlib.sha256((source/'tensions.npz').read_bytes()).hexdigest()
    source=OUT/'cmj_analysis';z=np.load(source/'tensions.npz');s=json.loads((source/'tension_summary.json').read_text());ids=[i for i,n in enumerate(s['names']) if n.startswith(('gas','soleus'))]
    calf=float((z['force_N'][:,ids].sum(1)/2).max())
    report=dict(status='preliminary_literature_screening_not_validation',review=spec,cases=cases,
        force_file_sha256=provenance,CMJ_calf_scalar_sum_peak_per_side_N=calf,
        CMJ_force_file_sha256=hashlib.sha256((source/'tensions.npz').read_bytes()).hexdigest(),
        CMJ_calf_is_not_identical_to_reference_Achilles_tendon_force=True,
        changes_to_force_solution=False,reference_used_as_force_cap=False)
    (OUT/'muscle_reference_review.json').write_text(json.dumps(report,indent=2))
    rows=''
    for row in reference['quantities']:
        vals=[cases[f'squat_{kg}'][row['key']]['mean_per_side_peak_N']/1000 for kg in [60,100,140]]
        rows+=f"<tr><td>{html.escape(row['label'])}</td><td>{row['mean_peak_N']/1000:.2f} ± {row['SD_N']/1000:.2f}</td>"+''.join(f'<td>{v:.2f}</td>' for v in vals)+'</tr>'
    refs=''.join(f'<li><a href="{html.escape(r["url"],quote=True)}">{html.escape(r["title"])}</a><p>{html.escape(r.get("notes",r.get("protocol","")))}</p></li>' for r in spec['references'])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Muscle force reference review</title><style>body{background:#0d1726;color:#edf3fb;font:17px/1.6 system-ui;margin:24px auto;padding:16px;max-width:1050px}a{color:#8ce2d5}td,th{padding:12px;border-bottom:1px solid #34495e;text-align:left;white-space:nowrap}p,li{color:#b6c9dd}.scroll{overflow:auto}.note{border-left:3px solid #efb45b;padding:12px}img{width:100%}</style><h1>Muscle force: magnitude screening</h1><p class="note">Literature values are model estimates from different participants, loads and motions. This is an order-of-magnitude review, not a matched validation or a reason to clip forces.</p><p>All values below are peak kN per side. The reference reports the right side; our values average left and right. Vasti excludes rectus femoris; gluteus maximus excludes medius and minimus. Separate muscle peaks must not be added as though simultaneous.</p><div class="scroll"><table><tr><th>Muscle / group</th><th>Goodman 2023<br>Mean ± SD</th><th>Model 60 kg</th><th>Model 100 kg</th><th>Model 140 kg</th></tr>'''+rows+'''</table></div><p>Reference: 13 trained women, parallel squat at 85% 1RM; static optimization, Table 1. Our prescribed 180 cm / 88 kg subject and cadences are different. Several large model muscle forces are multiple times the study means. Their shared kN scale does not establish agreement.</p><h2>CMJ calf check</h2>'''+f'<p>Model gastrocnemius + soleus scalar-sum peak: {calf/1000:.2f} kN per side. Wade 2018 estimated right Achilles peak force of 3.66–3.90 kN across unweighted height conditions, and 4.50 kN with total mass at 160% baseline. These quantities are related but not identical.</p>'+'''<p>Absolute CMJ quadriceps/hamstring/gluteal magnitude and upper-body muscle-force benchmarks remain incomplete. The selected antagonist distribution and anatomical moment arms require further review.</p><h2>Joint reaction direction</h2><p>Compression/distraction below refers to a rotating bone long axis. Red is axial compression, blue is axial distraction, green is shear. These are conditional ideal-joint reactions, not cartilage contact pressure or individual ligament forces. The portrait videos still colour total reaction magnitude.</p><img src="squat_100_joint_components.png" alt="Axial compression, axial distraction and shear for the 100 kg squat"><p><a href="squat_100/joint_components_audit.json">Axis definitions and decomposition checks</a> · <a href="squat_100/joint_components.csv">100 kg component time series</a></p><h2>References and scope</h2><ol>'''+refs+'''</ol><p><a href="muscle_reference_review.json">Definitions, exact numbers and force-file hashes</a> · <a href="comparison.html">Comparison viewer</a></p></html>'''
    (OUT/'muscle_reference_review.html').write_text(page,encoding='utf-8')
    print('REFERENCE_REVIEW_READY','CMJ calf',round(calf,2),'N',flush=True)
    print(json.dumps(cases['squat_100'],indent=2))


if __name__=='__main__':main()
