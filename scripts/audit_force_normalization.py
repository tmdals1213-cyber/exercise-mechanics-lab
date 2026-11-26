"""Expose absolute loads and two distinct normalizations without changing forces.

F/Fref uses imported actuator force parameters. It is NOT activation, measured
strength utilization, or a force-length-velocity adjusted capacity estimate.
Group ratios are reference-force weighted means, not the largest element ratio.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from case_contours import ROWS, overview

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/power_comparison'


def analyze(key):
    source = OUT / ('cmj_analysis' if key == 'cmj' else key + '/analysis')
    z = np.load(source / 'tensions.npz')
    data = json.loads((source / 'visualization/data.json').read_text())
    summary = json.loads((source / 'tension_summary.json').read_text())
    t, F, ref = z['times'], z['force_N'], z['reference_N']
    mass = float(data['mass_kg'])
    bw = mass * 9.81
    values, normalized, rows = overview(data), [], []
    assert np.isfinite(ref).all() and np.all(ref > 0)
    for j, (label, keys) in enumerate(ROWS):
        ids = [i for g in data['groups'] if g['key'] in keys for i in g['indices']]
        assert len(ids) == len(set(ids)) and ids
        reference_per_side = ref[ids].sum() / 2
        ratio = F[:, ids].sum(1) / ref[ids].sum()
        active = z['active_force_N'][:, ids].sum(1) / ref[ids].sum()
        passive = z['passive_force_N'][:, ids].sum(1) / ref[ids].sum()
        np.testing.assert_allclose(ratio, active + passive, atol=1e-10)
        np.testing.assert_allclose(values[j], F[:, ids].sum(1) / 2, atol=1e-8)
        k = int(np.argmax(ratio))
        element_ratio = F[:, ids] / ref[ids]
        kt, ki = np.unravel_index(element_ratio.argmax(), element_ratio.shape)
        normalized.append(ratio)
        rows.append(dict(group=label, reference_force_per_side_N=float(reference_per_side),
            peak_group_tension_per_side_N=float(values[j].max()),
            peak_group_tension_per_bodyweight=float(values[j].max() / bw),
            peak_group_reference_ratio=float(ratio[k]),
            active_reference_ratio_at_group_peak=float(active[k]),
            passive_reference_ratio_at_group_peak=float(passive[k]),
            peak_time_s=float(t[k]),
            peak_element_reference_ratio=float(element_ratio[kt, ki]),
            peak_element=summary['names'][ids[ki]], peak_element_time_s=float(t[kt])))
    audit_path = (OUT / 'cmj_analysis' if key == 'cmj' else OUT / key) / 'tension_audit.json'
    audit = json.loads(audit_path.read_text())
    return dict(case=key, body_mass_kg=mass, bodyweight_N=bw,
                force_file_sha256=hashlib.sha256((source/'tensions.npz').read_bytes()).hexdigest(),
                force_selection=summary.get('selection_mode'),
                numerical_checks_status=audit['status'], rows=rows,
                quadriceps_moment_audit=audit['quadriceps_moment_audit']), (t, values, np.array(normalized))


def main():
    records, curves = [], []
    keys = ['cmj', 'squat_60', 'squat_100', 'squat_140']
    for key in keys:
        record, curve = analyze(key)
        records.append(record)
        curves.append(curve)
    report = dict(status='diagnostic_only', force_values_changed=False,
        physiological_capacity_validated=False, upper_strength_bound_enforced=False,
        definitions=dict(bodyweight_ratio='F/(body mass * 9.81); excludes bar mass from denominator.',
            reference_ratio='Sum of group tensions / sum of imported group reference-force parameters.',
            element_ratio='Element tension / its imported reference-force parameter.',
            group_scope='Bilateral sum / 2; not a vector resultant or a single tendon force.',
            active_passive='Both retained and reported separately; ratios are unbounded.'),
        warnings=['Reference-force parameters are generic model inputs, not this subject\'s measured maximum.',
            'No active force-length-velocity capacity constraint is enforced in these tension solutions.',
            'A group average can hide a large individual element demand; inspect both.',
            'Normalization changes interpretation, not physical validity or force balance.',
            'Do not label reference ratio as activation, percent MVC, effort, efficiency or injury risk.',
            'CMJ retargeting retains its disclosed external-force/motion residual.'],
        references=[
            'https://pubmed.ncbi.nlm.nih.gov/23348130/',
            'https://pubmed.ncbi.nlm.nih.gov/11323225/',
            'https://github.com/opensim-org/opensim-core/blob/main/OpenSim/Simulation/Model/Muscle.h',
            'https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085189/Working+with+Static+Optimization'],
        cases=records)
    (OUT/'force_normalization_audit.json').write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout='constrained', sharey='row')
    upper_bw = max(float(v.max()/r['bodyweight_N']) for r, (_, v, _) in zip(records, curves))*1.08
    upper_ref = max(float(v.max()) for _, _, v in curves)*1.08
    for col, (key, record, (t, force, ratios)) in enumerate(zip(keys, records, curves)):
        for j, (label, _) in enumerate(ROWS):
            axes[0, col].plot(t, force[j]/record['bodyweight_N'], label=label)
            axes[1, col].plot(t, ratios[j], label=label)
        axes[0, col].set_title(key.replace('_', ' ').upper())
        axes[1, col].axhline(1, color='black', ls='--', lw=.8)
        for row_index, ax in enumerate(axes[:, col]):
            ax.set_xlabel('Physical time [s]')
            ax.grid(alpha=.2)
            ax.set_ylim(0, upper_bw if row_index == 0 else upper_ref)
    axes[0, 0].set_ylabel('Group tension / bodyweight\nMean per side [dimensionless]')
    axes[1, 0].set_ylabel('Group tension / reference force\n[dimensionless; NOT activation]')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, fontsize=9)
    fig.suptitle('Same conditional forces, different denominators\n'
                 'No strength cap; imported force references are not measured capacity', fontsize=15)
    fig.savefig(OUT/'force_normalization_review.png', dpi=150)
    plt.close(fig)
    print(json.dumps({r['case']: next(x for x in r['rows'] if x['group']=='Quadriceps') for r in records}, indent=2))


if __name__ == '__main__':
    main()
