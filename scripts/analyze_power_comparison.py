"""Audit a measured CMJ power baseline before comparing box jumps and squats.

Primary metric is vertical centre-of-mass power, Fz * v_COM,z. It is a
pseudo-work rate for COM translation, not contact-point work or muscle power.
No force amplitudes, masses, trajectories or time axes are fitted here.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/measured_cmj'
OUT = ROOT / 'results/power_comparison'
G = 9.80665


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integral(t, y):
    return float(np.trapezoid(y, t))


def above_duration(t, y, level):
    """Piecewise-linear threshold duration, including partial sample intervals."""
    total = 0.
    for a, b, u, v in zip(t[:-1], t[1:], y[:-1], y[1:]):
        if min(u, v) >= level:
            total += b - a
        elif max(u, v) > level:
            total += (b - a) * (max(u, v) - level) / abs(v - u)
    return float(total)


def analytic_checks():
    t = np.linspace(0., 1., 1001)
    m, speed, acceleration = 88., .5, 1.7
    F = np.full_like(t, m * G)
    assert integral(t, F * 0) == 0
    assert abs(integral(t, F * speed) - m * G * speed) < 1e-9
    v = acceleration * t
    y = .5 * acceleration * t**2
    E = .5 * m * v**2 + m * G * y
    assert abs(integral(t, m * (G + acceleration) * v) - (E[-1] - E[0])) < 1e-9
    v = 3. - G * t
    y = 1. + 3. * t - .5 * G * t**2
    assert np.ptp(.5 * m * v**2 + m * G * y) < 1e-9
    # Peaks of force and speed need not coincide.
    f, v = np.array([2., 1.]), np.array([1., 2.])
    assert max(f*v) == 2 and max(f)*max(v) == 4
    # A stationary bar does not erase body COM movement, and vice versa.
    body_mass, bar_mass, body_v, bar_v = 88., 100., .4, .6
    system_v = (body_mass*body_v + bar_mass*bar_v)/(body_mass+bar_mass)
    assert abs((body_mass+bar_mass)*G*system_v - G*(body_mass*body_v+bar_mass*bar_v)) < 1e-9
    assert abs(above_duration(np.array([0., 1., 2.]), np.array([0., 2., 0.]), 1.) - 1.) < 1e-12
    return {'status': 'passed', 'analytic_cases': 7,
            'cases': ['static', 'constant velocity', 'accelerating mass energy',
                      'ballistic energy', 'noncoincident peaks',
                      'body plus bar COM definition', 'threshold interval interpolation']}


def analyze_trial(reference):
    trial = reference['trial']
    mechanics_path = SOURCE / trial / 'mechanics_replay.npz'
    plates_path = SOURCE / f'{trial}_measured.npz'
    mech, plates = np.load(mechanics_path), np.load(plates_path)
    t = mech['times']
    assert np.max(abs(t - plates['times'])) < 1e-10
    assert np.all(np.diff(t) > 0)
    # OpenSim world Y is vertical. Sum both plates before taking a product.
    force = plates['applied_ground_force_N'].sum(axis=1)[:, 1]
    y, v, a = (mech[key][:, 1] for key in
               ['COM_m', 'COM_velocity_m_s', 'COM_acceleration_m_s2'])
    mass = float(mech['body_mass_kg'].sum())
    assert abs(mass - reference['mass_kg']) < 1e-8
    onset, takeoff = reference['movement_onset_s'], reference['takeoff_s']
    candidates = np.flatnonzero((t >= onset) & (t < takeoff))
    bottom_index = int(candidates[np.argmin(y[candidates])])
    bottom = float(t[bottom_index])
    # Preserve the source's 250 Hz bottom event. Include exact takeoff endpoint.
    phase_t = np.r_[t[(t >= bottom) & (t < takeoff)], takeoff]
    at = lambda z: np.interp(phase_t, t, z)
    phase_F, phase_v, phase_y, phase_a = map(at, (force, v, y, a))
    P = force * v
    phase_P = phase_F * phase_v
    i = int(np.argmax(phase_P))
    duration = float(takeoff - bottom)
    work = integral(phase_t, phase_P)
    energy_change = float(mass*G*(phase_y[-1]-phase_y[0])
                          + .5*mass*(phase_v[-1]**2-phase_v[0]**2))
    kinematic_P = mass*(phase_a+G)*phase_v
    residual = at(mech['root_force_residual_N'][:, 1])
    # This identity is a bookkeeping check, not an independent human validation.
    identity_error = float(np.max(abs((kinematic_P-phase_P)-residual*phase_v)))
    assert identity_error < 1e-7
    # Alternative velocity from force, anchored at the same bottom event with
    # zero vertical speed. Shared force plates and event; not independent data.
    force_velocity = cumulative_trapezoid(phase_F/mass-G, phase_t, initial=0.)
    force_power = phase_F * force_velocity
    force_peak = float(force_power.max())
    peak = float(phase_P[i])
    row = dict(
        trial=trial, mass_kg=mass, source_bottom_s=bottom,
        source_takeoff_s=takeoff, source_movement_onset_s=onset,
        bottom_event_resolution_s=float(np.median(np.diff(t))),
        propulsion_duration_s=duration, peak_COM_vertical_power_W=peak,
        mean_propulsion_COM_vertical_power_W=work/duration,
        propulsion_COM_vertical_work_J=work,
        peak_power_source_time_s=float(phase_t[i]),
        force_at_peak_power_N=float(phase_F[i]),
        velocity_at_peak_power_m_s=float(phase_v[i]),
        peak_measured_vertical_GRF_N=float(force[t < takeoff].max()),
        COM_rise_bottom_to_takeoff_m=float(phase_y[-1]-phase_y[0]),
        bottom_vertical_velocity_m_s=float(phase_v[0]),
        marker_takeoff_vertical_velocity_m_s=float(phase_v[-1]),
        time_above_80_percent_peak_s=above_duration(phase_t, phase_P, .8*peak),
        COM_vertical_energy_change_J=energy_change,
        measured_work_minus_COM_energy_J=work-energy_change,
        work_energy_relative_difference=(work-energy_change)/abs(energy_change),
        force_velocity_peak_power_W=force_peak,
        force_velocity_peak_relative_difference=(force_peak-peak)/peak,
        force_integrated_takeoff_velocity_m_s=float(force_velocity[-1]),
        force_integration_anchor='Zero vertical speed at marker-derived lowest COM event; reported mass unchanged.',
        peak_values_are_sampled_not_continuous_maxima=True,
        residual_power_identity_max_error_W=identity_error,
        active_vertical_GRF_RMS_bodyweights=reference['active_phase_vertical_GRF_RMS_error_bodyweights'],
        provenance={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                    for p in [mechanics_path, plates_path]},
    )
    # Include lead-in and flight for plotting. Landing and subsequent samples
    # are not joined to propulsion or used for peak/mean/work statistics.
    view = (t >= onset-.6) & (t < reference['landing_s']-.025)
    arrays = dict(times_source_s=t[view], times_from_bottom_s=t[view]-bottom,
                  vertical_GRF_N=force[view], COM_vertical_velocity_m_s=v[view],
                  COM_height_from_bottom_m=y[view]-phase_y[0],
                  COM_vertical_power_W=P[view],
                  propulsion_times_source_s=phase_t,
                  propulsion_power_W=phase_P,
                  propulsion_force_integrated_power_W=force_power)
    np.savez_compressed(OUT/f'{trial}_power.npz', **arrays)
    with (OUT/f'{trial}_power.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_time_s', 'time_from_bottom_s', 'vertical_GRF_N',
                         'COM_vertical_velocity_m_s', 'COM_height_from_bottom_m', 'COM_vertical_power_W'])
        writer.writerows(zip(*(arrays[k] for k in [
            'times_source_s', 'times_from_bottom_s', 'vertical_GRF_N',
            'COM_vertical_velocity_m_s', 'COM_height_from_bottom_m', 'COM_vertical_power_W'])))
    return row, arrays


def comparison_spec():
    return dict(
        schema_version=1,
        status='measured_CMJ_baseline_ready_box_and_loaded_squat_trajectories_pending',
        subject={'sex':'male', 'height_m':1.8, 'mass_kg':88.},
        conditions=[{'id':f'box_{cm}', 'movement':'box_jump', 'box_height_m':cm/100,
                     'status':'trajectory_and_contact_not_yet_solved',
                     'power_result_W':None} for cm in [40,60,80]]
                   +[{'id':f'squat_{kg}', 'movement':'high_bar_back_squat',
                      'bar_mass_kg':kg, 'status':'trajectory_and_contact_not_yet_solved',
                      'power_result_W':None} for kg in [60,100,140]],
        timing={'source':'measured or validated movement-specific timing',
                'display_speed':1., 'replays_per_condition':3,
                'force_equal_cycle_duration':False, 'phase_time_warp':False,
                'independent_playback_clocks':True,
                'note':'Load alone does not identify a person-specific movement speed; no assumed 1RM or strength ceiling.'},
        primary_metric={'name':'Vertical system COM power', 'unit':'W',
                        'equation':'P_COM,z(t) = F_ground,z(t) * v_system_COM,z(t)',
                        'system':'All body segments plus bar where present',
                        'loaded_system_COM':'(sum(m_i*r_i) + m_bar*r_bar)/(sum(m_i)+m_bar)',
                        'interpretation':'COM translational pseudo-work rate; excludes segment-relative and rotational energy.'},
        secondary_metrics=['peak and mean positive-phase COM power',
                           'propulsion or ascent duration', 'time above 80% of own power peak',
                           'peak vertical GRF', 'net vertical impulse',
                           'bar-only power separately labelled', 'hip/knee/ankle net joint powers'],
        box_jump={'box_height_is_not_COM_rise':True,
                  'required_outputs':['box height', 'takeoff COM velocity', 'airborne COM rise',
                                      'foot clearance', 'landing COM height', 'takeoff and landing forces separately'],
                  'do_not_infer_power_from_box_height':True},
        validation=['Force and moment balance at all contacts',
                    'Separate system COM power from joint/muscle power',
                    'Impulse/velocity and work/COM energy checks',
                    'No residual force applied to hide mismatch',
                    'No arbitrary force or muscle strength clipping'],
        render={'language':'en', 'canvas_px':[1080,1920], 'bottom_reserved_px':384,
                'main_view':'full body, front oblique; posterior inset',
                'phase_cursor':'each condition uses its own physical time',
                'no_automatic_slow_motion':True},
        known_limitations=['CMJ baseline is hands-on-hips, not an arm-swing box jump.',
                           'A box-only change need not increase takeoff power at maximal intent.',
                           'Current muscle tension remains conditional; this audit does not validate muscle recruitment.',
                           'No published-data or model-based conversion from box centimetres to squat kilograms is assumed.'])


def plots(rows, arrays):
    colors=['#48c8b4','#efb45b','#a3a1ff']
    with plt.rc_context({'figure.facecolor':'#0d1726','axes.facecolor':'#142338',
                         'axes.edgecolor':'#7891a7','axes.labelcolor':'#d8e5ef',
                         'text.color':'#edf3fb','xtick.color':'#a6bcd1','ytick.color':'#a6bcd1',
                         'grid.color':'#52667a','legend.facecolor':'#142338'}):
        fig,axs=plt.subplots(2,2,figsize=(13,8.8),layout='constrained')
        for j,(row,data) in enumerate(zip(rows,arrays)):
            t=data['times_from_bottom_s']; color=colors[j]; label=f"Trial {j+1}"
            for ax,key,scale in [(axs[0,0],'COM_vertical_power_W',.001),
                                  (axs[0,1],'vertical_GRF_N',.001),
                                  (axs[1,0],'COM_vertical_velocity_m_s',1)]:
                ax.plot(t,data[key]*scale,color=color,label=label,lw=2)
            axs[0,0].plot(row['peak_power_source_time_s']-row['source_bottom_s'],
                          row['peak_COM_vertical_power_W']/1000,'o',color=color)
        for ax,title,unit in [(axs[0,0],'Vertical COM power','kW'),
                              (axs[0,1],'Both-feet ground reaction','kN'),
                              (axs[1,0],'COM vertical velocity','m/s')]:
            ax.set(title=title,ylabel=unit,xlabel='Seconds from lowest COM (actual time)')
            ax.axvline(0,color='#b5c2cf',lw=1,ls=':');ax.axhline(0,color='#708093',lw=.7)
            ax.set_xlim(-.6,.65);ax.grid(alpha=.25);ax.legend(fontsize=9)
        ax=axs[1,1];x=np.arange(3);w=.33
        a=[r['peak_COM_vertical_power_W']/1000 for r in rows]
        b=[r['force_velocity_peak_power_W']/1000 for r in rows]
        ax.bar(x-w/2,a,w,color='#48c8b4',label='Marker-derived COM velocity')
        ax.bar(x+w/2,b,w,color='#efb45b',label='Force-integrated velocity')
        for i,row in enumerate(rows):
            ax.text(i,max(a[i],b[i])+.05,f"{100*row['force_velocity_peak_relative_difference']:.1f}%",ha='center',fontsize=10)
        ax.set(title='Two velocity methods · same plates and bottom event',
               ylabel='Peak COM power [kW]',xticks=x,xticklabels=['Trial 1','Trial 2','Trial 3'],ylim=(0,4.7))
        ax.legend(fontsize=9);ax.grid(axis='y',alpha=.25)
        fig.suptitle('Measured CMJ power baseline · 180 cm / 88 kg',fontsize=20)
        fig.text(.5,-.095,'Three measured repetitions, hands on hips. Original forces and physical time preserved. Landing excluded.\nCOM power is not muscle power or energy expenditure. Method agreement is not an independent accuracy bound.',ha='center',fontsize=10)
        fig.savefig(OUT/'power_baseline.png',dpi=170,bbox_inches='tight');plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tests=analytic_checks()
    reference=json.loads((SOURCE/'summary.json').read_text())
    results=[analyze_trial(row) for row in reference['trials']]
    rows,arrays=map(list,zip(*results))
    spec=comparison_spec()
    spec_path=ROOT/'scenarios/jump_box_squat_power.json'
    spec_path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    (OUT/'comparison_setup.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    report=dict(
        status='CMJ_power_baseline_audited_comparison_incomplete',
        definition=spec['primary_metric'],
        data_source='https://doi.org/10.6084/m9.figshare.28890545',
        paper='https://doi.org/10.1038/s41597-025-05934-5',
        original_force_amplitudes_preserved=True, strength_clipping=False,
        gravity_m_s2=G, trials=rows, analytic_checks=tests,
        event_scope='Lowest marker-derived COM sample through exact force-plate takeoff, inclusive.',
        means_do_not_include_flight_or_display_replays=True,
        actual_power_accuracy_bound_established=False,
        human_muscle_tension_validated=False,
        limitations=[
            'COM position and velocity are estimated by a generic segment model fitted to markers.',
            'Force-integrated velocity shares force plates and the marker-derived bottom event; agreement is not independent validation.',
            'Reported participant mass is 88 kg; quiet force implies about 91.8 kg. No silent mass or force rescaling.',
            'Work-energy mismatch is retained, not corrected. Numerical identity checks do not prove physiological accuracy.',
            'Vertical COM power excludes segment-relative, rotational, muscle/tendon and metabolic power.',
            'Box and loaded-squat output values are not yet computed.'],
        references=[
            {'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC11166134/',
             'use':'Box-jump height study: maximal-intent mean power did not systematically rise with box height in this sample. Not a prediction for this participant or 80 cm boxes.'},
            {'url':'https://pubmed.ncbi.nlm.nih.gov/32187147/',
             'use':'Low/high box jump comparison; no significant peak-power difference in ten female handball players.'},
            {'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC11412665/',
             'use':'Natural-speed squat source screened. Force-only reconstruction is not accepted as a loaded-squat power result.'}])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plots(rows,arrays)
    print(json.dumps({'status':report['status'],'trials':[{k:r[k] for k in ['trial','peak_COM_vertical_power_W','mean_propulsion_COM_vertical_power_W','propulsion_duration_s','work_energy_relative_difference','force_velocity_peak_relative_difference']} for r in rows]},indent=2))


if __name__=='__main__':
    main()
