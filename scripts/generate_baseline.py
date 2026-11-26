"""Prescribed OHP preview and BAR-ONLY mechanics; Python standard library only.

Coordinates in metres: X right, Y up, Z forward. No muscle-force solver here.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def scale(a, s):
    return [x * s for x in a]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def profile(t, duration, ramp, displacement):
    """Analytic position, velocity, acceleration with start/end at rest."""
    vmax = displacement / (duration - ramp)
    if t <= 0:
        return 0.0, 0.0, 0.0
    if t >= duration:
        return displacement, 0.0, 0.0
    if t > duration-ramp:
        p, v, a = profile(duration-t, duration, ramp, displacement)
        return displacement-p, v, -a
    if t < ramp:
        angle = math.pi*t/ramp
        return (vmax*0.5*(t-ramp/math.pi*math.sin(angle)),
                vmax*0.5*(1-math.cos(angle)),
                vmax*0.5*math.pi/ramp*math.sin(angle))
    return vmax*(t-ramp/2), vmax, 0.0


def elbow_position(shoulder, grip, upper, lower):
    """Two-link IK with elbow-down pole; lower includes hand-to-grip distance.

    This is an illustrative elbow/hand chain, not an anatomical wrist model.
    """
    delta = sub(grip, shoulder)
    distance = norm(delta)
    if distance < abs(upper-lower)-1e-8 or distance > upper+lower+1e-8:
        raise ValueError(f"Unreachable grip target: {distance:.5f} m")
    direction = scale(delta, 1/distance)
    pole = [0.0, -1.0, 0.0]
    perpendicular = sub(pole, scale(direction, dot(pole, direction)))
    if norm(perpendicular) < 1e-8:
        perpendicular = [0.0, 0.0, 1.0]
    perpendicular = scale(perpendicular, 1/norm(perpendicular))
    along = (upper*upper-lower*lower+distance*distance)/(2*distance)
    height = math.sqrt(max(0.0, upper*upper-along*along))
    return add(add(shoulder, scale(direction, along)), scale(perpendicular, height))


def generate(config):
    s, g, motion, load = (config[k] for k in ('subject', 'geometry', 'movement', 'load'))
    duration, ramp, hz = motion['ascent_s'], motion['ramp_s'], motion['sample_hz']
    assert duration > 0 and 0 < ramp < duration/2 and hz > 0
    assert motion['lockout_hold_s'] >= 0
    assert s['height_m'] > 0 and s['mass_kg'] > 0
    assert load['barbell_total_mass_kg'] > 0
    assert all(g[k] > 0 for k in ('upper_arm_m', 'forearm_m', 'wrist_to_grip_m', 'shoulder_width_m', 'grip_width_m'))
    shoulder_y = s['height_m']*g['shoulder_height_fraction']
    upper, lower = g['upper_arm_m'], g['forearm_m']+g['wrist_to_grip_m']
    lateral = (g['grip_width_m']-g['shoulder_width_m'])/2
    assert abs(lateral) < upper+lower
    lockout_y = shoulder_y+math.sqrt((upper+lower)**2-lateral**2)
    rise = lockout_y-g['bar_start_height_m']
    assert rise > 0
    frames = []
    count = round((duration+motion['lockout_hold_s'])*hz)
    mass, gravity = load['barbell_total_mass_kg'], load['gravity_m_s2']
    for index in range(count+1):
        t = index/hz
        displacement, vy, ay = profile(t, duration, ramp, rise)
        progress = displacement/rise
        y = g['bar_start_height_m']+displacement
        z0 = g['bar_start_forward_offset_m']
        z = z0*(1-progress)**2
        vz = -2*z0*(1-progress)*vy/rise
        az = 2*z0*((vy/rise)**2-(1-progress)*ay/rise)
        fy, fz = mass*(gravity+ay), mass*az
        phase = 'lockout_hold' if t >= duration else ('accelerating' if t < ramp else 'braking' if t > duration-ramp else 'constant_vertical_speed')
        shoulders = [[sign*g['shoulder_width_m']/2, shoulder_y, 0] for sign in (-1, 1)]
        grips = [[sign*g['grip_width_m']/2, y, z] for sign in (-1, 1)]
        elbows = [elbow_position(sh, grip, upper, lower) for sh, grip in zip(shoulders, grips)]
        hand_force = [0, -fy/2, -fz/2]  # bar on each hand; zero lateral squeeze assumed.
        arm_rows = []
        for shoulder, elbow, grip in zip(shoulders, elbows, grips):
            wrist = add(elbow, scale(sub(grip, elbow), g['forearm_m']/lower))
            angle_cos = max(-1, min(1, dot(sub(shoulder, elbow), sub(grip, elbow))/(upper*lower)))
            arm_rows.append({'shoulder': shoulder, 'elbow': elbow, 'wrist': wrist, 'grip': grip,
                             'elbow_flexion_deg': 180-math.degrees(math.acos(angle_cos))})
        frames.append({'time_s': t, 'phase': phase, 'bar_y_m': y, 'bar_z_m': z,
                       'bar_vy_m_s': vy, 'bar_vz_m_s': vz, 'bar_ay_m_s2': ay,
                       'bar_az_m_s2': az, 'bar_support_y_N': fy, 'bar_support_z_N': fz,
                       'per_hand_resultant_N': math.hypot(fy, fz)/2,
                       'right_shoulder_bar_only_external_moment_Nm': norm(cross(sub(grips[1], shoulders[1]), hand_force)),
                       'right_elbow_bar_only_external_moment_Nm': norm(cross(sub(grips[1], elbows[1]), hand_force)),
                       'bar_input_power_W': fy*vy+fz*vz, 'arms': arm_rows})
    integrated_work = sum((b['time_s']-a['time_s'])*(a['bar_input_power_W']+b['bar_input_power_W'])/2
                          for a, b in zip(frames, frames[1:]))
    summary = {
        'scenario': config['id'], 'stage': config['analysis']['stage'],
        'subject_height_m': s['height_m'], 'subject_mass_kg': s['mass_kg'],
        'bar_mass_kg': mass, 'ascent_s': duration, 'constant_speed_interval_s': [ramp, duration-ramp],
        'lockout_height_m': lockout_y, 'rise_m': rise,
        'constant_vertical_speed_m_s': rise/(duration-ramp),
        'static_bar_weight_N': mass*gravity,
        'peak_total_vertical_support_N': max(f['bar_support_y_N'] for f in frames),
        'minimum_total_vertical_support_N': min(f['bar_support_y_N'] for f in frames),
        'bar_gravitational_work_J': mass*gravity*rise,
        'bar_input_work_integrated_J': integrated_work,
        'muscle_forces': None, 'muscle_stress': None, 'fatigue': None, 'damage': None,
        'limitations': [
            'Geometry proportions are editable illustration assumptions, not validated male population averages.',
            'Successful motion is prescribed; 100 kg feasibility is not inferred from height or body mass.',
            'Body mass is metadata at this stage; limb gravity/inertia and full joint inverse dynamics are not included.',
            'Moment outputs include bar loading only, not net anatomical joint or muscle moments.',
            'Hands split bar load equally; no lateral squeeze forces, wrist mobility, scapular mechanics or head clearance solution.',
            'Vertical speed is constant only in the middle; the bar also moves backwards along a chosen illustrative path.'
        ]
    }
    return frames, summary


def verify(config, frames, summary):
    g = config['geometry']
    upper, lower = g['upper_arm_m'], g['forearm_m']+g['wrist_to_grip_m']
    assert abs(frames[0]['bar_vy_m_s']) < 1e-10
    assert abs(frames[-1]['bar_vy_m_s']) < 1e-10
    assert abs(frames[-1]['bar_y_m']-summary['lockout_height_m']) < 1e-10
    assert min(f['bar_support_y_N'] for f in frames) > 0, 'Motion would require downward bar control.'
    assert all(b['bar_y_m'] >= a['bar_y_m']-1e-10 for a, b in zip(frames, frames[1:]))
    for frame in frames:
        for arm in frame['arms']:
            assert abs(norm(sub(arm['elbow'], arm['shoulder']))-upper) < 1e-7
            assert abs(norm(sub(arm['grip'], arm['elbow']))-lower) < 1e-7
    assert max(a['elbow_flexion_deg'] for a in frames[-1]['arms']) < 1e-4
    assert abs(summary['bar_input_work_integrated_J']-summary['bar_gravitational_work_J']) < 0.1
    assert summary['muscle_forces'] is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=Path, default=ROOT/'scenarios/ohp_100kg_2s.json')
    args = parser.parse_args()
    config = json.loads(args.scenario.read_text(encoding='utf-8-sig'))
    frames, summary = generate(config)
    verify(config, frames, summary)
    out = ROOT/'results'/config['id']
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'motion.json').write_text(json.dumps({'config': config, 'frames': frames}, ensure_ascii=False), encoding='utf-8')
    fields = [key for key in frames[0] if key != 'arms']
    with (out/'bar_mechanics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: frame[key] for key in fields} for frame in frames)
    template = (ROOT/'assets/preview_template.html').read_text(encoding='utf-8')
    payload = json.dumps({'config': config, 'frames': frames, 'summary': summary}, ensure_ascii=False).replace('<', '\\u003c')
    (out/'preview.html').write_text(template.replace('__PAYLOAD__', payload), encoding='utf-8')
    print('PASS: rest endpoints, rigid link lengths, lockout, positive support, work-energy balance.')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Preview: {out / "preview.html"}')


if __name__ == '__main__':
    main()
