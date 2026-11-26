"""Finite-difference moment arms and file/equilibrium cross-checks."""
import argparse,json
import numpy as np
from solve_fullbody import Body,ROOT,CONFIG


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--grip',type=int,choices=[65,80],required=True); args=ap.parse_args()
    dest=ROOT/f'results/model_foundation/grip_{args.grip}'; z=np.load(dest/'telemetry.npz'); motion=np.load(dest/'fullbody_motion.npz'); op=np.load(dest/'analysis_operators.npz')
    source=np.load(ROOT/f'results/grip_comparison/grip_{args.grip}/cycle/physiology/fullbody_motion.npz')
    assert np.array_equal(motion['qpos'],source['qpos']), 'Movement changed in an analysis-only revision'
    assert abs(CONFIG['grip_width_m']-args.grip/100)<1e-10
    for key in z.files: assert np.isfinite(z[key]).all(),key
    assert np.min(z['force_N'])>=0
    assert abs(z['force_N']-z['active_force_N']-z['passive_force_N']).max()<1e-7
    # Shoulder forces reconstructed from a separate cached affine spatial-
    # tendon operator, adjusted to the unified gravity, must agree with all-
    # segment cut force assembly.
    shoulder=op['shoulder_force_offset']-np.einsum('tscm,tm->tsc',op['shoulder_force_coeff'],z['force_N'])
    shoulder_error=float(abs(shoulder-z['joint_force_world_N'][:,[0,6]]).max())
    assert shoulder_error<1e-6,shoulder_error
    b=Body(); rows=[]; max_error=0.; eps=1e-6
    for frame in [0,30,60,90,138]:
        q=motion['qpos'][frame]
        for j,jid in enumerate(b.independent):
            address=b.m.jnt_qposadr[jid]; plus=q.copy(); minus=q.copy(); plus[address]+=eps; minus[address]-=eps
            b.pose(plus); lp=b.d.actuator_length.copy(); b.pose(minus); lm=b.d.actuator_length.copy()
            finite=-(lp-lm)/(2*eps); expected=op['R'][frame,6+j]
            error=float(abs(finite-expected).max()); max_error=max(max_error,error)
            if error>2e-5:
                muscle=int(np.argmax(abs(finite-expected))); rows.append({'frame':frame,'coordinate':b.ind_names[j],'muscle':b.names[muscle],'error_native_m_per_rad':error})
    result={'status':'passed' if not rows else 'geometry_review_required','grip_cm':args.grip,'frames':[0,30,60,90,138],
            'independent_joint_coordinates_checked':len(b.independent),'muscle_elements':b.m.nu,'finite_difference_step_rad':eps,
            'maximum_native_moment_arm_difference_m_per_rad':max_error,'review_threshold_m_per_rad':2e-5,'review_items':rows,
            'independent_affine_shoulder_force_error_N':shoulder_error,'motion_unchanged':True,
            'scope':'Internal path derivative / force consistency. Does not establish anatomical accuracy or agreement with the original OpenSim moment arms.'}
    (dest/'geometry_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)
    if rows: raise RuntimeError('Native muscle path derivative needs review')


if __name__=='__main__': main()
