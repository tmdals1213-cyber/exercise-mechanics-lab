"""Regression acceptance: reject the prior tumble, audit the corrected loads."""
import json,hashlib,numpy as np
from jump_rig import ROOT

def main():
    out=ROOT/'results/jump_contact_visualization';old=json.loads((ROOT/'results/jump_correction_audit/previous/motion_quality.json').read_text())
    assert old['status']=='rejected_for_vertical_jump'
    runs=[]
    for hz in [600,1200]:
        p=ROOT/f'results/contact_jump_scapula/{hz}hz';q=json.loads((p/'motion_quality.json').read_text());s=json.loads((p/'tension_summary.json').read_text());z=np.load(p/'tensions.npz');calf=json.loads((p/'calf_moment_audit.json').read_text())
        sha=hashlib.sha256((p/'motion.npz').read_bytes()).hexdigest()
        assert q['status']=='passed' and q['motion_sha256']==sha==s['source_motion_sha256']
        assert not s['strength_ceiling_enforced'] and not s['foot_intrinsic_and_fascia_force_solution_complete']
        residual=z['target']-np.einsum('tcn,tn->tc',z['R'],z['force_N'])
        mtp=[i for i,n in enumerate(s['coordinate_names']) if n.startswith('mtp_angle_')];keep=[i for i in range(residual.shape[1]) if i not in mtp]
        assert abs(residual[:,keep]).max()<1e-6
        assert np.max(abs(residual-z['unidentified_joint_support_moment_Nm']))<1e-8
        assert np.min(z['force_N']-z['passive'])>-.00001
        for a in calf['audits']:
            assert a['moment_arm_finite_difference_max_error_m']<1e-7
            assert abs(a['required_net_ankle_moment_Nm']-a['calf_muscle_moment_Nm']-a['other_muscle_moment_Nm'])<1e-6
        runs.append(dict(hz=hz,root_and_task_passed=True,non_MTP_moment_error_Nm=float(abs(residual[:,keep]).max()),calf_path_derivative_max_error_m=max(a['moment_arm_finite_difference_max_error_m'] for a in calf['audits'])))
    v=json.loads((out/'validation.json').read_text())
    assert v['status']=='mechanics_passed_muscle_sensitivity_unresolved' if not v['significant_force_groups_converged'] else v['status']=='conditional_numerical_checks_passed'
    result=dict(status='passed',prior_rotating_motion_rejected=True,runs=runs,all_muscle_forces_validated=False,sensitivity_status=v['status'])
    (out/'revision_regression_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
