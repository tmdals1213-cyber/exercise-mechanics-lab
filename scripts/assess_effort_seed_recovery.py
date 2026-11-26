"""Run the same effort seed through recovery, without changing its primary score."""
import copy
from controlled_broad_jump import ROOT,run
from jump_monitor_data import read,atomic
from effort_launch_objective import evaluate_launch
import mujoco

def main():
    root=ROOT/'results/controlled_broad_jump/effort_launch_seed_v2';selected=read(root/'initial_solution.json');assert selected is not None
    p=copy.deepcopy(selected['parameters']);p.pop('stop_after_touchdown_s',None);p.update(seconds=6.,record_energy_ledger=True)
    out=root/'recovery_review_250us';report,z=run(p,out,dt=.00025);m=evaluate_launch(report,z,mujoco.MjModel.from_binary_path(str(out/'control_model.mjb')))
    expected=selected['cases'][1]['heel_projection_at_first_foot_contact_m'];error=abs(report['first_contact_heel_distance_m']-expected);assert error<1e-8
    result=dict(status=report['status'],recovery_evaluated=True,recovery_stable=report['stable_landing'],first_contact_replay_error_m=error,first_contact_distance_m=report['first_contact_heel_distance_m'],nearest_supported_contact_distance_m=report.get('valid_contact_distance_m'),metrics=m,primary_launch_score_unchanged=True,scope='Existing rigid-foot feedback controller only; active toes/stepping not added and human recoverability not determined.')
    atomic(root/'recovery_review.json',result);print('EFFORT_RECOVERY_REVIEW',report['status'],'primary replay error',error,flush=True)

if __name__=='__main__':main()
