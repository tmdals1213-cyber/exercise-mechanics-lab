"""Refine the newly all-four-stable2.9919m knee-drive candidate."""
import copy
import search_jump_launch_batch as batch
from jump_monitor_data import ROOT,read

def proposals(unused):
    report=read(ROOT/'deliverables/jump_robust_launch_batch/best_validation.json');assert report['passed'];seed=report['cases'][0]['parameters']
    for delta in [-.000125,.000125,-.00025,.00025,-.0000625,.0000625,-.000375,.000375]:
        p=copy.deepcopy(seed);p['propulsion_effort_knots_fraction']['knee_angle']=[v+delta for v in p['propulsion_effort_knots_fraction']['knee_angle']]
        yield p,dict(method='four_step_best_knee_drive_refinement',seed_report='deliverables/jump_robust_launch_batch/best_validation.json',knee_effort_delta_fraction=delta)
    for key,step in [('prepare_hip',.001),('prepare_knee',.001),('prepare_transition_s',.00025),('pitch_top',.00025),('push_s',.0000625)]:
        for delta in [-step,step]:
            p=copy.deepcopy(seed);p[key]+=delta;yield p,dict(method='four_step_best_small_coordinate_refinement',key=key,delta=delta)

if __name__=='__main__':
    batch.LABEL='robust_knee_refine_v1';batch.WEBNAME='jump_robust_knee_refine';batch.PROGRESS_NAME='robust_knee_refine_status.json';batch.proposals=proposals;batch.main()
