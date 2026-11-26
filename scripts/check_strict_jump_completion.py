"""Boundary checks for the explicitly requested four-timestep stop condition."""
import copy
from train_landing_stability import qualified_completion
from jump_monitor_data import ROOT,atomic

def main():
    rule=dict(min_retained_m=3.,required_timesteps_s=[.0005,.00025,.000125,.0000625])
    cases=[dict(dt_s=dt,stable=True,eligible=True,retained_distance_m=3.)for dt in rule['required_timesteps_s']]
    assert qualified_completion(cases,rule,True)
    assert not qualified_completion(cases[:3],rule,True)
    assert not qualified_completion(cases,rule,False)
    for field,value in [('retained_distance_m',2.999999999),('stable',False),('eligible',False)]:
        changed=copy.deepcopy(cases);changed[-1][field]=value;assert not qualified_completion(changed,rule,True)
    atomic(ROOT/'deliverables/jump_landing_live/strict_completion_check.json',dict(passed=True,all_four_required=True,rounding_cannot_trigger_stop=True,each_stable_and_eligible_required=True));print('STRICT_COMPLETION_BOUNDARIES_PASSED',flush=True)

if __name__=='__main__':main()
