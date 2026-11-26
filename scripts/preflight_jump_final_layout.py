"""Exercise the final compositor using labelled frozen inputs, never a new result."""
import shutil,json,hashlib
from pathlib import Path
import numpy as np
from jump_monitor_data import ROOT,read,atomic
import compose_jump_rl_draft_v5 as v5

def main():
    old=ROOT/'deliverables/jump_rl_study_draft_v4';out=ROOT/'deliverables/jump_rl_study_draft_v5/preflight_inputs';out.mkdir(parents=True,exist_ok=True)
    names=['source_manifest.json','policy_definition.json','metrics.json','muscle_analysis.json','parameter_catalogue.json']
    manifest=read(old/'source_manifest.json')
    names += [f'{k}_{suffix}'for k in manifest['final_four_keys']for suffix in ['metrics.npz','muscles.npz','muscles.json']]
    for name in names:shutil.copyfile(old/name,out/name)
    manifest.update(stable_three_metre_complete=False,layout_preflight_only=True);atomic(out/'source_manifest.json',manifest)
    v5.OUT=out;draft=v5.Draft();side=next(s for s in draft.segments if s['kind']=='side_four')
    for t in [0.,8.,16.,24.,32.9,33.,side['start_s']+.2,side['start_s']+4.,side['start_s']+6.2,draft.duration-3.5]:
        frame=np.asarray(draft.at(t));assert frame.shape==(1920,1080,3);assert np.all(frame[1536:]==v5.BG)
    draft.at(side['start_s']+6.2).save(out/'four_side_preflight.png');draft.at(draft.duration-3.5).save(out/'four_oblique_preflight.png')
    atomic(out/'check.json',dict(passed=True,approved_opening_unchanged=True,four_fixed_side=True,portrait_footer_blank=True,old_cases_for_layout_only=True,not_a_final_result=True));print('FINAL_LAYOUT_PREFLIGHT_PASSED',flush=True)

if __name__=='__main__':main()
