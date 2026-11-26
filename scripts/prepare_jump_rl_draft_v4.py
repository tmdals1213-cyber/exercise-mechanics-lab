from pathlib import Path
import copy,shutil
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v4';OLD=ROOT/'deliverables/jump_rl_study_draft_v3'
def main():
    m=copy.deepcopy(read(OLD/'source_manifest.json'));analysis=read(OUT/'muscle_analysis.json');m.update(version=4,preserve_first_seconds=33.,final_views=['front_oblique','rear_oblique','world_fixed_side_with_postlanding_COM_path'],final_summary_view='four_simultaneous_front_oblique_from_above',fullbody_tension_display=True,relative_muscle_reference_N=analysis['cases']['initial']['primary_peaks_N'])
    for c in m['cases']:
        if c['key']not in m['final_four_keys']:continue
        old=Path(c['multiview_folder']);folder=OUT/'scenes'/c['key'];folder.mkdir(parents=True,exist_ok=True)
        for name in ['motion.npz','control_summary.json','summary.json','reference_pose.npz','standing_pose.npz','body_names.json','joint_names.json','control_model.mjb','joint_overlay.npz']:shutil.copyfile(old/name,folder/name)
        shutil.copyfile(OUT/f'{c["key"]}_muscles.npz',folder/'muscles.npz');shutil.copyfile(OUT/f'{c["key"]}_muscles.json',folder/'muscles.json');c.update(multiview_folder=str(folder),multiview_fps=24,multiview_dir='front_rear_render',full_side_dir='fullbody_side_render',full_side_fps=24)
    atomic(OUT/'source_manifest.json',m)
    for name in ['metrics.json','policy_definition.json','ANATOMY_CREDITS.md','parameter_catalogue.json','training_compatibility.json']:shutil.copyfile(OLD/name,OUT/name)
    for key in m['final_four_keys']:shutil.copyfile(OLD/f'{key}_metrics.npz',OUT/f'{key}_metrics.npz')
    print('V4_SOURCES_READY',flush=True)
if __name__=='__main__':main()
