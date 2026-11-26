"""Keep the approved opening and real trials; add four multiview muscle cases."""
from pathlib import Path
import copy,shutil,time,json
import numpy as np
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v3';OLD=ROOT/'deliverables/jump_rl_study_draft_v2'

def main():
    m=copy.deepcopy(read(OLD/'source_manifest.json'));analysis=read(OUT/'muscle_analysis.json');ref={}
    for key in m['final_four_keys']:
        p=OUT/f'{key}_muscles.npz';z=dict(np.load(p));a=analysis['cases'][key];ids=[i for i,n in enumerate(a['names'])if n.endswith('_r')and n.startswith(('bflh','bfsh','semimem','semiten'))];z['hamstrings']=z['force_N'][:,ids].sum(1);c=next(c for c in m['cases']if c['key']==key);r=read(Path(c['folder'])/'control_summary.json');par=r['parameters'];bottom=par['ready_s']+par['crouch_s']+par.get('prepush_lean_s',0);mask=(z['times']>=bottom)&(z['times']<r['takeoff_s']);a['primary_peaks_N']['hamstrings']=float(z['hamstrings'][mask].max());np.savez_compressed(p,**z);atomic(OUT/f'{key}_muscles.json',a)
        folder=OUT/'scenes'/key;folder.mkdir(parents=True,exist_ok=True)
        for name in ['motion.npz','control_summary.json','summary.json','reference_pose.npz','standing_pose.npz','body_names.json','joint_names.json','control_model.mjb','joint_overlay.npz']:shutil.copyfile(Path(c['folder'])/name,folder/name)
        shutil.copyfile(p,folder/'muscles.npz');shutil.copyfile(OUT/f'{key}_muscles.json',folder/'muscles.json');c.update(multiview_folder=str(folder),multiview_fps=24,multiview_dir='study_render')
    ref=analysis['cases']['initial']['primary_peaks_N']
    for key in m['final_four_keys']:atomic(OUT/'scenes'/key/'muscle_reference.json',dict(reference='Initial successful model push-off peak, right side',peak_N=ref,normalization_is_not_strength_or_activation=True))
    atomic(OUT/'muscle_analysis.json',analysis);m.update(version=3,prepared_at=time.time(),muscle_tension_available=True,muscle_tension_is_conditional=True,display_muscle_keys=['glutes','quads','hamstrings','calves'],relative_muscle_reference_N=ref,persistent_failure_ghosts=4,parameter_controls_visible=True,final_views=['front_oblique','true_side'])
    atomic(OUT/'source_manifest.json',m)
    for name in ['metrics.json','policy_definition.json','ANATOMY_CREDITS.md']:shutil.copyfile(OLD/name,OUT/name)
    for key in m['final_four_keys']:shutil.copyfile(OLD/f'{key}_metrics.npz',OUT/f'{key}_metrics.npz')
    config=read(OUT/'policy_definition.json');catalog=[]
    for a in config['actions']:
        key=a['key'];phase='push drive'if a['kind']=='equivalent_effort'else'air / landing preparation'if key.startswith(('flight_','prepare_'))else'postcontact recovery'if key.startswith('landing_')else'ground posture / timing';catalog.append(dict(key=key,kind=a['kind'],unit=a['unit'],phase=phase,changes_airborne_COM_path_directly=False,values_by_stage={c['key']:(read(Path(c['folder'])/'control_summary.json')['parameters'][key]if a['kind']=='posture_timing'else read(Path(c['folder'])/'control_summary.json')['parameters']['propulsion_effort_knots_fraction'][a['stem']][a['knot']])for c in m['cases']if c['key']in m['final_four_keys']}))
    atomic(OUT/'parameter_catalogue.json',dict(outputs=58,posture_timing=43,equivalent_push_drive=15,torque_channels=11,sagittal_root_DOF=3,parameters=catalog,air_control='Internal joint torques change posture and foot reach. During continuous contact-free flight, COM remains ballistic and whole-body angular momentum is conserved. No midair propulsion.',learning='One policy action chooses the parameter sequence for an entire jump; a fixed PD/state controller executes it.'))
    print('V3_SOURCES_READY',ref,flush=True)

if __name__=='__main__':main()
