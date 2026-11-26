"""Independently scale the authors' reference model and fit measured markers.

No controller-generated trajectory, force amplitude scaling or residual-force
correction is used. This is not a claim to possess the original subject model.
"""
import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import opensim as osim
import numpy as np

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'datasets/cmj_calisti2025';OUT=ROOT/'results/measured_cmj'

def settext(parent,name,value):
    item=parent.find(name)
    if item is None:item=ET.SubElement(parent,name)
    item.text=value.as_posix() if isinstance(value,Path) else str(value)

def scale_model():
    source=DATA/'source';dest=OUT/'reference_model';dest.mkdir(exist_ok=True);modelpath=dest/'source_without_display_meshes.osim';tree=ET.parse(source/'catelli_3dof_model.osim')
    for node in tree.findall('.//attached_geometry'):
        for child in list(node):node.remove(child)
    tree.write(modelpath,encoding='utf-8',xml_declaration=True)
    subject=json.loads((DATA/'subject.json').read_text());tree=ET.parse(source/'setup_scaling_default.xml');scale=tree.getroot().find('ScaleTool');scale.set('name','measured_sub10');settext(scale,'mass',subject['mass_kg']);settext(scale,'height',subject['height_m']*1000);settext(scale,'age',0)
    maker=scale.find('GenericModelMaker');settext(maker,'model_file',modelpath.name);settext(maker,'marker_set_file','Unassigned')
    for section in ['ModelScaler','MarkerPlacer']:
        node=scale.find(section);settext(node,'marker_file','../static.trc');settext(node,'time_range','0.5 1.5');settext(node,'output_model_file','scaled_before_marker_fit.osim' if section=='ModelScaler' else 'scaled_model.osim')
    settext(scale.find('ModelScaler'),'output_scale_file','scale_factors.xml');settext(scale.find('MarkerPlacer'),'output_motion_file','static_pose.mot');settext(scale.find('MarkerPlacer'),'output_marker_file','placed_markers.xml')
    setup=dest/'scale_setup.xml';tree.write(setup,encoding='utf-8',xml_declaration=True);tool=osim.ScaleTool(str(setup));assert tool.run();path=dest/'scaled_model.osim';assert path.exists();model=osim.Model(str(path));state=model.initSystem();assert abs(model.getTotalMass(state)-subject['mass_kg'])<1e-5
    report=dict(subject_mass_kg=model.getTotalMass(state),body_count=model.getBodySet().getSize(),coordinates=model.getCoordinateSet().getSize(),markers=model.getMarkerSet().getSize(),scale_window_s=[.5,1.5],scaling='Authors measurement-based defaults and static trial; original participant-specific manually scaled model not available.',source_model_sha256=hashlib.sha256((source/'catelli_3dof_model.osim').read_bytes()).hexdigest(),scaled_model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),visual_meshes_removed_only=True,age_field='0 is unused informational placeholder, not a participant age',empirical_muscle_validation=False)
    (dest/'model_setup.json').write_text(json.dumps(report,indent=2));print('REFERENCE_MODEL_SCALED',json.dumps(report),flush=True);return path

def main():
    osim.Logger.setLevelString('info');path=scale_model();subject=json.loads((DATA/'subject.json').read_text());summary=[]
    for trial in subject['trials']:
        dest=OUT/trial;dest.mkdir(exist_ok=True);model=osim.Model(str(path));model.setUseVisualizer(False);model.initSystem();tool=osim.InverseKinematicsTool(str(DATA/'source/setup_IK_default.xml'));tool.setName(trial);tool.setModel(model);tool.setMarkerDataFileName(str(OUT/f'{trial}.trc'));tool.setOutputMotionFileName(str(dest/'ik.mot'));tool.setResultsDir(str(dest));z=np.load(OUT/f'{trial}_measured.npz');tool.setStartTime(float(z['times'][0]));tool.setEndTime(float(z['times'][-1]));tool.set_accuracy(1e-5);tool.set_report_errors(True);tool.set_report_marker_locations(True);tool.printToXML(str(dest/'ik_setup.xml'));assert tool.run();summary.append(dict(trial=trial,frames=len(z['times']),kinematics_file=f'{trial}/ik.mot',measured_external_forces=f'{trial}_grf.mot',IK_ran=True));print('MEASURED_IK_COMPLETE',trial,flush=True)
    (OUT/'replay_setup.json').write_text(json.dumps(dict(status='measured_marker_replay_completed',model=path.relative_to(OUT).as_posix(),trials=summary,source='https://doi.org/10.1038/s41597-025-05934-5',muscle_forces_validated=False),indent=2))

if __name__=='__main__':main()
