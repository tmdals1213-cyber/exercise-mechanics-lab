"""Isolated high-flexion motion profile; muscle-path port is not yet validated.

Ankle range comes from the official RajagopalLaiUhlrich2023 model; high hip
and knee flexion ranges come from the local published Catelli model. Changing
ranges does not import either model's muscle geometry or establish accuracy.
"""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def apply(r):
    source=ROOT/'models/high_flex/RajagopalLaiUhlrich2023.osim';doc=ET.parse(source).getroot();catelli=ROOT/'datasets/cmj_calisti2025/source/catelli_3dof_model.osim';deep=ET.parse(catelli).getroot();changes=[]
    for side in ['r','l']:
        for stem in ['hip_flexion','knee_angle','ankle_angle']:
            name=stem+'_'+side;selected=doc if stem=='ankle_angle' else deep;coordinate=selected.find('.//Coordinate[@name="'+name+'"]');limits=np.fromstring(coordinate.findtext('range'),sep=' ')
            joint=r.b.joints[name];old=r.m.jnt_range[joint].copy();r.m.jnt_range[joint]=limits
            changes.append(dict(coordinate=name,old_range_rad=old.tolist(),new_range_rad=limits.tolist(),range_source='RajagopalLaiUhlrich2023' if stem=='ankle_angle' else 'Catelli published deep-squat model'))
    report=dict(kind='high_flexion_motion_profile_only',source_url='https://raw.githubusercontent.com/opensim-org/opensim-models/master/Models/Rajagopal/RajagopalLaiUhlrich2023.osim',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),ranges=changes,source_muscle_paths_imported=False,deep_flexion_muscle_force_validated=False,native_knee_coupler_extrapolation_requires_review=True)
    report.update(Catelli_source_file=str(catelli.relative_to(ROOT)),Catelli_source_sha256=hashlib.sha256(catelli.read_bytes()).hexdigest(),range_combination_is_a_project_motion_assumption=True)
    r.profile['high_flexion']=report;r.b.report['high_flexion']=report
    r.b.report['implementation_sha256']=hashlib.sha256((r.b.report['implementation_sha256']+json.dumps(report,sort_keys=True)).encode()).hexdigest()
    return report
