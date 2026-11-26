"""Per-anatomical-muscle colours from existing element forces; no new force solve."""
import json
from pathlib import Path
import numpy as np
import bpy
from atlas_mapping import map_atlas
from case_tension_render import linear
from launch_relative_render_palette import color
from launch_sources import analysis_path

def setup(root,folder,anatomy,skin):
    cases={}
    for key in ['cmj','squat_100']:
        p=analysis_path(root,key)
        d=json.loads((p/'visualization/data.json').read_text());z=np.load(p/'tensions.npz')
        cases[key]=(np.array(z['times']),np.array(z['force_N']),d['names'])
    current='cmj' if folder.name=='cmj' else 'squat_100'
    times,forces,names=cases[current];items=[];audit=[]
    for ob in anatomy:
        if ob==skin or not any(m.type=='ARMATURE' for m in ob.modifiers):continue
        ids=map_atlas(ob.name,names)
        if not ids:continue
        trace=forces[:,ids].sum(axis=1)
        ref=max(float(f[:,[n.index(names[i]) for i in ids]].sum(axis=1).max()) for _,f,n in cases.values())
        ma=bpy.data.materials.new('Absolute anatomical tension '+ob.name);ma.use_nodes=True
        ma.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.6
        ob.data.materials.clear();ob.data.materials.append(ma);items.append((ma,trace,ref))
        audit.append(dict(mesh=ob.name,elements=[names[i] for i in ids],reference_peak_N=ref))
    assert any('rectus femoris' in x['mesh'].lower() for x in audit)
    upper=float(np.ceil(max(x['reference_peak_N'] for x in audit)/1000)*1000)
    (folder/'study_muscle_mapping.json').write_text(json.dumps(dict(meshes=audit,common_upper_N=upper,
        normalization='One common absolute linear N scale for all anatomical muscles and both movements; not activation.',
        forces_changed=False),indent=2))
    def update(t):
        for ma,trace,ref in items:
            u=float(np.interp(t,times,trace))/upper
            ma.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(*linear(color(u)),1)
    return update

def setup_exposure(root,folder,anatomy,skin):
    report=json.loads((root/'deliverables/instagram_launch/tension_exposure.json').read_text());cases=report['cases']
    current='cmj' if folder.name=='cmj' else 'squat_100';names=cases[current]['names'];items=[];audit=[]
    for ob in anatomy:
        if ob==skin or not any(m.type=='ARMATURE' for m in ob.modifiers):continue
        ids=map_atlas(ob.name,names)
        if not ids:continue
        val=[float(np.array(c['element_integral_Ns'])[ids].sum()) for c in cases.values()]
        items.append((ob,val));audit.append(dict(mesh=ob.name,elements=[names[i] for i in ids],integrals_Ns=val))
    upper=float(np.ceil(max(max(v) for _,v in items)/1000)*1000)
    for ob,values in items:
        ma=bpy.data.materials.new('Integrated tension '+ob.name);ma.use_nodes=True
        value=values[0 if current=='cmj' else 1]
        ma.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(*linear(color(value/upper)),1)
        ma.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.6
        ob.data.materials.clear();ob.data.materials.append(ma)
    for ma in bpy.data.materials:
        if ma.name.startswith('Joint reaction '):ma.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.004,.004,.005,1)
    (folder/'exposure_render_mapping.json').write_text(json.dumps(dict(meshes=audit,common_upper_Ns=upper,
        normalization='One common absolute linear N s scale for all displayed muscles and both movements.',
        standing_pose_is_display_only=True),indent=2))
    return lambda t:None
