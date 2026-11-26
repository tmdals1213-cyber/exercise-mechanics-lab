"""Common absolute anatomical tension colours, using only the current case."""
import json
import numpy as np,bpy
from atlas_mapping import map_atlas
from case_tension_render import linear
from launch_relative_render_palette import color

def setup(folder,anatomy,skin,upper=30000.):
    data=json.loads((folder/'analysis/visualization/data.json').read_text())
    z=np.load(folder/'analysis/tensions.npz');times=z['times'];forces=z['force_N'];names=data['names'];items=[];audit=[]
    for ob in anatomy:
        if ob==skin or not any(m.type=='ARMATURE'for m in ob.modifiers):continue
        ids=map_atlas(ob.name,names)
        if not ids:continue
        trace=forces[:,ids].sum(1);ma=bpy.data.materials.new('Absolute tension '+ob.name);ma.use_nodes=True
        ma.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.6
        ob.data.materials.clear();ob.data.materials.append(ma);items.append((ma,trace))
        audit.append(dict(mesh=ob.name,elements=[names[i]for i in ids],peak_N=float(trace.max())))
    (folder/'absolute_render_mapping.json').write_text(json.dumps(dict(meshes=audit,upper_N=upper,normalization='Common absolute linear anatomical muscle tension; not activation or stress.',forces_changed=False),indent=2))
    window=data.get('display_valid_window_s')
    def update(t):
        valid=window is None or window[0]<=t<=window[1]
        for ma,trace in items:
            c=color(float(np.interp(t,times,trace))/upper) if valid else np.array([147.,160.,171.])
            ma.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(*linear(c),1)
    return update
