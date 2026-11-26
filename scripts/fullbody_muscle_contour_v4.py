"""Every exposed muscle surface gets a model tension or an explicit N/A style."""
import json,numpy as np
from atlas_mapping import map_atlas
COLORS=np.array([[43,97,142],[48,163,169],[240,183,78],[238,82,69]])
STOPS=np.log1p([0.,100.,1000.,30000.])

def setup(folder,anatomy,skin):
    import bpy
    from case_tension_render import linear
    z=np.load(folder/'muscles.npz');meta=json.loads((folder/'muscles.json').read_text());names=meta['names'];records=[];audit=[];unknown=[]
    for ob in anatomy:
        if ob==skin or ob.hide_render or not any(x.type=='ARMATURE'for x in ob.modifiers):continue
        ids=map_atlas(ob.name,names);ma=bpy.data.materials.new('Full model tension '+ob.name);ma.use_nodes=True;shader=ma.node_tree.nodes.get('Principled BSDF');shader.inputs['Roughness'].default_value=.62;ob.data.materials.clear();ob.data.materials.append(ma)
        if not ids:
            shader.inputs['Base Color'].default_value=(*linear(np.array([109,116,125])),1);unknown.append(ob.name);audit.append(dict(mesh=ob.name,status='not_identified_by_this_model',force_N=None));continue
        values=z['force_N'][:,ids].sum(1);records.append((shader,values));audit.append(dict(mesh=ob.name,status='conditional_model_tension',model_elements=[names[i]for i in ids],peak_N=float(values.max())))
    (folder/'fullbody_contour_mapping.json').write_text(json.dumps(dict(mapped_surfaces=len(records),unidentified_surfaces=unknown,anatomical_surface_count=len(audit),surfaces=audit,metric='Sum of conditional model path tensions explicitly assigned to this anatomical surface, N',color_stops_N=[0,100,1000,30000],palette=COLORS.tolist(),palette_is_common_to_all_candidates=True,unknown_is_not_zero=True,force_values_changed=False),indent=2))
    def update(t):
        for shader,values in records:
            f=float(np.interp(t,z['times'],values));color=np.array([np.interp(np.log1p(max(0.,f)),STOPS,COLORS[:,j])for j in range(3)]);shader.inputs['Base Color'].default_value=(*linear(color),1)
    update(float(z['times'][0]));return update
