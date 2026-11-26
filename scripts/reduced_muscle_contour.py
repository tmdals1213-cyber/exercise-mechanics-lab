"""Conditional leg-muscle group contours, relative to one fixed initial case."""
import json,numpy as np
from atlas_mapping import map_atlas
from case_tension_render import linear
PREFIX={'glutes':('glmax',),'quads':('vas','recfem'),'hamstrings':('bflh','bfsh','semimem','semiten'),'calves':('gas','soleus')}
COLORS=np.array([[32,63,93],[46,151,171],[240,188,84],[237,89,68],[173,39,57]])

def setup(folder,anatomy,skin):
    import bpy
    z=np.load(folder/'muscles.npz');meta=json.loads((folder/'muscles.json').read_text());reference=json.loads((folder/'muscle_reference.json').read_text())['peak_N'];names=meta['names'];records=[];audit=[]
    for ob in anatomy:
        if ob==skin or not any(x.type=='ARMATURE'for x in ob.modifiers):continue
        mapped=map_atlas(ob.name,names)
        if not mapped:continue
        key=next((key for key,prefix in PREFIX.items()if all(names[i].startswith(prefix)for i in mapped)),None)
        if key is None:continue
        side='l'if'left'in ob.name.lower()else'r';ids=[i for i,n in enumerate(names)if n.endswith('_'+side)and n.startswith(PREFIX[key])];series=z['force_N'][:,ids].sum(1)/reference[key];ma=bpy.data.materials.new('Conditional '+key+' '+side+' '+ob.name);ma.use_nodes=True;shader=ma.node_tree.nodes.get('Principled BSDF');shader.inputs['Roughness'].default_value=.62;ob.data.materials.clear();ob.data.materials.append(ma);records.append((shader,series));audit.append(dict(mesh=ob.name,group=key,side=side,model_elements=[names[i]for i in ids],reference_N=reference[key]))
    (folder/'conditional_contour_mapping.json').write_text(json.dumps(dict(mapped=audit,metric='Conditional group tension / initial successful push-off peak, per side',stops=[0,.5,1.,1.5,2.],colors=COLORS.tolist(),strength_or_activation_claim=False,forces_clipped=False,grey_anatomy='Groups outside the four selected leg groups are intentionally uncoloured.'),indent=2))
    def update(t):
        for shader,series in records:
            v=float(np.interp(t,z['times'],series));color=np.array([np.interp(v,[0,.5,1,1.5,2],COLORS[:,i])for i in range(3)]);shader.inputs['Base Color'].default_value=(*linear(color),1)
    update(float(z['times'][0]));return update
