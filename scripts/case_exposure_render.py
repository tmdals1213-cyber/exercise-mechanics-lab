"""One-cycle force-time contours, on a common standing display pose."""
import json,numpy as np,bpy
from atlas_mapping import map_atlas
from case_tension_render import linear
from launch_relative_render_palette import color

def setup(folder,anatomy,skin):
 data=json.loads((folder/'analysis/exposure.json').read_text());names=data['names'];values=np.array(data['element_integral_Ns']);upper=data['common_upper_Ns'];audit=[]
 for ob in anatomy:
  if ob==skin or not any(m.type=='ARMATURE' for m in ob.modifiers):continue
  ids=map_atlas(ob.name,names)
  if not ids:continue
  total=float(values[ids].sum());ma=bpy.data.materials.new('Force time '+ob.name);ma.use_nodes=True;ma.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.6;ma.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(*linear(color(total/upper)),1);ob.data.materials.clear();ob.data.materials.append(ma);audit.append(dict(mesh=ob.name,integral_Ns=total,elements=[names[i] for i in ids]))
 (folder/'exposure_render_mapping.json').write_text(json.dumps(dict(common_upper_Ns=upper,meshes=audit,metric='Integral F dt; one active movement; not mean tension'),indent=2))
 return lambda t:None
