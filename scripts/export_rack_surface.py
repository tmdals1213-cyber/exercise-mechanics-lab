"""Export existing rendered upper-chest skinning for geometric rack placement."""
from pathlib import Path
import json,bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/grip_comparison'
bpy.ops.wm.open_mainfile(filepath=str(OUT/'grip_65/cycle/physiology/fullbody_scene.blend'))
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');bones={b.name:np.array(b.matrix_local).tolist() for b in rig.data.bones}
objects=[]
for ob in bpy.data.objects:
 if ob.type!='MESH' or ob.hide_render or 'pectoralis major' not in ob.name.lower():continue
 assert any(m.type=='ARMATURE' for m in ob.modifiers)
 assert not next(m for m in ob.modifiers if m.type=='ARMATURE').use_deform_preserve_volume
 objects.append({'name':ob.name,'vertices':[list(v.co) for v in ob.data.vertices],
  'faces':[list(p.vertices) for p in ob.data.polygons],
  'weights':[[[ob.vertex_groups[g.group].name,g.weight] for g in v.groups] for v in ob.data.vertices],
  'mesh_to_rig':np.array(rig.matrix_world.inverted()@ob.matrix_world).tolist()})
assert len(objects)>=2
skin=bpy.data.objects.get('Skin');deps=bpy.context.evaluated_depsgraph_get();evaluated=skin.evaluated_get(deps)
root=bpy.data.objects['Target stature 180cm'];transform=root.matrix_world.inverted()@evaluated.matrix_world
v=np.array([list(transform@p.co) for p in evaluated.data.vertices])
head_groups=set()
for bone in rig.data.bones:
 ancestor=bone
 while ancestor:
  if ancestor.name=='head':head_groups.add(bone.name);break
  ancestor=ancestor.parent
head_weights=[sum(g.weight for g in vertex.groups if skin.vertex_groups[g.group].name in head_groups) for vertex in skin.data.vertices]
metadata=json.loads((OUT/'grip_65/cycle/physiology/fullbody_metadata.json').read_text());T=np.array(metadata.get('head_pose_transform_native',np.eye(4)));w=np.array(head_weights)
# Recover the neutral head reference even when the saved scene uses extension.
A=(1-w[:,None,None])*np.eye(3)[None]+w[:,None,None]*T[:3,:3]
v=np.linalg.solve(A,(v-w[:,None]*T[:3,3])[:,:,None])[:,:,0]
selected=v[:,2]>1.58;faces=[list(p.vertices) for p in evaluated.data.polygons if all(selected[i] for i in p.vertices)]
(OUT/'rack_surface.json').write_text(json.dumps({'bones':bones,'objects':objects,'head_native_vertices':v.tolist(),'head_weights':head_weights,'head_faces':faces},separators=(',',':')),encoding='utf-8')
print('RACK_SURFACE_EXPORTED',len(objects),flush=True)
