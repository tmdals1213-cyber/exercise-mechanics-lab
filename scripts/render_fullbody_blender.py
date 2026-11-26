"""Blender 4.5: real BodyParts3D muscle surfaces driven by MyoSim bone transforms."""
import bpy,bmesh,json,sys,math,os
from pathlib import Path
import numpy as np
from mathutils import Matrix,Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from atlas_mapping import map_atlas
from visual_anatomy import enlarge_muscle_belly
from torso_surfaces import add_torso_surfaces,fit_lat_insertion,fit_lower_rectus_depth
OUT=Path(os.environ.get('EXERCISE_OUTPUT',str(ROOT/'results/ohp_fullbody')));frames=np.load(OUT/'fullbody_motion.npz')
meta=json.loads((OUT/'fullbody_metadata.json').read_text(encoding='utf-8'))
names=meta['body_names'];idx={n:i for i,n in enumerate(names)};f0=np.array(meta['reference_force_N'])
BODY_ALIAS={'pelvis':'sacrum'}


def matrix(position,rotation):
 M=Matrix.Identity(4);M.translation=Vector(position)
 for row in range(3):
  for col in range(3):M[row][col]=float(rotation[row*3+col])
 return M


def material(name,color,metal=0,rough=.45):
 mat=bpy.data.materials.new(name);mat.use_nodes=True
 p=mat.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Metallic'].default_value=metal;p.inputs['Roughness'].default_value=rough
 return mat


def color(r):
 stops=[(0,np.array([.08,.19,.32])),(.25,np.array([.08,.47,.65])),(.65,np.array([.18,.68,.56])),(1.,np.array([.95,.66,.12])),(1.6,np.array([.92,.12,.055])),(2.5,np.array([.62,.025,.04]))]
 for (a,ca),(b,cb) in zip(stops,stops[1:]):
  if r<=b:return tuple(ca+(cb-ca)*max(0,(r-a)/(b-a)))
 return tuple(stops[-1][1])


def main():
 bpy.ops.wm.read_factory_settings(use_empty=True)
 bpy.ops.import_scene.gltf(filepath=str(ROOT/'assets/fullbody/fullbody_muscles_skin.glb'))
 rig=next(o for o in bpy.data.objects if o.type=='ARMATURE')
 supplement_count=add_torso_surfaces(ROOT,rig)
 # Keep face/hands/feet from the CC0 skin; body muscles remain actual visible meshes.
 skin=bpy.data.objects.get('Skin')
 if skin:
  keep=[]
  for v in skin.data.vertices:
   weights={skin.vertex_groups[g.group].name:g.weight for g in v.groups}
   world=skin.matrix_world@v.co
   face=sum(w for n,w in weights.items() if n=='head')>.35 and (world.z>1.64 or (world.z>1.58 and world.y<.19))
   ends=sum(w for n,w in weights.items() if any(k in n for k in ['ph_','proxph','midph','distph','thumb','firstmc','lunate','calcn','toes','talus']))>.65
   keep.append(face or ends)
  bm=bmesh.new();bm.from_mesh(skin.data);bm.verts.ensure_lookup_table()
  bmesh.ops.delete(bm,geom=[v for v in bm.verts if not keep[v.index]],context='VERTS');bm.to_mesh(skin.data);bm.free()
  skin.data.materials.clear();skin.data.materials.append(material('Face hands feet',(.56,.58,.58),rough=.45))
 presentation=json.loads(Path(os.environ.get('EXERCISE_SCENARIO',str(ROOT/'scenarios/ohp_fullbody.json'))).read_text(encoding='utf-8'))['presentation']
 mapping=[];materials=[];bulk_changes=[];hidden_layers=[];insertion_fits=[];abdominal_fits=[]
 for ob in list(bpy.data.objects):
  if ob.type!='MESH' or ob==skin:continue
  # glTF helper joint widgets are not anatomy.
  if not any(m.type=='ARMATURE' for m in ob.modifiers):ob.hide_render=True;continue
  # Remove the superficial neck sheet in this anatomical layer view to expose
  # clavicular pectoralis. Its unmapped force is never invented or redistributed.
  if 'platysma' in ob.name.lower():
   ob.hide_render=True;hidden_layers.append(ob.name)
  abdomen_fit=fit_lower_rectus_depth(ob)
  if abdomen_fit:abdominal_fits.append(abdomen_fit)
  change=enlarge_muscle_belly(ob,presentation['visual_bulk'])
  if change:bulk_changes.append(change)
  fit=fit_lat_insertion(ob,ROOT,OUT)
  if fit:insertion_fits.append(fit)
  ids=map_atlas(ob.name,meta['muscle_names'])
  mat=material('Heat '+ob.name,color(0) if ids else (.34,.26,.23),rough=.55)
  principled=mat.node_tree.nodes.get('Principled BSDF')
  principled.inputs['Subsurface Weight'].default_value=.055
  ob.data.materials.clear();ob.data.materials.append(mat)
  for p in ob.data.polygons:p.use_smooth=True
  mapping.append({'mesh':ob.name,'actuator_indices':ids,'actuators':[meta['muscle_names'][i] for i in ids]})
  materials.append((ob,mat,ids))
 # Real skeletal meshes from the same MuJoCo model.
 bone_material=material('Anatomical ivory',(.64,.62,.53),rough=.55)
 bone_objects=[]
 for item in json.loads((OUT/'bone_meshes.json').read_text(encoding='utf-8')):
  if item['body'] not in idx:continue
  # Hands already have opaque skin. Hide their separate native skeletal layer
  # so the two retargeted anatomical sources cannot produce a second hand.
  if item['body'] in meta.get('hand_display_body_names',[]):continue
  mesh=bpy.data.meshes.new(item['name']);mesh.from_pydata(item['vertices'],[],item['faces']);mesh.update()
  ob=bpy.data.objects.new('Bone '+item['name'],mesh);bpy.context.collection.objects.link(ob);mesh.materials.append(bone_material)
  for p in mesh.polygons:p.use_smooth=True
  bone_objects.append((ob,item['body']))
 root=bpy.data.objects.new('Target stature 180cm',None);bpy.context.collection.objects.link(root)
 for ob in list(bpy.data.objects):
  if ob!=root and ob.parent is None:ob.parent=root
 rest={n:matrix(frames['rest_body_xpos'][idx[n]],frames['rest_body_xmat'][idx[n]]) for n in names if n in idx}
 bone_rest={b.name:b.matrix_local.copy() for b in rig.data.bones}
 for pb in rig.pose.bones:pb.rotation_mode='QUATERNION'
 for frame_number in range(len(frames['times'])):
  desired={}
  for pb in rig.pose.bones:
   body_name=BODY_ALIAS.get(pb.name,pb.name)
   if body_name not in idx:continue
   j=idx[body_name];now=matrix(frames['body_xpos'][frame_number,j],frames['body_xmat'][frame_number,j])
   desired[pb.name]=now@rest[body_name].inverted()@bone_rest[pb.name]
  for pb in rig.pose.bones:
   if pb.name not in desired:continue
   if pb.parent and pb.parent.name in desired:
    basis=bone_rest[pb.name].inverted()@bone_rest[pb.parent.name]@desired[pb.parent.name].inverted()@desired[pb.name]
   else:basis=bone_rest[pb.name].inverted()@desired[pb.name]
   pb.matrix_basis=basis
   for key in ['location','rotation_quaternion','scale']:pb.keyframe_insert(data_path=key,frame=frame_number+1)
  for ob,name in bone_objects:
   j=idx[name];now=matrix(frames['body_xpos'][frame_number,j],frames['body_xmat'][frame_number,j])
   ob.matrix_basis=now@rest[name].inverted();ob.rotation_mode='QUATERNION'
   for key in ['location','rotation_quaternion']:ob.keyframe_insert(data_path=key,frame=frame_number+1)
  for ob,mat,ids in materials:
   if not ids:continue
   ratio=float(frames['force_N'][frame_number,ids].sum()/f0[ids].sum())
   socket=mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'];socket.default_value=(*color(ratio),1)
   socket.keyframe_insert(data_path='default_value',frame=frame_number+1)
 root.scale=(meta['target_height_m']/meta['native_visual_height_m'],)*3
 # Standard barbell geometry, animated from the two palm stations.
 metal=material('Bar brushed steel',(.42,.48,.53),metal=.9,rough=.22)
 plate=material('Plates navy',(.045,.063,.085),metal=.65,rough=.32)
 bar=bpy.data.objects.new('100kg barbell',None);bpy.context.collection.objects.link(bar);bar.parent=root
 bar.scale=(1/root.scale[0],)*3 # Preserve physical bar/plate dimensions at every stature.
 def cylinder(name,radius,depth,x,mat):
  bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=radius,depth=depth,location=(x,0,0),rotation=(0,math.pi/2,0))
  ob=bpy.context.object;ob.name=name;ob.parent=bar;ob.data.materials.append(mat)
  bevel=ob.modifiers.new('Rounded edges','BEVEL');bevel.width=.006;bevel.segments=2
  ob.modifiers.new('Normals','WEIGHTED_NORMAL')
 cylinder('Olympic bar',.014,2.2,0,metal)
 for sign in [-1,1]:
  for k in [0,1]:cylinder('20kg plate',.225,.065,sign*(.79+.072*k),plate)
  cylinder('Collar',.035,.04,sign*.94,metal)
 for k,points in enumerate(frames['grips']):bar.location=points.mean(axis=0);bar.keyframe_insert(data_path='location',frame=k+1)
 # Dark studio with broad key/rim lighting.
 scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
 # Eevee gives a fast deterministic first preview; CLI --cycles switches if desired.
 if '--cycles' not in sys.argv:scene.render.engine='CYCLES' if '--cpu' in sys.argv else 'BLENDER_EEVEE_NEXT'
 scene.render.resolution_x=1080;scene.render.resolution_y=1240;scene.render.resolution_percentage=100
 scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.fps=30
 scene.world=bpy.data.worlds.new('Studio World');scene.world.color=(.05,.06,.075)
 world=scene.world;world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.018,.025,.038,1);world.node_tree.nodes['Background'].inputs[1].default_value=.3
 bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,.028));ground=bpy.context.object;ground.data.materials.append(material('Ground',(.015,.021,.030),rough=.65))
 def light(name,location,energy,size):
  bpy.ops.object.light_add(type='AREA',location=location);ob=bpy.context.object;ob.name=name;ob.data.energy=energy;ob.data.shape='DISK';ob.data.size=size;ob.rotation_euler=(Vector((0,.1,1.0))-ob.location).to_track_quat('-Z','Y').to_euler()
 light('Key',(-3,-4,5),700,3);light('Fill',(3,-1,3.2),220,2.5);light('Rim',(1,3,4),800,3)
 bpy.ops.object.camera_add(location=(2.7,-7,2.2));camera=bpy.context.object;camera.rotation_euler=(Vector((-.025,.12,1.16))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=2.66;scene.camera=camera
 scene.view_settings.view_transform='AgX'
 scene.frame_start=1;scene.frame_end=len(frames['times'])
 scene.render.film_transparent=False
 (OUT/'atlas_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'visual_anatomy.json').write_text(json.dumps({'visual_bulk_changes_physics':False,'settings':presentation['visual_bulk'],
  'method':'PCA cross-section enlargement, sinusoidal belly taper, long axis unchanged',
  'hidden_superficial_layers':hidden_layers,'supplemental_real_surfaces':supplement_count,
  'lat_insertion_fits':insertion_fits,
  'lower_rectus_registration':abdominal_fits,
  'supplemental_registration':'approximate torso alignment; abdominal/spinal surfaces restricted to trunk bones, latissimus insertion restricted to proximal humerus; not validated tissue deformation',
  'changed_meshes':bulk_changes},ensure_ascii=False,indent=2),encoding='utf-8')
 # Check actual skinning bone transforms before exporting a render.
 scene.frame_set(31);bpy.context.view_layer.update()
 skinning_error=0.
 for pb in rig.pose.bones:
  body_name=BODY_ALIAS.get(pb.name,pb.name)
  if body_name not in idx:continue
  j=idx[body_name];expected=matrix(frames['body_xpos'][30,j],frames['body_xmat'][30,j])@rest[body_name].inverted()@bone_rest[pb.name]
  skinning_error=max(skinning_error,(pb.matrix.translation-expected.translation).length)
 assert skinning_error<1e-4,skinning_error
 landmarks=json.loads((OUT/'anatomy_landmarks.json').read_text('utf-8'));insertion_error=0.
 attachment_frames=sorted(set([1,31,61,76]+([100,130,len(frames['times'])] if len(frames['times'])>100 else [])))
 for frame_number in attachment_frames:
  scene.frame_set(frame_number);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
  for fit in insertion_fits:
   ob=bpy.data.objects[fit['mesh']];evaluated=ob.evaluated_get(deps)
   actual=evaluated.matrix_world@evaluated.data.vertices[fit['insertion_vertex']].co
   side=1 if 'Left' in ob.name else 0;expected=Vector(landmarks['lat_insertion_frames'][frame_number-1][side])*root.scale[0]
   insertion_error=max(insertion_error,(actual-expected).length)
 assert insertion_error<1e-4,('Latissimus insertion detached',insertion_error)
 required=['trapezius','latissimus dorsi','rectus abdominis','pectoralis major','deltoid','serratus anterior','sternocleidomastoid','external oblique','internal oblique','transversus abdominis','tensor fasciae latae','gluteus','vastus','rectus femoris','biceps femoris','gastrocnemius','soleus']
 coverage={name:[ob.name for ob,_,_ in materials if name in ob.name.lower() and not ob.hide_render] for name in required}
 assert all(len(objects)>=2 for objects in coverage.values()),coverage
 hand_geometries={item['name'] for item in json.loads((OUT/'bone_meshes.json').read_text()) if item['body'] in meta.get('hand_display_body_names',[])}
 assert not any(ob.name[5:] in hand_geometries for ob in scene.objects if ob.type=='MESH' and ob.name.startswith('Bone ') and not ob.hide_render)
 (OUT/'anatomy_validation.json').write_text(json.dumps({'status':'passed','major_group_surface_objects':coverage,
  'lat_insertion_max_error_m':insertion_error,'validated_frames':attachment_frames,
  'hand_display':'opaque skin representation; native hand bone display layer omitted','separate_visible_native_hand_bone_objects':0,
  'limits':'Attachment point and major-group presence checks; not tissue deformation or anatomical expert validation'},ensure_ascii=False,indent=2),encoding='utf-8')
 scene.frame_set(31)
 (OUT/'rig_validation.json').write_text(json.dumps({'max_bone_head_error_m':skinning_error,'mapped_meshes':sum(bool(x[2]) for x in materials),'total_muscle_meshes':len(materials)},indent=2),encoding='utf-8')
 scene.render.use_persistent_data=True
 bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'fullbody_scene.blend'),compress=True)
 if '--build-only' in sys.argv:
  print('Scene prepared without rendering',flush=True)
 elif '--all' in sys.argv:
  scene.render.filepath=str(OUT/'body_frames'/'body_');(OUT/'body_frames').mkdir(exist_ok=True)
  bpy.ops.render.render(animation=True)
 else:
  scene.render.filepath=str(OUT/'body_preview.png');bpy.ops.render.render(write_still=True)
 print('REAL_ATLAS_RENDER_COMPLETE',len(materials),'mapped',sum(bool(x[2]) for x in materials),flush=True)


if __name__=='__main__':main()
