"""Render synchronized front-isometric, front, side and rear anatomical views."""
import bpy,json,sys,math
from pathlib import Path
from mathutils import Vector
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ohp_fullbody';MV=OUT/'multiview'
PRESENTATION=json.loads((ROOT/'scenarios/ohp_fullbody.json').read_text(encoding='utf-8'))['presentation']
yaw=math.radians(PRESENTATION['main_yaw_deg']);elevation=math.radians(PRESENTATION['main_elevation_deg'])
VIEWS={'iso':{'direction':(math.sin(yaw)*math.cos(elevation),-math.cos(yaw)*math.cos(elevation),math.sin(elevation)),'size':(660,1116),'label':'전면 ISO'},
       'front':{'direction':(0,-1,0),'size':(262,320),'label':'정면'},
       'side':{'direction':(1,0,0),'size':(262,320),'label':'측면'},
       'back':{'direction':(0,1,0),'size':(262,320),'label':'후면'}}

def main():
 bpy.ops.wm.open_mainfile(filepath=str(OUT/'fullbody_scene.blend'))
 scene=bpy.context.scene;MV.mkdir(exist_ok=True)
 data=np.load(OUT/'fullbody_motion.npz');root=bpy.data.objects['Target stature 180cm']
 path=bpy.data.curves.new('Prescribed bar midpoint path','CURVE');path.dimensions='3D';path.bevel_depth=.004;path.bevel_resolution=3
 line=path.splines.new('POLY');points_native=data['grips'].mean(axis=1);line.points.add(len(points_native)-1)
 for point,co in zip(line.points,points_native):point.co=(*co,1)
 guide=bpy.data.objects.new('Side view bar trajectory',path);scene.collection.objects.link(guide);guide.parent=root
 mat=bpy.data.materials.new('Trajectory teal');mat.diffuse_color=(.08,.85,.70,1);mat.use_nodes=True
 shader=mat.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(.08,.85,.70,1)
 shader.inputs['Emission Color'].default_value=(.08,.85,.70,1);shader.inputs['Emission Strength'].default_value=1
 path.materials.append(mat)
 anatomy=[o for o in scene.objects if o.type=='MESH' and not o.hide_render and (
  o.name.startswith('Bone ') or o.name=='Skin' or any(m.type=='ARMATURE' for m in o.modifiers) or (o.parent and o.parent.name=='100kg barbell'))]
 points=[]
 for frame in [1,16,31,46,61,76]:
  scene.frame_set(frame);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
  for ob in anatomy:
   evaluated=ob.evaluated_get(deps)
   points.extend(evaluated.matrix_world@Vector(corner) for corner in evaluated.bound_box)
 selected=sys.argv[sys.argv.index('--view')+1] if '--view' in sys.argv else None
 manifest=json.loads((MV/'views.json').read_text(encoding='utf-8')) if selected and (MV/'views.json').exists() else {'frame_count':76,'fps':30,'same_physics_and_colour_data':True,'views':{}}
 for name,settings in VIEWS.items():
  if selected and name!=selected:continue
  # Side plates would hide the head and hand alignment in an end-on view.
  # Their physics/load remains unchanged; only their auxiliary-view visibility changes.
  for ob in scene.objects:
   if ob.name.startswith(('20kg plate','Collar')):ob.hide_render=(name=='side')
  guide.hide_render=(name!='side')
  camera_data=bpy.data.cameras.new('Camera '+name);camera=bpy.data.objects.new('Camera '+name,camera_data);scene.collection.objects.link(camera)
  target=Vector((-.025,.12,1.16));direction=Vector(settings['direction']).normalized()
  camera.location=target+direction*8;camera.rotation_euler=(-direction).to_track_quat('-Z','Y').to_euler()
  rotation=camera.rotation_euler.to_matrix();inverse=rotation.transposed()
  local=np.array([list(inverse@(p-target)) for p in points]);lo=local.min(axis=0);hi=local.max(axis=0)
  centre=Vector(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,0));offset=rotation@centre
  camera.location+=offset;target+=offset
  aspect=settings['size'][0]/settings['size'][1]
  camera_data.type='ORTHO';camera_data.ortho_scale=max(hi[1]-lo[1],(hi[0]-lo[0])/aspect)*1.065
  camera_data.clip_start=.01;camera_data.clip_end=100
  scene.camera=camera;scene.render.resolution_x=settings['size'][0];scene.render.resolution_y=settings['size'][1]
  scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.use_persistent_data=True
  directory=MV/name;directory.mkdir(exist_ok=True)
  manifest['views'][name]={'label':settings['label'],'size':list(settings['size']),'ortho_scale':camera_data.ortho_scale,'direction':list(direction),'sampled_bounds_padding_fraction':.065,'plates_hidden_for_visibility':name=='side','trajectory_guide':name=='side'}
  scene.render.filepath=str(directory/'frame_')
  if '--preview' in sys.argv:
   scene.frame_set(31);scene.render.filepath=str(directory/'preview.png');bpy.ops.render.render(write_still=True)
  else:
   scene.frame_start=1;scene.frame_end=76;bpy.ops.render.render(animation=True)
  print('VIEW_COMPLETE',name,flush=True)
 (MV/'views.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 scene.camera=bpy.data.objects.get('Camera iso');scene.render.resolution_x=660;scene.render.resolution_y=1116
 guide.hide_render=True
 if not selected:bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'fullbody_multiview_scene.blend'),compress=True)
 print('SYNCHRONIZED_MULTIVIEW_COMPLETE',flush=True)

if __name__=='__main__':main()
