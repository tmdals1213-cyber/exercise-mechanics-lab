"""Neutral-colour front/rear anatomical QA at rest, start, mid-lift and lockout."""
import bpy,json
from pathlib import Path
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ohp_fullbody';QA=OUT/'anatomy_check';QA.mkdir(exist_ok=True)
landmarks=json.loads((OUT/'anatomy_landmarks.json').read_text('utf-8'))
manifest={}
for phase,frame in [('rest',None),('start',1),('middle',31),('lockout',61)]:
 bpy.ops.wm.open_mainfile(filepath=str(OUT/'fullbody_scene.blend'))
 scene=bpy.context.scene;scene.frame_set(frame or 1)
 if frame is None:
  for ob in scene.objects:
   if ob.type=='ARMATURE':
    ob.animation_data_clear()
    for bone in ob.pose.bones:bone.matrix_basis=Matrix.Identity(4)
   if ob.name.startswith('Bone '):ob.animation_data_clear();ob.matrix_basis=Matrix.Identity(4)
 for ob in scene.objects:
  if ob.parent and ob.parent.name=='100kg barbell':ob.hide_render=True
 for mat in bpy.data.materials:
  if mat.name.startswith('Heat '):
   mat.node_tree.animation_data_clear();mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.44,.18,.14,1)
 bpy.context.view_layer.update()
 scale=bpy.data.objects['Target stature 180cm'].scale[0]
 for view,direction in [('front',(0,-1,0)),('back',(0,1,0))]:
  target=Vector((-.025,.15,1.10));direction=Vector(direction)
  data=bpy.data.cameras.new('QA camera');camera=bpy.data.objects.new('QA camera',data);scene.collection.objects.link(camera)
  camera.location=target+direction*8;camera.rotation_euler=(-direction).to_track_quat('-Z','Y').to_euler()
  data.type='ORTHO';data.ortho_scale=2.28;scene.camera=camera
  scene.render.resolution_x=500;scene.render.resolution_y=700;scene.render.resolution_percentage=100
  bpy.context.view_layer.update()
  projected=[]
  for point in landmarks['lat_insertion_rest'] if frame is None else landmarks['lat_insertion_frames'][frame-1]:
   uv=world_to_camera_view(scene,camera,Vector(point)*scale);projected.append([uv.x*500,(1-uv.y)*700])
  scene.render.filepath=str(QA/f'{view}_{phase}.png');bpy.ops.render.render(write_still=True)
  manifest[f'{view}_{phase}']={'lat_insertion_projected_px':projected,'size':[500,700]}
(QA/'views.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('ANATOMY_CHECKS_RENDERED',flush=True)
