"""Matched camera and scale for 65/80 cm full-body comparison panels."""
import bpy,json,math,sys
from pathlib import Path
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/grip_comparison'
CYCLES='--full-cycle' in sys.argv
PHYSIOLOGY='--physiology' in sys.argv or CYCLES
CASE_ROOTS={g:OUT/f'grip_{g}'/('cycle' if CYCLES else '')/('physiology' if PHYSIOLOGY else '') for g in [65,80]}
SIZE=(492,640);yaw=math.radians(25);elevation=math.radians(15)
DIRECTION=Vector((math.sin(yaw)*math.cos(elevation),-math.cos(yaw)*math.cos(elevation),math.sin(elevation)))

def main():
 points=[]
 for grip in [65,80]:
  bpy.ops.wm.open_mainfile(filepath=str(CASE_ROOTS[grip]/'fullbody_scene.blend'))
  scene=bpy.context.scene
  anatomy=[o for o in scene.objects if o.type=='MESH' and not o.hide_render and (o.name.startswith('Bone ') or o.name=='Skin' or any(m.type=='ARMATURE' for m in o.modifiers) or (o.parent and o.parent.name=='100kg barbell'))]
  count=scene.frame_end
  for frame in sorted(set([1,16,31,46,61,76,count])):
   scene.frame_set(frame);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
   for ob in anatomy:points.extend(ob.evaluated_get(deps).matrix_world@Vector(p) for p in ob.evaluated_get(deps).bound_box)
 rotation=(-DIRECTION).to_track_quat('-Z','Y').to_matrix();inverse=rotation.transposed();target=Vector((-.025,.12,1.16))
 local=np.array([list(inverse@(p-target)) for p in points]);lo=local.min(0);hi=local.max(0)
 offset=rotation@Vector(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,0));target+=offset
 scale=max(hi[1]-lo[1],(hi[0]-lo[0])/(SIZE[0]/SIZE[1]))*1.065
 for grip in [65,80]:
  bpy.ops.wm.open_mainfile(filepath=str(CASE_ROOTS[grip]/'fullbody_scene.blend'))
  scene=bpy.context.scene;data=bpy.data.cameras.new('Matched comparison camera');camera=bpy.data.objects.new('Matched comparison camera',data);scene.collection.objects.link(camera)
  camera.location=target+DIRECTION*8;camera.rotation_euler=(-DIRECTION).to_track_quat('-Z','Y').to_euler();data.type='ORTHO';data.ortho_scale=scale
  scene.camera=camera;scene.render.resolution_x=SIZE[0];scene.render.resolution_y=SIZE[1];scene.render.resolution_percentage=100
  scene.render.image_settings.file_format='PNG';scene.render.use_persistent_data=True
  directory=CASE_ROOTS[grip]/'comparison_frames';directory.mkdir(exist_ok=True)
  if '--preview' in sys.argv:
   scene.frame_set(31);scene.render.filepath=str(directory/'preview.png');bpy.ops.render.render(write_still=True)
  else:
   scene.frame_start=1;scene.render.filepath=str(directory/'frame_');bpy.ops.render.render(animation=True)
  print('MATCHED_GRIP_RENDER_COMPLETE',grip,flush=True)
 (OUT/('cycle_camera.json' if CYCLES else 'physiology_camera.json' if PHYSIOLOGY else 'camera.json')).write_text(json.dumps({'size':SIZE,'location':list(target+DIRECTION*8),'direction':list(DIRECTION),'ortho_scale':scale,'shared_for_both_cases':True,'backend':'active_passive_gh' if PHYSIOLOGY else 'legacy_minimum_torque','fit':'union of both cases across sampled poses, 6.5% margin'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
