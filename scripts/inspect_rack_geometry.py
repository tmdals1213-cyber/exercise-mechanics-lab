"""Repose a previous display scene for rapid rack geometry inspection."""
from pathlib import Path
import json,bpy,numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/grip_comparison'
def matrix(p,r):
 M=Matrix.Identity(4);M.translation=Vector(p)
 for i in range(3):
  for j in range(3):M[i][j]=float(r[i*3+j])
 return M
def distance_to_axis(v,faces,bar):
 tri=v[np.array(faces)];p=tri[:,:,1:]-bar[None,None,1:]
 cross=lambda a,b:a[:,0]*b[:,1]-a[:,1]*b[:,0]
 area=cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);c=np.stack([cross(p[:,0],p[:,1]),cross(p[:,1],p[:,2]),cross(p[:,2],p[:,0])],axis=1)
 if np.any((abs(area)>1e-12)&((c.min(1)>=-1e-12)|(c.max(1)<=1e-12))):return -.014
 best=1e9
 for a,b in [(p[:,0],p[:,1]),(p[:,1],p[:,2]),(p[:,2],p[:,0])]:
  e=b-a;t=np.clip(-(a*e).sum(1)/np.maximum((e*e).sum(1),1e-20),0,1);best=min(best,float(np.linalg.norm(a+t[:,None]*e,axis=1).min()))
 return best-.014
reports=[]
for grip in [65,80]:
 case=OUT/f'grip_{grip}';pose=json.loads((case/'rack_candidate_pose.json').read_text());old=np.load(case/'cycle/physiology/fullbody_motion.npz')
 names=json.loads((case/'cycle/physiology/fullbody_metadata.json').read_text())['body_names'];idx={n:i for i,n in enumerate(names)}
 bpy.ops.wm.open_mainfile(filepath=str(case/'cycle/physiology/fullbody_scene.blend'));scene=bpy.context.scene;scene.frame_set(1)
 for ob in scene.objects:ob.animation_data_clear()
 geom_body={item['name']:item['body'] for item in json.loads((case/'bone_meshes.json').read_text())}
 rig=next(o for o in scene.objects if o.type=='ARMATURE');rest={n:matrix(old['rest_body_xpos'][i],old['rest_body_xmat'][i]) for n,i in idx.items()};br={b.name:b.matrix_local.copy() for b in rig.data.bones}
 desired={}
 for pb in rig.pose.bones:
  name='sacrum' if pb.name=='pelvis' else pb.name
  if name in idx:
   i=idx[name];desired[pb.name]=matrix(pose['body_xpos'][i],pose['body_xmat'][i])@rest[name].inverted()@br[pb.name]
 for pb in rig.pose.bones:
  if pb.name not in desired:continue
  pb.matrix_basis=br[pb.name].inverted()@br[pb.parent.name]@desired[pb.parent.name].inverted()@desired[pb.name] if pb.parent and pb.parent.name in desired else br[pb.name].inverted()@desired[pb.name]
 for ob in scene.objects:
  if not ob.name.startswith('Bone '):continue
  body=geom_body.get(ob.name[5:])
  if body in pose.get('hand_body_names',[]):ob.hide_render=True
  if body in idx:
   i=idx[body];ob.matrix_basis=matrix(pose['body_xpos'][i],pose['body_xmat'][i])@rest[body].inverted()
 bar=bpy.data.objects['100kg barbell'];bar.location=Vector(pose['bar_native']);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get();origin=np.array(bar.matrix_world.translation)
 distances={}
 for ob in scene.objects:
  if ob.type!='MESH' or ob.hide_render or not any(k in ob.name.lower() for k in ['pectoralis major','deltoid','bone clavicle']):continue
  ev=ob.evaluated_get(deps);vertices=np.array([list(ev.matrix_world@v.co) for v in ev.data.vertices]);ev.data.calc_loop_triangles()
  faces=[list(p.vertices) for p in ev.data.loop_triangles];distances[ob.name]=distance_to_axis(vertices,faces,origin)
 reports.append({'grip_cm':grip,'bar_world_m':origin.tolist(),'surface_gap_m':distances})
 # A close oblique and a side diagnostic, with plates hidden to reveal contact.
 for ob in scene.objects:
  if ob.parent==bar and ob.name!='Olympic bar':ob.hide_render=True
 for view,direction in [('oblique',Vector((.65,-1,.15))),('side',Vector((1,0,0)))]:
  camera=scene.camera;target=Vector((-.025,.14,1.46));camera.location=target+direction.normalized()*4;camera.rotation_euler=(-direction).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=.95
  scene.render.resolution_x=800;scene.render.resolution_y=800;scene.render.resolution_percentage=100;scene.cycles.samples=24
  scene.render.filepath=str(OUT/f'rack_{grip}_{view}.png');bpy.ops.render.render(write_still=True)
(OUT/'rack_candidate_geometry.json').write_text(json.dumps(reports,indent=2),encoding='utf-8');print('RACK_GEOMETRY_INSPECTED',flush=True)
