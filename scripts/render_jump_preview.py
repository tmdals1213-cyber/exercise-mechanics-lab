"""Anatomical diagnostic stills. Display geometry is independent of dynamics."""
import bpy,bmesh,json,sys,math
from pathlib import Path
import numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from torso_surfaces import add_torso_surfaces,fit_lower_rectus_depth,fit_lat_insertion
from visual_anatomy import enlarge_muscle_belly

def mat(name,color):
 m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=.55;return m

def matrix(p,r):
 M=Matrix.Identity(4);M.translation=Vector(p)
 for a in range(3):
  for b in range(3):M[a][b]=float(r[a*3+b])
 return M

def main():
 args=sys.argv[sys.argv.index('--')+1:];kind=args[0];hz=int(args[1]);dest=ROOT/f'results/jump_validation/{kind}/acceleration_{hz}hz';path=dest/'motion.npz';z=dict(np.load(path if path.exists() else dest/'partial_motion.npz'));rest=dict(np.load(dest/'reference_pose.npz'));meta=json.loads((dest/'setup.json').read_text());idx={n:i for i,n in enumerate(meta['body_names'])};scale=1.8/1.788053
 bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(ROOT/'assets/fullbody/fullbody_muscles_skin.glb'));rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');add_torso_surfaces(ROOT,rig)
 skin=bpy.data.objects.get('Skin');skinmat=mat('Skin ends',(.43,.48,.50));musclemat=mat('Anatomical muscle neutral',(.40,.22,.16));bonemat=mat('Bone ivory',(.62,.62,.54))
 if skin:
  keep=[]
  for v in skin.data.vertices:
   weights={skin.vertex_groups[g.group].name:g.weight for g in v.groups};world=skin.matrix_world@v.co
   face=sum(w for n,w in weights.items() if n=='head')>.35 and (world.z>1.64 or (world.z>1.58 and world.y<.19));ends=sum(w for n,w in weights.items() if any(k in n for k in ['ph_','proxph','midph','distph','thumb','firstmc','lunate','calcn','toes','talus']))>.65;keep.append(face or ends)
  bm=bmesh.new();bm.from_mesh(skin.data);bm.verts.ensure_lookup_table();bmesh.ops.delete(bm,geom=[v for v in bm.verts if not keep[v.index]],context='VERTS');bm.to_mesh(skin.data);bm.free();skin.data.materials.clear();skin.data.materials.append(skinmat)
 bulk=json.loads((ROOT/'scenarios/ohp_fullbody.json').read_text())['presentation']['visual_bulk']
 for ob in list(bpy.data.objects):
  if ob.type!='MESH' or ob==skin:continue
  if not any(m.type=='ARMATURE' for m in ob.modifiers) or 'platysma' in ob.name.lower():ob.hide_render=True;continue
  fit_lower_rectus_depth(ob);enlarge_muscle_belly(ob,bulk);fit_lat_insertion(ob,ROOT,ROOT/'results/ohp_fullbody');ob.data.materials.clear();ob.data.materials.append(musclemat)
  for p in ob.data.polygons:p.use_smooth=True
 oldmeta=json.loads((ROOT/'results/ohp_fullbody/fullbody_metadata.json').read_text());bones=[]
 for item in json.loads((ROOT/'results/ohp_fullbody/bone_meshes.json').read_text()):
  if item['body'] not in idx or item['body'] in oldmeta.get('hand_display_body_names',[]):continue
  mesh=bpy.data.meshes.new(item['name']);mesh.from_pydata(item['vertices'],[],item['faces']);mesh.materials.append(bonemat);mesh.update();ob=bpy.data.objects.new('Bone '+item['name'],mesh);bpy.context.collection.objects.link(ob);bones.append((ob,item['body']))
 root=bpy.data.objects.new('Physical stature',None);bpy.context.collection.objects.link(root)
 for ob in list(bpy.data.objects):
  if ob!=root and ob.parent is None:ob.parent=root
 root.scale=(scale,)*3
 restbody={n:matrix(rest['body_xpos'][j]/scale,rest['body_xmat'][j]) for n,j in idx.items()};bonerest={b.name:b.matrix_local.copy() for b in rig.data.bones}
 for pb in rig.pose.bones:pb.rotation_mode='QUATERNION'
 def pose(t):
  i=int(np.argmin(abs(z['times']-t)));desired={}
  for pb in rig.pose.bones:
   name='sacrum' if pb.name=='pelvis' else pb.name
   if name in idx:
    j=idx[name];desired[pb.name]=matrix(z['body_xpos'][i,j]/scale,z['body_xmat'][i,j])@restbody[name].inverted()@bonerest[pb.name]
  for pb in rig.pose.bones:
   if pb.name not in desired:continue
   if pb.parent and pb.parent.name in desired:pb.matrix_basis=bonerest[pb.name].inverted()@bonerest[pb.parent.name]@desired[pb.parent.name].inverted()@desired[pb.name]
   else:pb.matrix_basis=bonerest[pb.name].inverted()@desired[pb.name]
  for ob,name in bones:
   j=idx[name];ob.matrix_basis=matrix(z['body_xpos'][i,j]/scale,z['body_xmat'][i,j])@restbody[name].inverted()
  bpy.context.view_layer.update();return i
 scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.eevee.taa_render_samples=24;scene.render.resolution_x=720;scene.render.resolution_y=1040;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
 world=bpy.data.worlds.new('Studio');scene.world=world;world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.022,.032,.05,1);world.node_tree.nodes['Background'].inputs[1].default_value=.35
 scene.view_settings.view_transform='AgX'
 def aim(ob,target):ob.rotation_euler=(Vector(target)-ob.location).to_track_quat('-Z','Y').to_euler()
 for loc,power,size in [((3,-4,5),1500,4),((-3,1,3),1200,3),((0,4,5),1800,3)]:
  data=bpy.data.lights.new('Softbox','AREA');data.energy=power;data.shape='DISK';data.size=size;ob=bpy.data.objects.new('Softbox',data);bpy.context.collection.objects.link(ob);ob.location=loc;aim(ob,(0,0,1))
 bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,.025*scale));ground=bpy.context.object;ground.name='Ground';ground.data.materials.append(mat('Ground',(.045,.061,.08)))
 camdata=bpy.data.cameras.new('Front oblique');camera=bpy.data.objects.new('Front oblique',camdata);bpy.context.collection.objects.link(camera);scene.camera=camera;camdata.type='ORTHO';camdata.ortho_scale=2.9
 snapshots=[('ready',0),('push',.92),('apex',meta['takeoff_s']+.25),('landing',meta['landing_s']+.07)]
 output=dest/'preview';output.mkdir(exist_ok=True)
 for label,t in snapshots:
  i=pose(t);center=Vector((float(z['com'][i,0]),float(z['com'][i,1]),1.25));yaw=math.radians(25);elev=math.radians(15);direction=Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)));camera.location=center+direction*6;aim(camera,center);scene.render.filepath=str(output/(label+'.png'));bpy.ops.render.render(write_still=True)
  print('JUMP_PREVIEW',kind,label,flush=True)
 (output/'manifest.json').write_text(json.dumps({'kind':kind,'poses':snapshots,'muscle_colors':'anatomy only; no force/activation heatmap','visual_bulk_is_physics':False,'native_bone_geometry':'existing neutral anatomical export; rendering diagnostic only'},indent=2))

if __name__=='__main__':main()
