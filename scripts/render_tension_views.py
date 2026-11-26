"""Paired anatomy / display-only mechanical proxy, plus a shared region mask.

The shaded geometry is independent of the force solver. Object mask IDs point
to the same group/side IDs for both rigs; the viewer supplies the solved values.
"""
import bpy,bmesh,sys,json,math
from pathlib import Path
import numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from torso_surfaces import add_torso_surfaces,fit_lower_rectus_depth,fit_lat_insertion
from visual_anatomy import enlarge_muscle_belly
from atlas_mapping import map_atlas
from smooth_humanoid import copy_surface,finish_surfaces,restore_regions
from tension_groups import MASK_STEP
OUT=ROOT/'results/tension_visualization'

def matrix(p,r):
    M=Matrix.Identity(4);M.translation=Vector(p)
    for i in range(3):
        for j in range(3):M[i][j]=float(r[i*3+j])
    return M

def material(name,color,rough=.6):
    m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=rough;return m

def linear(v):return v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4

def tag(ob,gid):
    ob['tension_group_id']=gid
    if gid:
        c=gid*MASK_STEP;ob.color=(linear(c/255),linear((255-c)/255),linear(128/255),1.)
    else:ob.color=(0,0,0,1)

def main():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [];meta=OUT/'data.json';data=json.loads((meta if meta.exists() else OUT/'scene_metadata.json').read_text(encoding='utf-8'));z=dict(np.load(OUT/'geometry.npz'));groups=data['groups'];scale=data['scale'];idx={n:i for i,n in enumerate(data['body_names'])};names=json.loads((ROOT/'results/tension_reference/jump/summary.json').read_text(encoding='utf-8'))['names'];node_names=data['node_names']
    if '--neutral' in args:
        neutral=np.load(OUT/'neutral_geometry.npz')
        z['body_xpos'][0]=neutral['body_xpos'];z['body_xmat'][0]=neutral['body_xmat']
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(ROOT/'assets/fullbody/fullbody_muscles_skin.glb'));rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');add_torso_surfaces(ROOT,rig)
    white=material('Mapped tissue shading',(.72,.75,.78));unknown=material('Unmapped or review',(.30,.34,.39));skinmat=material('Face hands feet',(.42,.47,.52));ivory=material('Skeleton',(.29,.34,.40));proxywhite=material('Tension proxy shading',(.85,.87,.89));bonegray=material('Mechanical segments',(.18,.23,.29))
    skin=bpy.data.objects.get('Skin')
    smooth,region_counts=copy_surface(skin,groups,names,map_atlas)
    if skin:
        keep=[]
        for v in skin.data.vertices:
            weights={skin.vertex_groups[g.group].name:g.weight for g in v.groups};world=skin.matrix_world@v.co
            face=sum(w for n,w in weights.items() if n=='head')>.35 and (world.z>1.64 or (world.z>1.58 and world.y<.19));ends=sum(w for n,w in weights.items() if any(k in n for k in ['ph_','proxph','midph','distph','thumb','firstmc','lunate','calcn','toes','talus']))>.65;keep.append(face or ends)
        bm=bmesh.new();bm.from_mesh(skin.data);bm.verts.ensure_lookup_table();bmesh.ops.delete(bm,geom=[v for v in bm.verts if not keep[v.index]],context='VERTS');bm.to_mesh(skin.data);bm.free();skin.data.materials.clear();skin.data.materials.append(skinmat);tag(skin,0)
    bulk=json.loads((ROOT/'scenarios/ohp_fullbody.json').read_text(encoding='utf-8'))['presentation']['visual_bulk'];mapping=[]
    for ob in list(bpy.data.objects):
        if ob.type!='MESH' or ob in [skin,smooth]:continue
        if not any(m.type=='ARMATURE' for m in ob.modifiers) or 'platysma' in ob.name.lower():ob.hide_render=True;continue
        fit_lower_rectus_depth(ob);enlarge_muscle_belly(ob,bulk);fit_lat_insertion(ob,ROOT,ROOT/'results/ohp_fullbody');ids=map_atlas(ob.name,names);group=next((g for g in groups if ids and set(ids).issubset(g['indices'])),None);gid=group['display_id'] if group else 0;tag(ob,gid);ob.data.materials.clear();ob.data.materials.append(white if gid else unknown)
        for p in ob.data.polygons:p.use_smooth=True
        mapping.append({'mesh':ob.name,'group_id':gid,'group_key':group['key'] if group else None,'side':group['side'] if group else None,'status':group['status'] if group else 'unmapped'})
    oldmeta=json.loads((ROOT/'results/ohp_fullbody/fullbody_metadata.json').read_text(encoding='utf-8'));bones=[]
    for item in json.loads((ROOT/'results/ohp_fullbody/bone_meshes.json').read_text(encoding='utf-8')):
        if item['body'] not in idx or item['body'] in oldmeta.get('hand_display_body_names',[]):continue
        mesh=bpy.data.meshes.new(item['name']);mesh.from_pydata(item['vertices'],[],item['faces']);mesh.materials.append(ivory);mesh.update();ob=bpy.data.objects.new('Bone '+item['name'],mesh);bpy.context.collection.objects.link(ob);tag(ob,0);bones.append((ob,item['body']))
    root=bpy.data.objects.new('Anatomical display',None);bpy.context.collection.objects.link(root)
    for ob in list(bpy.data.objects):
        if ob!=root and ob.parent is None:ob.parent=root
    root.scale=(scale,)*3;yaw=math.radians(25);right=Vector((math.cos(yaw),math.sin(yaw),0));root.location=-.72*right
    rest={n:matrix(z['rest_body_xpos'][j]/scale,z['rest_body_xmat'][j]) for n,j in idx.items()};bonerest={bone.name:bone.matrix_local.copy() for bone in rig.data.bones}
    for pb in rig.pose.bones:pb.rotation_mode='QUATERNION'
    simple=bpy.data.objects.new('Mechanical display only',None);bpy.context.collection.objects.link(simple);simple.location=.72*right
    segments=[];joints=[];curves=[];shells=[]
    shellmat=material('Rigid segment envelope',(.32,.42,.52));shellmat.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value=.16
    if hasattr(shellmat,'surface_render_method'):shellmat.surface_render_method='BLENDED'
    def envelope(name,a,b,radius):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,radius=1);ob=bpy.context.object;ob.name='Envelope '+name;ob.parent=simple;ob.data.materials.append(shellmat);tag(ob,0)
        for p in ob.data.polygons:p.use_smooth=True
        shells.append((ob,a,b,radius));return ob
    envelope('torso',node_names.index('pelvis'),node_names.index('neck'),(.155,.09))
    for side in ['r','l']:
        for a,b,radius in [('shoulder','elbow',(.052,.048)),('elbow','wrist',(.041,.037)),('hip','knee',(.073,.065)),('knee','ankle',(.05,.043))]:envelope(a+' '+side,node_names.index(a+'_'+side),node_names.index(b+'_'+side),radius)
    pelvis=envelope('pelvis',node_names.index('hip_r'),node_names.index('hip_l'),(.09,.075))
    for a,b in data['links']:
        bpy.ops.mesh.primitive_cylinder_add(vertices=20,radius=1,depth=2);ob=bpy.context.object;ob.name='Segment '+node_names[a]+' '+node_names[b];ob.parent=simple;ob.data.materials.append(bonegray);tag(ob,0)
        for p in ob.data.polygons:p.use_smooth=True
        segments.append((ob,a,b))
    for name in node_names:
        if name=='head':continue
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,radius=.025);ob=bpy.context.object;ob.name='Joint '+name;ob.parent=simple;ob.data.materials.append(bonegray);tag(ob,0)
        for p in ob.data.polygons:p.use_smooth=True
        joints.append((ob,node_names.index(name)))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=28,ring_count=18,radius=1);head=bpy.context.object;head.parent=simple;head.scale=(.073,.065,.095);head.data.materials.append(skinmat);tag(head,0)
    for p in head.data.polygons:p.use_smooth=True
    for group in groups:
        curve=bpy.data.curves.new('Path '+group['key']+' '+group['side'],'CURVE');curve.dimensions='3D';curve.resolution_u=1;curve.bevel_depth=.019 if group['key'] not in ['rotator_cuff','forearm'] else .012;curve.bevel_resolution=3;curve.use_fill_caps=True;spline=curve.splines.new('POLY');spline.points.add(7);ob=bpy.data.objects.new(curve.name,curve);bpy.context.collection.objects.link(ob);ob.parent=simple;ob.data.materials.append(proxywhite if group['display_id'] else unknown);tag(ob,group['display_id']);curves.append(ob)
    pivot=Vector(z['nodes'][0,node_names.index('pelvis')]);pivot.z=0
    surfaces,surface_masks,update_surface_pose=finish_surfaces(smooth,rig,scale,right,len(groups),material,linear,pivot)
    def pose(i):
        desired={}
        for pb in rig.pose.bones:
            name='sacrum' if pb.name=='pelvis' else pb.name
            if name in idx:
                j=idx[name];desired[pb.name]=matrix(z['body_xpos'][i,j]/scale,z['body_xmat'][i,j])@rest[name].inverted()@bonerest[pb.name]
        for pb in rig.pose.bones:
            if pb.name not in desired:continue
            if pb.parent and pb.parent.name in desired:pb.matrix_basis=bonerest[pb.name].inverted()@bonerest[pb.parent.name]@desired[pb.parent.name].inverted()@desired[pb.name]
            else:pb.matrix_basis=bonerest[pb.name].inverted()@desired[pb.name]
        for ob,name in bones:
            j=idx[name];ob.matrix_basis=matrix(z['body_xpos'][i,j]/scale,z['body_xmat'][i,j])@rest[name].inverted()
        nodes=z['nodes'][i]
        for ob,a,b in segments:
            p=Vector(nodes[a]);v=Vector(nodes[b])-p;ob.location=p+v/2;ob.rotation_mode='QUATERNION';ob.rotation_quaternion=v.to_track_quat('Z','Y');r=.026 if a<3 and b<4 else .017;ob.scale=(r,r,max(v.length/2,.001))
        for ob,j in joints:ob.location=nodes[j]
        for ob,a,b,radius in shells:
            p=Vector(nodes[a]);v=Vector(nodes[b])-p;ob.location=p+v/2;ob.rotation_mode='QUATERNION';ob.rotation_quaternion=v.to_track_quat('Z','Y');ob.scale=(radius[0],radius[1],v.length/2+.025)
        head.location=nodes[node_names.index('head')];head.rotation_mode='QUATERNION';head.rotation_quaternion=(Vector(nodes[node_names.index('head')])-Vector(nodes[node_names.index('neck')])).to_track_quat('Z','Y')
        for g,ob in enumerate(curves):
            for point,co in zip(ob.data.splines[0].points,z['paths'][i,g]):point.co=(*co,1)
        update_surface_pose();bpy.context.view_layer.update()
    scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.eevee.taa_render_samples=16;scene.render.resolution_x=1080;scene.render.resolution_y=900;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.compression=25;scene.render.film_transparent=False
    world=bpy.data.worlds.new('Studio');scene.world=world;world.use_nodes=True;background=world.node_tree.nodes['Background'];background.inputs[0].default_value=(.018,.025,.038,1);background.inputs[1].default_value=.35
    def aim(ob,target):ob.rotation_euler=(Vector(target)-ob.location).to_track_quat('-Z','Y').to_euler()
    for location,power,size in [((3,-4,5),700,4),((-3,-1,3),500,3),((0,4,4),900,3)]:
        light=bpy.data.lights.new('Softbox','AREA');light.energy=power;light.shape='DISK';light.size=size;ob=bpy.data.objects.new('Softbox',light);bpy.context.collection.objects.link(ob);ob.location=location;aim(ob,(0,0,1))
    bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,.025*scale));ground=bpy.context.object;ground.data.materials.append(material('Ground',(.018,.026,.038)));tag(ground,0)
    ground.hide_render=True;scene.render.image_settings.color_mode='RGBA'
    camdata=bpy.data.cameras.new('Shared front oblique');camera=bpy.data.objects.new('Shared front oblique',camdata);bpy.context.collection.objects.link(camera);scene.camera=camera;camdata.type='ORTHO';camdata.ortho_scale=3.05;elev=math.radians(15);center=Vector((-.025,.04,1.17));camera.location=center+Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)))*7;aim(camera,center)
    mask=bpy.data.materials.new('Shared numerical group mask');mask.use_nodes=True;nt=mask.node_tree;nt.nodes.clear();info=nt.nodes.new('ShaderNodeObjectInfo');em=nt.nodes.new('ShaderNodeEmission');output=nt.nodes.new('ShaderNodeOutputMaterial');nt.links.new(info.outputs['Color'],em.inputs['Color']);nt.links.new(em.outputs[0],output.inputs['Surface'])
    for folder in ['base','mask']:(OUT/folder).mkdir(exist_ok=True)
    original_materials={ob.name:list(ob.data.materials) for ob in bpy.data.objects if ob.type in ['MESH','CURVE']}
    selected=list(range(len(z['times'])))
    if '--neutral' in args:selected=[0]
    if '--poster' in args:selected=[int(np.argmin(abs(z['times']-s))) for s in [0.,.9,1.24]]
    if '--start' in args:selected=[i for i in selected if i>=int(args[args.index('--start')+1])]
    if '--end' in args:selected=[i for i in selected if i<int(args[args.index('--end')+1])]
    (OUT/'render_mapping.json').write_text(json.dumps({'display':'opaque continuous humanoid skin','views':['front_oblique','back_oblique'],'skin_faces_per_region':region_counts,'surface_mapping':'nearest rest-space anatomical surface; label transfer only, not skin stress','visual_bulk_applied_to_skin':False,'display_only_pose_changes':False,'meshes':mapping,'mask_encoding':'sRGB code R=7*group_id, G=255-R, B=128; black=unknown/background','same_values_both_rigs':True,'camera_yaw_deg':25,'camera_elevation_deg':15,'display_geometry_is_physics':False},indent=2),encoding='utf-8')
    for i in selected:
        pose(i);scene.view_layers[0].material_override=None
        for name,mats in original_materials.items():
            ob=bpy.data.objects[name];ob.data.materials.clear()
            for mat in mats:ob.data.materials.append(mat)
        restore_regions(surfaces)
        scene.render.film_transparent=True;scene.view_settings.view_transform='AgX';background.inputs[0].default_value=(.018,.025,.038,1);background.inputs[1].default_value=.35;scene.render.filepath=str(OUT/'neutral_pose.png' if '--neutral' in args else OUT/f'base/frame_{i:04d}.png');bpy.ops.render.render(write_still=True)
        if '--neutral' in args:continue
        if '--base-only' in args:
            print('PAIRED_BASE_FRAME',i,flush=True);continue
        for ob,_,_,_ in shells:ob.hide_render=True
        for name in original_materials:
            ob=bpy.data.objects[name];ob.data.materials.clear()
            for mat in (surface_masks if ob in surfaces else [mask]):ob.data.materials.append(mat)
        restore_regions(surfaces)
        scene.render.film_transparent=False;scene.view_settings.view_transform='Standard';background.inputs[0].default_value=(0,0,0,1);background.inputs[1].default_value=0;scene.render.filepath=str(OUT/f'mask/frame_{i:04d}.png');bpy.ops.render.render(write_still=True);print('PAIRED_FRAME',i,flush=True)
        for ob,_,_,_ in shells:ob.hide_render=True

if __name__=='__main__':main()
