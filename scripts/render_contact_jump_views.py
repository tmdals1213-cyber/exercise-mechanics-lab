"""Paired front/back anatomy and retained opaque humanoid, one native pose."""
import bpy,bmesh,sys,json,math,os
from pathlib import Path
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from torso_surfaces import add_torso_surfaces,fit_lower_rectus_depth,fit_lat_insertion
from atlas_mapping import map_atlas
from smooth_humanoid import copy_surface,finish_surfaces,restore_regions
from tension_groups import MASK_STEP
OUT=ROOT/os.environ.get('EXERCISE_JUMP_VISUALS','results/jump_contact_visualization')

def matrix(p,r):
    m=Matrix.Identity(4);m.translation=Vector(p)
    for i in range(3):
        for j in range(3):m[i][j]=float(r[3*i+j])
    return m
def linear(v):return v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4
def material(name,color,rough=.6):
    m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=rough;return m
def tag(ob,gid):
    c=gid*MASK_STEP;ob.color=(*[linear(x/255) for x in (c,255-c,128)],1) if gid else (0,0,0,1);ob['tension_group_id']=gid

def main():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [];d=json.loads((OUT/'data.json').read_text('utf-8'));z=dict(np.load(OUT/'geometry.npz'));scale=d['scale'];idx={n:i for i,n in enumerate(d['body_names'])};names=d['names']
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(ROOT/'assets/fullbody/fullbody_muscles_skin.glb'));rig=next(o for o in bpy.data.objects if o.type=='ARMATURE');add_torso_surfaces(ROOT,rig)
    skin=bpy.data.objects['Skin'];smooth,counts=copy_surface(skin,d['groups'],names,map_atlas)
    white=material('Muscle surface',(.74,.77,.80));unknown=material('Unmodeled or parameter review',(.31,.36,.43));endsmat=material('Head hands feet',(.43,.48,.54));bonemat=material('Skeleton',(.34,.39,.45));mapping=[];anatomy=[]
    keep=[]
    for v in skin.data.vertices:
        w={skin.vertex_groups[g.group].name:g.weight for g in v.groups};p=skin.matrix_world@v.co
        face=w.get('head',0)>.35 and (p.z>1.64 or (p.z>1.58 and p.y<.19));ends=sum(v for n,v in w.items() if any(k in n for k in ['ph_','proxph','midph','distph','thumb','firstmc','lunate','calcn','toes','talus']))>.65;keep.append(face or ends)
    bm=bmesh.new();bm.from_mesh(skin.data);bm.verts.ensure_lookup_table();bmesh.ops.delete(bm,geom=[v for v in bm.verts if not keep[v.index]],context='VERTS');bm.to_mesh(skin.data);bm.free();skin.data.materials.clear();skin.data.materials.append(endsmat);tag(skin,0);anatomy.append(skin)
    for ob in list(bpy.data.objects):
        if ob.type!='MESH' or ob in [skin,smooth]:continue
        if not any(m.type=='ARMATURE' for m in ob.modifiers) or 'platysma' in ob.name.lower():ob.hide_render=True;continue
        fit_lower_rectus_depth(ob);fit_lat_insertion(ob,ROOT,ROOT/'results/ohp_fullbody');ids=map_atlas(ob.name,names);g=next((g for g in d['groups'] if ids and set(ids).issubset(g['indices'])),None);gid=g['display_id'] if g else 0;tag(ob,gid);ob.data.materials.clear();ob.data.materials.append(white if gid else unknown)
        for p in ob.data.polygons:p.use_smooth=True
        anatomy.append(ob);mapping.append({'mesh':ob.name,'group_id':gid,'key':g['key'] if g else None,'side':g['side'] if g else None})
    root=bpy.data.objects.new('Front view root',None);bpy.context.collection.objects.link(root)
    for ob in list(bpy.data.objects):
        if ob!=root and ob.parent is None:ob.parent=root
    bones=[];old=json.loads((ROOT/'results/ohp_fullbody/fullbody_metadata.json').read_text('utf-8'))
    for item in json.loads((ROOT/'results/ohp_fullbody/bone_meshes.json').read_text('utf-8')):
        if item['body'] not in idx or item['body'] in old.get('hand_display_body_names',[]):continue
        mesh=bpy.data.meshes.new(item['name']);mesh.from_pydata(item['vertices'],[],item['faces']);mesh.materials.append(bonemat);mesh.update();ob=bpy.data.objects.new('Bone '+item['name'],mesh);bpy.context.collection.objects.link(ob);ob.parent=root;tag(ob,0);bones.append((ob,item['body']));anatomy.append(ob)
    yaw=math.radians(25);right=Vector((math.cos(yaw),math.sin(yaw),0));root.scale=(scale,)*3;root.location=-.72*right
    rest={n:matrix(z['rest_body_xpos'][j]/scale,z['rest_body_xmat'][j]) for n,j in idx.items()};br={b.name:b.matrix_local.copy() for b in rig.data.bones};pivot=Vector(z['nodes'][0,d['node_names'].index('pelvis')]);pivot.z=0
    for pb in rig.pose.bones:pb.rotation_mode='QUATERNION'
    surfaces,surface_masks,update_skin=finish_surfaces(smooth,rig,scale,right,len(d['groups']),material,linear,pivot);rig2=surfaces[1].parent;backroot=rig2.parent;backcopies=[]
    for ob in list(anatomy):
        cp=ob.copy();cp.name=ob.name+' posterior';bpy.context.collection.objects.link(cp);cp.parent=backroot if ob.parent==root else rig2
        for mod in cp.modifiers:
            if mod.type=='ARMATURE':mod.object=rig2
        cp.hide_render=True;backcopies.append((ob,cp));anatomy.append(cp)
    # Contact glyphs show external ground reaction, never intrinsic foot tension.
    contact=[];force_mat=material('External ground reaction',(.28,.78,.91))
    for parent in [root,backroot]:
        for side in ['r','l']:
            items=[]
            for kind in ['shaft','head']:
                if kind=='shaft':bpy.ops.mesh.primitive_cylinder_add(vertices=16,radius=1,depth=2)
                else:bpy.ops.mesh.primitive_cone_add(vertices=20,radius1=1,radius2=0,depth=2)
                ob=bpy.context.object;ob.parent=parent;ob.data.materials.append(force_mat);tag(ob,0);items.append(ob)
            contact.append((side,items))
    scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.eevee.taa_render_samples=32;scene.render.resolution_x=1080;scene.render.resolution_y=900;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.compression=25
    world=bpy.data.worlds.new('Studio');scene.world=world;world.use_nodes=True;bg=world.node_tree.nodes['Background'];bg.inputs[0].default_value=(.018,.025,.038,1);bg.inputs[1].default_value=.35
    def aim(ob,target):ob.rotation_euler=(Vector(target)-ob.location).to_track_quat('-Z','Y').to_euler()
    for loc,power,size in [((3,-4,5),700,4),((-3,-1,3),500,3),((0,4,4),900,3)]:
        li=bpy.data.lights.new('Softbox','AREA');li.energy=power;li.shape='DISK';li.size=size;ob=bpy.data.objects.new('Softbox',li);bpy.context.collection.objects.link(ob);ob.location=loc;aim(ob,(0,0,1))
    cd=bpy.data.cameras.new('Shared front oblique');cam=bpy.data.objects.new('Shared front oblique',cd);bpy.context.collection.objects.link(cam);scene.camera=cam;cd.type='ORTHO';cd.ortho_scale=3.05;elev=math.radians(15);center=Vector((-.025,.04,1.17));cam.location=center+Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)))*7;aim(cam,center)
    mask=bpy.data.materials.new('Anatomical group mask');mask.use_nodes=True;nt=mask.node_tree;nt.nodes.clear();info=nt.nodes.new('ShaderNodeObjectInfo');em=nt.nodes.new('ShaderNodeEmission');out=nt.nodes.new('ShaderNodeOutputMaterial');nt.links.new(info.outputs['Color'],em.inputs['Color']);nt.links.new(em.outputs[0],out.inputs['Surface'])
    joint_materials=[]
    for jid in range(1,len(d['joints'])+1):
        ma=bpy.data.materials.new('Joint region mask '+str(jid));ma.use_nodes=True;nodes=ma.node_tree.nodes;nodes.clear();em=nodes.new('ShaderNodeEmission');out=nodes.new('ShaderNodeOutputMaterial');c=jid*16;em.inputs['Color'].default_value=(*[linear(v/255) for v in (c,255-c,32)],1);ma.node_tree.links.new(em.outputs[0],out.inputs['Surface']);joint_materials.append(ma)
    centres={n:np.array(z['rest_nodes'][k])/scale for k,n in enumerate(d['node_names'])};regions=[]
    for jid,joint in enumerate(d['joints'],1):
        key,side=joint['key'].rsplit('_',1);c=centres[joint['key']]
        upstream={'shoulder':'elbow','elbow':'shoulder','wrist':'elbow','hip':'knee','knee':'hip','ankle':'knee'}[key];axis=centres[upstream+'_'+side]-c;axis/=np.linalg.norm(axis)
        half={'shoulder':.055,'elbow':.040,'wrist':.025,'hip':.065,'knee':.045,'ankle':.035}[key];radius=half;regions.append((jid,c,axis,half,radius))
    joint_faces={};joint_counts=np.zeros(len(d['joints']),dtype=int)
    for ob in anatomy:
        if not ob.name.startswith('Bone '):continue
        if ob.data.name in joint_faces:continue
        centers=np.array([p.center[:] for p in ob.data.polygons]);best=np.ones(len(centers));labels=np.zeros(len(centers),dtype=int)
        for jid,c,axis,half,radius in regions:
            delta=centers-c;long=delta@axis;radial=np.sum(delta*delta,axis=1)-long*long;score=(long/half)**2+radial/radius**2;inside=score<best;labels[inside]=jid;best[inside]=score[inside]
        joint_faces[ob.data.name]=labels
        for jid in range(1,len(d['joints'])+1):joint_counts[jid-1]+=int((labels==jid).sum())
    mats={ob.name:list(ob.data.materials) for ob in bpy.data.objects if ob.type=='MESH'}
    # Copies share mesh data; material restoration is idempotent.
    def restore():
        for name,ms in mats.items():
            ob=bpy.data.objects[name];ob.data.materials.clear()
            for ma in ms:ob.data.materials.append(ma)
            if ob.data.name in joint_faces:
                for poly in ob.data.polygons:poly.material_index=0
        restore_regions(surfaces)
    def pose(i):
        desired={}
        for pb in rig.pose.bones:
            name='sacrum' if pb.name=='pelvis' else pb.name
            if name in idx:
                j=idx[name];desired[pb.name]=matrix(z['body_xpos'][i,j]/scale,z['body_xmat'][i,j])@rest[name].inverted()@br[pb.name]
        for pb in rig.pose.bones:
            if pb.name not in desired:continue
            pb.matrix_basis=br[pb.name].inverted()@br[pb.parent.name]@desired[pb.parent.name].inverted()@desired[pb.name] if pb.parent and pb.parent.name in desired else br[pb.name].inverted()@desired[pb.name]
        update_skin()
        for ob,name in bones:
            j=idx[name];ob.matrix_basis=matrix(z['body_xpos'][i,j]/scale,z['body_xmat'][i,j])@rest[name].inverted()
        for ob,cp in backcopies:
            if ob.parent==root:cp.matrix_basis=ob.matrix_basis.copy()
        for side,items in contact:
            inds=np.flatnonzero(z['contact_sides']==(0 if side=='r' else 1));fs=z['contact_forces_N'][i,inds];ps=z['contact_positions_m'][i,inds];F=fs.sum(axis=0);norm=np.linalg.norm(F);w=np.maximum(fs[:,2],0);p=np.average(ps,axis=0,weights=w) if w.sum()>1e-5 else ps.mean(axis=0);p[0]+=-.11 if side=='r' else .11;p=Vector(p/scale);v=Vector(F/max(norm,1e-8));length=min(.38,.10+.11*math.sqrt(norm/(90*9.80665)))/scale;h=.045/scale
            for ob in items:ob.hide_render=norm<2;ob.rotation_mode='QUATERNION';ob.rotation_quaternion=v.to_track_quat('Z','Y')
            items[0].location=p+v*(length-h)/2;items[0].scale=(.006,.006,(length-h)/2);items[1].location=p+v*(length-h/2);items[1].scale=(.018,.018,h/2)
        bpy.context.view_layer.update()
    selected=list(range(len(d['times'])))
    if '--poster' in args:selected=d['preview_indices']
    (OUT/'render_mapping.json').write_text(json.dumps({'display_modes':['anatomy'],'default':'anatomy','meshes':mapping,'skin_faces_per_region':counts,'same_values_front_back':True,'mask_step':MASK_STEP,'camera_yaw_deg':25,'camera_elevation_deg':15,'muscle_bulk_enlargement':False,'ground_arrows':'external side-summed contact force; glyph length display-scaled and capped, values uncapped','foot_intrinsic_force_invented':False,'display_geometry_is_physics':False},indent=2),encoding='utf-8')
    projections={}
    for i in selected:
        pose(i)
        projected=[]
        for parent in [root,backroot]:
            pp=[]
            for position in z['nodes'][i]:
                v=world_to_camera_view(scene,cam,parent.matrix_world@(Vector(position)/scale));pp.append([float(v.x*1080),float((1-v.y)*900)])
            projected.append(pp)
        projections[i]=projected
        for mode in ['anatomy']:
            for ob in anatomy:ob.hide_render=mode!='anatomy'
            for ob in surfaces:ob.hide_render=mode!='humanoid'
            restore();scene.render.film_transparent=True;scene.view_settings.view_transform='AgX';base=OUT/('anatomy/base' if mode=='anatomy' else 'base');base.mkdir(parents=True,exist_ok=True);scene.render.filepath=str(base/f'frame_{i:04d}.png');bpy.ops.render.render(write_still=True)
            for ob in bpy.data.objects:
                if ob.type!='MESH':continue
                ob.data.materials.clear()
                for ma in (surface_masks if ob in surfaces else [mask]):ob.data.materials.append(ma)
            for ob in anatomy:
                if ob.data.name not in joint_faces:continue
                ob.data.materials.clear();ob.data.materials.append(mask)
                for ma in joint_materials:ob.data.materials.append(ma)
                for poly,jid in zip(ob.data.polygons,joint_faces[ob.data.name]):poly.material_index=int(jid)
            restore_regions(surfaces);scene.render.film_transparent=False;scene.view_settings.view_transform='Standard';bg.inputs[0].default_value=(0,0,0,1);bg.inputs[1].default_value=0;folder=OUT/('anatomy/mask' if mode=='anatomy' else 'mask');folder.mkdir(parents=True,exist_ok=True);scene.render.filepath=str(folder/f'frame_{i:04d}.png');bpy.ops.render.render(write_still=True);bg.inputs[0].default_value=(.018,.025,.038,1);bg.inputs[1].default_value=.35
        print('WHOLE_BODY_FRAME',i,flush=True)
    (OUT/'projections.json').write_text(json.dumps(projections))
    (OUT/'joint_region_mapping.json').write_text(json.dumps(dict(region_keys=[j['key'] for j in d['joints']],rest_face_counts=joint_counts.tolist(),display='Visible articular bone surfaces, black-to-red by joint reaction force; no point markers',tissue_stress_computed=False),indent=2))
if __name__=='__main__':main()
