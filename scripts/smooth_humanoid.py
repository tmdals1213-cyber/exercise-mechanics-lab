"""Continuous opaque skin surface, with rest-space atlas region labels."""
import bpy,bmesh,math
from mathutils.kdtree import KDTree
from tension_groups import MASK_STEP

def copy_surface(skin,groups,names,map_atlas):
    """Copy original skin before anatomy-only bulk corrections or masking."""
    samples=[];labels=[]
    for ob in list(bpy.data.objects):
        if ob.type!='MESH' or ob==skin or not any(m.type=='ARMATURE' for m in ob.modifiers):continue
        ids=map_atlas(ob.name,names);group=next((g for g in groups if ids and set(ids).issubset(g['indices'])),None);gid=group['display_id'] if group else 0
        for v in ob.data.vertices:samples.append(ob.matrix_world@v.co);labels.append(gid)
    tree=KDTree(len(samples))
    for i,p in enumerate(samples):tree.insert(p,i)
    tree.balance();ob=skin.copy();ob.data=skin.data.copy();ob.name='Smooth humanoid front';bpy.context.collection.objects.link(ob)
    # The atlas skin has duplicated UV-boundary vertices. Weld coincident
    # vertices before smoothing so the mannequin cannot split along its seams.
    bm=bmesh.new();bm.from_mesh(ob.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00005);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(ob.data);bm.free();ob.data.update()
    counts={};attr=ob.data.attributes.new('tension_region','INT','FACE')
    for poly in ob.data.polygons:
        votes={}
        for co,j,dist in tree.find_n(ob.matrix_world@poly.center,5):votes[labels[j]]=votes.get(labels[j],0)+1/max(dist,.002)**2
        gid=max(votes,key=votes.get);ws={}
        for vi in poly.vertices:
            for w in ob.data.vertices[vi].groups:
                n=ob.vertex_groups[w.group].name;ws[n]=ws.get(n,0)+w.weight/len(poly.vertices)
        ends=sum(w for n,w in ws.items() if n=='head' or any(k in n for k in ['ph_','proxph','midph','distph','thumb','firstmc','lunate','calcn','toes','talus']))
        if ends>.5:gid=0
        attr.data[poly.index].value=gid;poly.material_index=gid;poly.use_smooth=True;counts[str(gid)]=counts.get(str(gid),0)+1
    # Surface labels are an illustrative projection, not exposed muscle edges.
    # Remove isolated label speckles while retaining connected region borders.
    edge_faces={}
    for p in ob.data.polygons:
        for edge in p.edge_keys:edge_faces.setdefault(tuple(sorted(edge)),[]).append(p.index)
    neighbors=[set() for _ in ob.data.polygons]
    for faces in edge_faces.values():
        for i in faces:neighbors[i].update(j for j in faces if i!=j)
    for _ in range(10):
        old=[x.value for x in attr.data];new=[]
        for i,nb in enumerate(neighbors):
            vote={old[i]:1.3}
            for j in nb:vote[old[j]]=vote.get(old[j],0)+1
            new.append(max(vote,key=vote.get))
        for x,v in zip(attr.data,new):x.value=v
    counts={}
    for x in attr.data:counts[str(x.value)]=counts.get(str(x.value),0)+1
    # Smooth shading is retained on the source continuous surface. A single
    # corner color attribute carries region IDs without dozens of shaders.
    colors=ob.data.color_attributes.new(name='RegionCode',type='FLOAT_COLOR',domain='CORNER')
    def lin(v):return v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4
    for p in ob.data.polygons:
        gid=attr.data[p.index].value;c=gid*MASK_STEP;rgb=tuple(lin(x/255) for x in (c,255-c,128)) if gid else (0,0,0)
        for li in p.loop_indices:colors.data[li].color=(*rgb,1)
    smooth=ob.modifiers.new('Gentle mannequin surface smoothing','SMOOTH');smooth.factor=.5;smooth.iterations=4
    ob['smooth_surface']=True
    return ob,counts

def finish_surfaces(front,rig,scale,right,group_count,material,linear,pivot):
    rig2=rig.copy();rig2.data=rig.data.copy();bpy.context.collection.objects.link(rig2)
    root=bpy.data.objects.new('Posterior view root',None);bpy.context.collection.objects.link(root);root.scale=(scale,)*3;root.location=.72*right+2*pivot;root.rotation_euler.z=math.pi;rig2.parent=root
    back=front.copy();back.data=front.data.copy();back.name='Smooth humanoid back';bpy.context.collection.objects.link(back);back.parent=rig2
    for mod in back.modifiers:
        if mod.type=='ARMATURE':mod.object=rig2
    for ob in bpy.data.objects:
        if ob.type in ['MESH','CURVE']:ob.hide_render=ob not in [front,back]
    base=material('Opaque humanoid surface',(.80,.83,.87),.53);unknown=material('Unmapped humanoid region',(.39,.44,.50),.53)
    m=bpy.data.materials.new('Humanoid region mask');m.use_nodes=True;nt=m.node_tree;nt.nodes.clear();color=nt.nodes.new('ShaderNodeVertexColor');color.layer_name='RegionCode';e=nt.nodes.new('ShaderNodeEmission');nt.links.new(color.outputs['Color'],e.inputs['Color']);o=nt.nodes.new('ShaderNodeOutputMaterial');nt.links.new(e.outputs[0],o.inputs['Surface']);maskm=[m]
    for ob in [front,back]:
        ob.data.materials.clear()
        ob.data.materials.append(base)
    def update():
        for pb in rig2.pose.bones:pb.matrix_basis=rig.pose.bones[pb.name].matrix_basis.copy()
    return [front,back],maskm,update

def restore_regions(surfaces):
    for ob in surfaces:
        for poly in ob.data.polygons:poly.material_index=0
