"""Bind calculated force groups and joint-region colours to rendered anatomy."""
import json
import numpy as np
import bpy
from atlas_mapping import map_atlas
from case_contours import rgb,joint_rgb

def linear(c):
    c=np.asarray(c)/255
    return np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)

def setup(folder,anatomy,skin,bones,scale):
    source=folder/'analysis/visualization';data=json.loads((source/'data.json').read_text());geo=np.load(source/'geometry.npz')
    times=np.array(data['times']);F=np.array(data['group_tension_N']);J=np.array(data['joint_force_N']);names=data['names'];groups=data['groups']
    mats=[];mapping=[]
    for g in groups:
        ma=bpy.data.materials.new('Tension '+g['key']+' '+g['side']);ma.use_nodes=True;mats.append(ma)
    for ob in anatomy:
        if ob==skin or not any(mod.type=='ARMATURE' for mod in ob.modifiers):continue
        ids=map_atlas(ob.name,names);gi=next((i for i,g in enumerate(groups) if ids and set(ids).issubset(g['indices']) and g.get('released',True)),None)
        if gi is not None:
            ob.data.materials.clear();ob.data.materials.append(mats[gi]);mapping.append(dict(mesh=ob.name,group=groups[gi]['key'],side=groups[gi]['side']))
    jm=[]
    for k,item in enumerate(data['joints']):
        ma=bpy.data.materials.new('Joint reaction '+item['key']);ma.use_nodes=True;jm.append(ma)
    centres={n:np.asarray(geo['rest_nodes'][k])/scale for k,n in enumerate(data['node_names'])};regions=[]
    for j,item in enumerate(data['joints']):
        key,side=item['key'].rsplit('_',1);c=centres[item['key']]
        upstream={'shoulder':'elbow','elbow':'shoulder','wrist':'elbow','hip':'knee','knee':'hip','ankle':'knee'}[key]
        axis=centres[upstream+'_'+side]-c;axis/=np.linalg.norm(axis)
        half={'shoulder':.055,'elbow':.040,'wrist':.025,'hip':.065,'knee':.045,'ankle':.035}[key]
        regions.append((j,c,axis,half))
    used=set();counts=np.zeros(len(jm),int)
    for ob,body_name in bones:
        if ob.data.name in used:continue
        used.add(ob.data.name);centers=np.array([p.center[:] for p in ob.data.polygons]);best=np.ones(len(centers));labels=np.zeros(len(centers),int)
        for j,c,axis,half in regions:
            delta=centers-c;along=delta@axis;radial=np.sum(delta*delta,axis=1)-along*along;score=(along/half)**2+radial/half**2
            inside=score<best;labels[inside]=j+1;best[inside]=score[inside]
        if body_name.startswith('patella_'):
            side=body_name.rsplit('_',1)[1];labels[:]=next(i+1 for i,j in enumerate(data['joints']) if j['key']=='knee_'+side)
        for ma in jm:ob.data.materials.append(ma)
        for poly,label in zip(ob.data.polygons,labels):poly.material_index=int(label)
        for j in range(len(jm)):counts[j]+=int((labels==j+1).sum())
    (folder/'tension_render_mapping.json').write_text(json.dumps(dict(meshes=mapping,joint_face_counts=counts.tolist(),
        palette='common absolute muscle-group N, logarithmic; joint-region reaction black/red, linear',palette_upper_N=20000,
        values_are_conditional=True,muscle_activation_or_stress=False),indent=2))
    assert any(m['group']=='latissimus' for m in mapping) and any(m['group']=='trapezius' for m in mapping)
    def update(t):
        for i,ma in enumerate(mats):ma.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(*linear(rgb(np.interp(t,times,F[i]))),1)
        for i,ma in enumerate(jm):ma.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(*linear(joint_rgb(np.interp(t,times,J[i]))),1)
    return update
