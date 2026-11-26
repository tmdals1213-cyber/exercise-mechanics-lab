"""Add missing real trunk surfaces with anatomically restricted bone weights."""
import json
import bpy
import numpy as np
from mathutils import Vector


def fit_lower_rectus_depth(ob):
 """Fit the supplemental lower rectus to the existing anterior abdominal wall.

The two source atlases have different abdominal depths. Keep the real muscle
surface/inscriptions, translating local cross-sections rather than sculpting abs.
"""
 if 'rectus abdominis' not in ob.name.lower():return None
 side='Left' if 'Left' in ob.name else 'Right'
 reference=next(o for o in bpy.data.objects if side.lower()+' external oblique' in o.name.lower())
 ref=np.array([v.co[:] for v in reference.data.vertices]);vertices=np.array([v.co[:] for v in ob.data.vertices]);original=vertices.copy()
 lo,hi=vertices[:,2].min(),vertices[:,2].max();zs=np.linspace(lo,hi,48);offset=[]
 for z in zs:
  a=ref[(abs(ref[:,2]-z)<.022)&(ref[:,0]>=vertices[:,0].min()-.005)&(ref[:,0]<=vertices[:,0].max()+.005)]
  b=vertices[abs(vertices[:,2]-z)<.022]
  offset.append(min(0,float(np.quantile(a[:,1],.05)-np.quantile(b[:,1],.05)-.009)) if len(a)>3 and len(b)>3 else 0.)
 offset=np.convolve(np.pad(offset,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
 def smooth(x):
  x=np.clip(x,0,1);return x*x*(3-2*x)
 shift=np.interp(vertices[:,2],zs,offset)*smooth((vertices[:,2]-.925)/.055)*smooth((1.24-vertices[:,2])/.08)
 vertices[:,1]+=shift
 for v,co in zip(ob.data.vertices,vertices):v.co=co
 ob.data.update()
 return {'mesh':ob.name,'method':'lower-rectus regional anterior-envelope registration; original surface retained',
  'max_forward_registration_m':float(-shift.min()),'physics_changed':False}


def add_torso_surfaces(root,rig):
 asset=json.loads((root/'assets/fullbody/torso_supplement.json').read_text(encoding='utf-8'))
 def smooth(x):
  x=max(0,min(1,x));return x*x*(3-2*x)
 for item in asset['meshes']:
  mesh=bpy.data.meshes.new(item['name']);mesh.from_pydata(item['vertices'],[],item['faces']);mesh.update()
  ob=bpy.data.objects.new(item['name'],mesh);bpy.context.collection.objects.link(ob);ob.parent=rig
  groups={bone.name:ob.vertex_groups.new(name=bone.name) for bone in rig.data.bones}
  for vertex in mesh.vertices:
   if 'Latissimus' in item['name']:
    # Most of this broad muscle originates on the trunk. Nearest-neighbour
    # transfer alone wrongly assigned its lateral sheet to distal arm bones.
    lateral=abs(vertex.co.x+.025)
    t=smooth((vertex.co.z-1.34)/.09)*smooth((lateral-.105)/.045)
    groups['torso'].add([vertex.index],1-t,'REPLACE')
    groups['humerus_l' if 'Left' in item['name'] else 'humerus_r'].add([vertex.index],t,'REPLACE')
   elif 'Cervicis' in item['name']:
    w=smooth((vertex.co.z-1.50)/.12)
    groups['torso'].add([vertex.index],1-w,'REPLACE');groups['head'].add([vertex.index],w,'REPLACE')
   elif 'Tensor Fasciae' in item['name']:
    w=smooth((1.08-vertex.co.z)/.13)
    groups['pelvis'].add([vertex.index],1-w,'REPLACE')
    groups['femur_l' if 'Left' in item['name'] else 'femur_r'].add([vertex.index],w,'REPLACE')
   else:
    # No abdominal, spinal or lumbar surface is allowed to inherit arm weights.
    top=smooth((vertex.co.z-1.10)/.16);bottom=smooth((1.07-vertex.co.z)/.08)
    weights={'torso':top,'pelvis':bottom*(1-top),'lumbar3':(1-bottom)*(1-top)}
    for name,weight in weights.items():
     if weight>0:groups[name].add([vertex.index],weight,'REPLACE')
  modifier=ob.modifiers.new('Anatomical attachment weights','ARMATURE');modifier.object=rig
  ob['source']=item['source'];ob['license']=item['license'];ob['visual_registration_only']=True
 return len(asset['meshes'])


def fit_lat_insertion(ob,root,result_dir=None):
 """Match the small tendon end to the model's actual proximal-humerus sites."""
 if 'latissimus' not in ob.name.lower():return None
 info=json.loads(((result_dir or root/'results/ohp_fullbody')/'anatomy_landmarks.json').read_text('utf-8'))
 side=1 if 'left' in ob.name.lower() else 0
 anchor=np.array(info['lat_insertion_rest'][side])
 vertices=np.array([v.co[:] for v in ob.data.vertices])
 index=int(np.linalg.norm(vertices-anchor,axis=1).argmin());tip=vertices[index].copy();shift=anchor-tip
 distance=np.linalg.norm(vertices-tip,axis=1);falloff=np.exp(-(distance/.035)**2)
 vertices+=falloff[:,None]*shift
 groups={name:ob.vertex_groups[name] for name in ['torso','humerus_l' if side else 'humerus_r']}
 all_indices=list(range(len(vertices)))
 for group in ob.vertex_groups:group.remove(all_indices)
 for vertex,co,distance in zip(ob.data.vertices,vertices,np.linalg.norm(vertices-anchor,axis=1)):
  vertex.co=co
  # Broad thoracolumbar origin stays on trunk; only the narrow tendon end follows
  # the upper arm. No scapular, elbow, radius, or ulna weights are permitted.
  w=float(np.exp(-(distance/.045)**2))
  groups['torso'].add([vertex.index],1-w,'REPLACE')
  groups['humerus_l' if side else 'humerus_r'].add([vertex.index],w,'REPLACE')
 ob.data.update();ob['insertion_vertex']=index
 return {'mesh':ob.name,'insertion_vertex':index,'rest_model_insertion_native_m':anchor.tolist(),
  'rest_insertion_error_m':float(np.linalg.norm(vertices[index]-anchor)),'rest_fit_shift_m':shift.tolist()}
