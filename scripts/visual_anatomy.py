"""Presentation-only muscle belly enlargement; no joint or physics edits."""
import numpy as np


def enlarge_muscle_belly(ob, settings):
 name=ob.name.lower()
 broad=any(s in name for s in ['pectoralis major','latissimus dorsi','trapezius','rectus abdominis','external oblique'])
 if 'latissimus dorsi' in name:
  width,thickness=settings['lat_width_scale'],settings['lat_thickness_scale']
 elif 'rectus abdominis' in name:
  width,thickness=settings['rectus_width_scale'],settings['rectus_thickness_scale']
 elif 'trapezius' in name:
  width,thickness=1.12,2.0
 elif broad:
  width,thickness=settings['broad_width_scale'],settings['broad_thickness_scale']
 elif 'deltoid' in name:
  width=thickness=settings['deltoid_radius_scale']
 elif any(s in name for s in ['biceps brachii','triceps brachii','brachialis','gluteus','vastus','rectus femoris','biceps femoris','semitendinosus','semimembranosus','gastrocnemius','soleus']):
  width=thickness=settings['limb_radius_scale']
 else:return None
 vertices=np.array([v.co[:] for v in ob.data.vertices],dtype=float)
 if len(vertices)<8:return None
 centre=vertices.mean(axis=0);delta=vertices-centre
 values,axes=np.linalg.eigh(delta.T@delta)
 # Columns: long axis, in-plane width, thickness. Keep the long axis fixed.
 axes=axes[:,np.argsort(values)[::-1]];local=delta@axes
 low,high=local[:,0].min(),local[:,0].max()
 if high-low<1e-6:return None
 taper=np.sin(np.pi*np.clip((local[:,0]-low)/(high-low),0,1))**2
 local[:,1]*=1+(width-1)*taper
 local[:,2]*=1+(thickness-1)*taper
 result=local@axes.T+centre
 for vertex,co in zip(ob.data.vertices,result):vertex.co=co
 ob.data.update()
 return {'mesh':ob.name,'width_scale_at_belly':width,'thickness_scale_at_belly':thickness,
         'max_local_vertex_displacement_m':float(np.linalg.norm(result-vertices,axis=1).max())}
