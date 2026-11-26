"""Distance from a body triangle mesh to the circular bar shaft."""
import numpy as np

def shaft_gap(vertices,faces,bar,radius=.014):
 tri=vertices[np.asarray(faces)];p=tri[:,:,1:]-np.asarray(bar)[None,None,1:]
 cross=lambda a,b:a[:,0]*b[:,1]-a[:,1]*b[:,0]
 area=cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);c=np.stack([cross(p[:,0],p[:,1]),cross(p[:,1],p[:,2]),cross(p[:,2],p[:,0])],axis=1)
 if np.any((abs(area)>1e-12)&((c.min(1)>=-1e-12)|(c.max(1)<=1e-12))):return -radius
 best=1e9
 for a,b in [(p[:,0],p[:,1]),(p[:,1],p[:,2]),(p[:,2],p[:,0])]:
  e=b-a;t=np.clip(-(a*e).sum(1)/np.maximum((e*e).sum(1),1e-20),0,1)
  best=min(best,float(np.linalg.norm(a+t[:,None]*e,axis=1).min()))
 return best-radius
