"""Calibrate the shaft to the actual displayed upper-chest surface."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from solve_fullbody import Body,OUT,CONFIG,ROOT,ARM

def front_contact(vertices,edges,zbar,radius=.014,centre_x=-.025):
 a=vertices[edges[:,0]];delta=vertices[edges[:,1]]-a;lo=np.zeros(len(a));hi=np.ones(len(a))
 for axis,low,high in [(0,centre_x-.18,centre_x+.18),(2,zbar-radius+1e-9,zbar+radius-1e-9)]:
  changing=abs(delta[:,axis])>1e-12
  first=(low-a[:,axis])/np.where(changing,delta[:,axis],1);second=(high-a[:,axis])/np.where(changing,delta[:,axis],1)
  lo=np.maximum(lo,np.where(changing,np.minimum(first,second),0));hi=np.minimum(hi,np.where(changing,np.maximum(first,second),1))
  hi[~changing&((a[:,axis]<low)|(a[:,axis]>high))]=-1
 valid=hi>=lo;a=a[valid];delta=delta[valid];lo=lo[valid];hi=hi[valid]
 dz=delta[:,2];slope=delta[:,1]/np.where(abs(dz)>1e-12,dz,1)
 zopt=zbar-radius*slope/np.sqrt(1+slope*slope)
 t=np.where(abs(dz)>1e-12,(zopt-a[:,2])/np.where(abs(dz)>1e-12,dz,1),np.where(delta[:,1]>0,lo,hi))
 t=np.clip(t,lo,hi);point=a+t[:,None]*delta
 centre_y=point[:,1]-np.sqrt(np.maximum(0,radius**2-(point[:,2]-zbar)**2));k=centre_y.argmin()
 return float(centre_y[k]),point[k]

def main():
 body=Body();m=body.m;data=np.load(OUT/'fullbody_motion.npz');surface=json.loads((ROOT/'results/grip_comparison/rack_surface.json').read_text('utf-8'))
 for item in json.loads((OUT/'bone_meshes.json').read_text('utf-8')):
  if item['body'] in ['clavicle_r','clavicle_l','torso']:
   surface['objects'].append({'name':item['name'],'vertices':item['vertices'],'faces':item['faces'],
    'weights':[[[item['body'],1.]] for _ in item['vertices']],'mesh_to_rig':np.eye(4).tolist()})
 vertices=[];weights={};edges=[]
 for ob in surface['objects']:
  offset=len(vertices);assert np.allclose(ob['mesh_to_rig'],np.eye(4))
  vertices.extend(ob['vertices'])
  for face in ob['faces']:
   edges.extend([[offset+face[i],offset+face[(i+1)%len(face)]] for i in range(len(face))])
  for i,row in enumerate(ob['weights']):
   for name,w in row:weights.setdefault(name,[]).append((offset+i,w))
 vertices=np.array(vertices);edges=np.array(edges);names=[m.body(i).name for i in range(m.nbody)]
 source_rest=data['rest_body_xpos'];source_rot=data['rest_body_xmat'];s=body.length_scale
 addresses=[m.jnt_qposadr[body.joints[n+'_r']] for n in ARM];guess=data['qpos'][0,addresses].copy()
 lower=np.array([-1.65,.01,-1.56,0,-1.56,-.174,-.78]);upper=np.array([2.26,3.13,2.09,np.deg2rad(150),1.56,.436,.78])
 x=-.025-CONFIG['grip_width_m']/s/2;z=CONFIG['bar_start_height_m'];y=.12
 history=[]
 for iteration in range(20):
  target=np.array([x,y/s,z/s])
  def residual(v):
   body.arm(v);elbow=body.d.xanchor[body.joints['elbow_flexion_r']]
   below=max(0.,elbow[2]-target[2]+.15)
   return np.r_[(body.grip()-target)*1000,body.axis()[1:]*3,(elbow[0]-target[0])*4,below*100,
    max(0.,elbow[1]-target[1]+.02)*100,(v-guess)*.001]
  seeds=[guess] if iteration else [guess,np.array([0,.6,0,2.,-1.3,0,0]),np.array([1.,.7,.5,1.8,-1.3,0,0]),np.array([-1.,.7,-.5,1.8,-1.3,0,0])]
  fits=[least_squares(residual,seed,bounds=(lower,upper),max_nfev=300,ftol=1e-11,xtol=1e-11,gtol=1e-11) for seed in seeds]
  fit=min(fits,key=lambda result:result.cost);guess=fit.x
  q=body.base.copy()
  for side in ['r','l']:
   for name,value in zip(ARM,guess):q[m.jnt_qposadr[body.joints[name+'_'+side]]]=value
  body.pose(q);deformed=np.zeros_like(vertices)
  for name,entries in weights.items():
   j=names.index(name);entries=np.array(entries);ids=entries[:,0].astype(int);w=entries[:,1]
   local=(vertices[ids]-source_rest[j])@source_rot[j].reshape(3,3)
   current=local@body.d.xmat[j].reshape(3,3).T+body.d.xpos[j]
   deformed[ids]+=current*w[:,None]
  desired,point=front_contact(deformed*s,edges,z);desired-=.001
  error=float(np.linalg.norm(body.grip()-target)*s);history.append({'y_m':y,'desired_y_m':desired,'hand_error_m':error})
  if abs(desired-y)<2e-5:break
  y=desired
 assert error<.001 and abs(desired-y)<2e-5,history
 report={'status':'calibrated','grip_width_m':CONFIG['grip_width_m'],'shaft_radius_m':.014,'target_surface_gap_m':.001,
  'bar_start_height_m':z,'bar_start_y_m':y,'bar_start_forward_m':float(.199545*s-y),'surface_contact_point_m':point.tolist(),
  'initial_arm_coordinates_rad':guess.tolist(),'iterations':history,
  'method':'Rendered enlarged pectoralis surface, exact linear skinning; circular shaft tangent to projected triangle-edge envelope within central upper chest. Geometric placement only; external bar load remains applied through hands.'}
 (OUT/'rack_start.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
