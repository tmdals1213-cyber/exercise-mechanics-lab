"""OpenSim MoBL-ARMS geometry + inverse dynamics + bounded static recruitment.

Right arm is solved and mirrored for a symmetric barbell press. Coupled
coordinates are eliminated explicitly; no scapular stabilizer forces are claimed.
"""
from pathlib import Path
import os, json, math, csv, hashlib, xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize, LinearConstraint, Bounds
from scipy.interpolate import CubicSpline
import opensim as osim

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/mobl/MOBL_ARMS_41.osim'
OUT=ROOT/'results/ohp_100kg_2s'
INDEPENDENT=['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion']
GROUPS={
 'deltoid':('삼각근',['DELT1','DELT2','DELT3']),
 'deltoid_anterior':('전면 삼각근',['DELT1']),
 'deltoid_middle':('측면 삼각근',['DELT2']),
 'deltoid_posterior':('후면 삼각근',['DELT3']),
 'triceps':('삼두근',['TRIlong','TRIlat','TRImed']),
 'pectoralis':('대흉근',['PECM1','PECM2','PECM3']),
 'latissimus':('광배근',['LAT1','LAT2','LAT3']),
 'biceps':('이두근',['BIClong','BICshort']),
 'rotator_cuff':('회전근개',['SUPSP','INFSP','SUBSC','TMIN'])
}


class ArmModel:
 def __init__(self,config):
  osim.Logger.setLevelString('error')
  self.config=config
  os.chdir(MODEL.parent)  # OpenSim C++ file I/O cannot reliably open Unicode absolute paths.
  original=osim.Model(MODEL.name)
  self.couplers=[]
  root=ET.parse(MODEL).getroot()
  for node in root.findall('.//ConstraintSet/objects/CoordinateCouplerConstraint'):
   xs=np.fromstring(node.findtext('.//SimmSpline/x'),sep=' ')
   ys=np.fromstring(node.findtext('.//SimmSpline/y'),sep=' ')
   function=CubicSpline(xs,ys,bc_type='natural')
   self.couplers.append((node.findtext('independent_coordinate_names').strip(),node.findtext('dependent_coordinate_name').strip(),function))
  original.updConstraintSet().clearAndDestroy()
  state=original.initSystem()
  # Anatomical segment dimensions: target geometry, not a strength calibration.
  bodies=original.getBodySet()
  h=bodies.get('humerus').getPositionInGround(state).to_numpy()
  u=bodies.get('ulna').getPositionInGround(state).to_numpy()
  hand=bodies.get('hand').getPositionInGround(state).to_numpy()
  upper_target=config['geometry']['upper_arm_m']
  lower_target=config['geometry']['forearm_m']
  upper_scale=upper_target/np.linalg.norm(u-h)
  lower_scale=lower_target/np.linalg.norm(hand-u)
  torso_scale=(config['geometry']['shoulder_width_m']/2)/abs(h[0])
  factors={}
  scales=osim.ScaleSet()
  for b in bodies:
   factor=(upper_scale if b.getName()=='humerus' else lower_scale if b.getName() in ['ulna','radius'] else torso_scale if b.getName() in ['thorax','clavicle','clavphant','scapula','scapphant'] else 1.0)
   item=osim.Scale(); item.setSegmentName(b.getName());item.setScaleFactors(osim.Vec3(factor));scales.cloneAndAppend(item);factors[b.getName()]=factor
  original.scale(state,scales,False)
  # de Leva male body mass fractions; original inertia shape and COM are retained.
  bodymass=config['subject']['mass_kg']
  forearm_original=sum(original.getBodySet().get(n).getMass() for n in ['ulna','radius'])
  for name,target in [('humerus',bodymass*.0271),('ulna',bodymass*.0162*original.getBodySet().get('ulna').getMass()/forearm_original),('radius',bodymass*.0162*original.getBodySet().get('radius').getMass()/forearm_original),('hand',bodymass*.0061)]:
   b=original.updBodySet().get(name);ratio=target/b.getMass();inertia=b.getInertia()
   components=np.r_[inertia.getMoments().to_numpy(),inertia.getProducts().to_numpy()]*ratio
   b.setInertia(osim.Inertia(*[float(v) for v in components]));b.setMass(target)
  processor=osim.ModelProcessor(original)
  processor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
  processor.append(osim.ModOpIgnoreTendonCompliance())
  processor.append(osim.ModOpIgnoreActivationDynamics())
  self.model=processor.process(); self.model.finalizeConnections()
  self.state=self.model.initSystem()
  self.coords=list(self.model.getCoordinateSet())
  self.byname={c.getName():c for c in self.coords}
  self.defaults={c.getName():c.getDefaultValue() for c in self.coords}
  self.defaults['t_y']=config['subject']['height_m']*config['geometry']['shoulder_height_fraction']
  self.hand=self.model.getBodySet().get('hand')
  self.station=osim.Vec3(0,-config['geometry']['wrist_to_grip_m'],0)
  self.muscles=list(self.model.getMuscles());self.names=[m.getName() for m in self.muscles]
  # Determine the coordinate-to-mobility mapping rather than relying on XML order.
  for c in self.coords:c.setSpeedValue(self.state,0)
  self.uindices=[]
  for c in self.coords:
   c.setSpeedValue(self.state,1); indexes=np.flatnonzero(abs(self.state.getU().to_numpy())>.5)
   assert len(indexes)==1
   self.uindices.append(int(indexes[0]));c.setSpeedValue(self.state,0)
  self.B=np.zeros((len(self.coords),len(INDEPENDENT)))
  self.set_pose(np.array([math.pi/2,.6,0,1.8,0,0,0]))
  for j in range(len(INDEPENDENT)):
   q=np.array([math.pi/2,.6,0,1.8,0,0,0]);eps=1e-5
   qp=q.copy();qm=q.copy();qp[j]+=eps;qm[j]-=eps
   self.B[:,j]=(self.full_values(qp)-self.full_values(qm))/(2*eps)
  self.Bu=np.zeros_like(self.B)
  for i,ui in enumerate(self.uindices):self.Bu[ui]=self.B[i]
  # Scapulohumeral coupling prescribes the motion, but must NOT transfer scapular
  # virtual work onto GH muscles. Treat the shoulder girdle as an externally
  # supported moving base for the arm-force analysis.
  self.Bload=self.B.copy()
  for i,c in enumerate(self.coords):
   if c.getName().startswith(('sternoclavicular_','unrotscap_','acromioclavicular_','unrothum_')):
    self.Bload[i,:]=0
  self.Buload=np.zeros_like(self.Bload)
  for i,ui in enumerate(self.uindices):self.Buload[ui]=self.Bload[i]
  self.idsolver=osim.InverseDynamicsSolver(self.model)
  for force in self.model.getForceSet():force.setAppliesForce(self.state,False)
  self.scale_factors=factors

 def full_values(self,q):
  values=self.defaults.copy();values.update(zip(INDEPENDENT,q))
  for a,b,function in self.couplers:values[b]=float(function(values[a]))
  return np.array([values[c.getName()] for c in self.coords])

 def set_pose(self,q):
  values=self.full_values(q)
  for c,value in zip(self.coords,values):c.setValue(self.state,float(value),False)
  self.model.realizePosition(self.state)

 def grip(self):return self.hand.findStationLocationInGround(self.state,self.station).to_numpy()

 def axis(self):return self.hand.expressVectorInGround(self.state,osim.Vec3(0,0,1)).to_numpy()

 def kinematics(self):
  return {name:self.model.getBodySet().get(name).getPositionInGround(self.state).to_numpy() for name in ['thorax','clavicle','scapula','humerus','ulna','radius','hand']}

 def lengths(self):return np.array([m.getLength(self.state) for m in self.muscles])

 def point_jacobian(self,q):
  jac=np.empty((3,len(q)));eps=1e-5
  full=self.full_values(q)
  for j in range(len(q)):
   for c,v in zip(self.coords,full+self.Bload[:,j]*eps):c.setValue(self.state,float(v),False)
   self.model.realizePosition(self.state);a=self.grip()
   for c,v in zip(self.coords,full-self.Bload[:,j]*eps):c.setValue(self.state,float(v),False)
   self.model.realizePosition(self.state);b=self.grip();jac[:,j]=(a-b)/(2*eps)
  self.set_pose(q);return jac

 def moment_arms(self,q):
  result=np.empty((len(q),len(self.muscles)));eps=1e-5
  full=self.full_values(q)
  for j in range(len(q)):
   for c,v in zip(self.coords,full+self.Bload[:,j]*eps):c.setValue(self.state,float(v),False)
   self.model.realizePosition(self.state);a=self.lengths()
   for c,v in zip(self.coords,full-self.Bload[:,j]*eps):c.setValue(self.state,float(v),False)
   self.model.realizePosition(self.state);b=self.lengths();result[j]=-(a-b)/(2*eps)
  self.set_pose(q);return result

 def dynamics(self,q,dq,ddq,bar_force):
  self.set_pose(q)
  u=self.Bu@dq;udot=self.Bu@ddq
  for i,v in enumerate(u):self.state.updU().set(i,float(v))
  self.model.realizeVelocity(self.state)
  tau=self.idsolver.solve(self.state,osim.Vector.createFromMat(udot)).to_numpy()
  required=self.Buload.T@tau-self.point_jacobian(q).T@bar_force
  self.set_pose(q)
  for i,v in enumerate(u):self.state.updU().set(i,float(v))
  self.model.realizeDynamics(self.state)
  capacity=[];passive=[];fiber=[]
  for mus in self.muscles:
   cos=float(mus.getCosPennationAngle(self.state))
   capacity.append(mus.getMaxIsometricForce()*mus.getActiveForceLengthMultiplier(self.state)*mus.getForceVelocityMultiplier(self.state)*cos)
   passive.append(max(0.,mus.getPassiveFiberForce(self.state)*cos))
   fiber.append(mus.getFiberLength(self.state))
  return required,np.array(capacity),np.array(passive),np.array(fiber)


def make_motion(arm,config):
 # Full independent shoulder/elbow/pronation/wrist IK, including hand-bar alignment.
 lower=np.array([-1.65,.01,-1.56,0,-1.56,-.1745,-1.2217]);upper=np.array([2.26,3.12,2.09,2.2689,1.56,.4363,1.2217])
 grip_x=-config['geometry']['grip_width_m']/2
 shoulder_y=config['subject']['height_m']*config['geometry']['shoulder_height_fraction']
 arm.defaults['t_y']=shoulder_y
 def residual(x,target,reference):
  arm.set_pose(x)
  return np.r_[(arm.grip()-target)*5,arm.axis()[1:]*.25,(x-reference)*1e-4]
 # Exact extension endpoint: solve x,z and hand-bar alignment, with elbow fixed at zero.
 def endpoint(x):
  q=np.array([x[0],x[1],x[2],0,x[3],x[4],x[5]]);arm.set_pose(q)
  p=arm.grip();return np.r_[(p[[0,2]]-np.array([grip_x,0]))*5,arm.axis()[1:]*.25,x[4:]*.0001]
 indices=[0,1,2,4,5,6]
 sol=least_squares(endpoint,[1.5,2.8,0,0,.05,0],bounds=(lower[indices],upper[indices]),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=400)
 assert np.linalg.norm(sol.fun)<1e-3,sol.fun
 qend=np.array([sol.x[0],sol.x[1],sol.x[2],0,sol.x[3],sol.x[4],sol.x[5]]);arm.set_pose(qend);end=arm.grip().copy()
 start=np.array([grip_x,config['geometry']['bar_start_height_m'],config['geometry']['bar_start_forward_offset_m']])
 n=181;progress=1-np.linspace(1,0,n)**2;qs=[];errors=[];guess=qend.copy()
 for s in progress[::-1]:
  target=np.array([grip_x,start[1]+s*(end[1]-start[1]),start[2]*(1-s)**2])
  if s==1:q=qend
  else:
   result=least_squares(residual,guess,args=(target,guess.copy()),bounds=(lower,upper),max_nfev=150,xtol=1e-10,ftol=1e-10,gtol=1e-10)
   guess=result.x;q=guess
  arm.set_pose(q);errors.append(float(np.linalg.norm(arm.grip()-target)));qs.append(q)
 if max(errors)>.004:raise RuntimeError(f'IK grip position error exceeds 4 mm: {max(errors)}')
 # sqrt(distance-to-lockout) parameter avoids the straight-elbow IK singularity.
 return CubicSpline(np.linspace(0,1,n),np.array(qs)),start,end,max(errors)


def smooth_profile(t,duration,ramp,rise):
 """Ninth-order velocity ramp, flat through fourth derivative at endpoints."""
 vmax=rise/(duration-ramp)
 if t<=0:return 0.,0.,0.
 if t>=duration:return rise,0.,0.
 if t>duration-ramp:
  x,v,a=smooth_profile(duration-t,duration,ramp,rise);return rise-x,v,-a
 if t>=ramp:return vmax*(t-ramp/2),vmax,0.
 u=t/ramp
 speed=126*u**5-420*u**6+540*u**7-315*u**8+70*u**9
 integral=21*u**6-60*u**7+67.5*u**8-35*u**9+7*u**10
 return vmax*ramp*integral,vmax*speed,vmax/ramp*630*u**4*(1-u)**4


def recruit(R,capacity,passive,tau,previous=None):
 """Minimum squared normalized active FORCE DEMAND, no physiological cap.

 Also check bounded feasibility separately; demands above capacity remain visible.
 """
 capacity=np.maximum(capacity,1e-5)
 A=R*capacity[None,:];target=tau-R@passive
 rowscale=np.maximum(np.linalg.norm(A,axis=1),1.)
 M=A/rowscale[:,None];b=target/rowscale
 seed=lsq_linear(M,b,bounds=(0,np.inf),tol=1e-10,max_iter=300).x
 result=minimize(lambda a:.5*np.dot(a,a),seed,jac=lambda a:a,method='SLSQP',
                 bounds=Bounds(np.zeros(len(seed)),np.full(len(seed),np.inf)),
                 constraints=[LinearConstraint(M,b,b)],options={'ftol':1e-10,'maxiter':250})
 activation=result.x if result.success else seed
 residual=tau-R@(capacity*activation+passive)
 bounded=lsq_linear(M,b,bounds=(0,1),tol=1e-10,max_iter=300)
 bounded_residual=target-A@bounded.x
 feasible=bool(np.max(abs(bounded_residual))<.25)
 if np.max(abs(residual))>.1:raise RuntimeError(f'Recruitment equilibrium error {np.max(abs(residual)):.3f} Nm')
 return capacity*activation+passive,activation,feasible,float(np.max(abs(residual))),bounded_residual


def main():
 config=json.loads((ROOT/'scenarios/ohp_100kg_2s.json').read_text(encoding='utf-8'))
 arm=ArmModel(config);curve,start,end,ikerror=make_motion(arm,config)
 print('IK ready',start,end,'error',ikerror,flush=True)
 OUT.mkdir(exist_ok=True,parents=True)
 # Save processed anatomical model alongside data, with original retained separately.
 arm.model.printToXML('MOBL_ARMS_scaled_rigid_tendon.osim')
 np.savez(OUT/'ik_reference.npz',sqrt_progress=np.linspace(0,1,181),q=curve(np.linspace(0,1,181)),start=start,end=end)
 motion=config['movement'];duration=motion['ascent_s'];ramp=motion['ramp_s'];rise=end[1]-start[1]
 times=np.linspace(0,duration+motion['lockout_hold_s'],151)
 frames=[];allforce=[];allfiber=[];feasibility=[];maxresidual=0.;max_position_error=0.
 for index,t in enumerate(times):
  x,vy,ay=smooth_profile(t,duration,ramp,rise);p=x/rise
  r=math.sqrt(max(0,1-p));q=curve(r)
  if t<=0 or t>=duration:dq=np.zeros(7);ddq=np.zeros(7)
  else:
   dr=-vy/rise/(2*r);ddr=-ay/rise/(2*r)-(vy/rise)**2/(4*r**3)
   dq=curve(r,1)*dr;ddq=curve(r,2)*dr*dr+curve(r,1)*ddr
  z=start[2]*(1-p)**2;vz=-2*start[2]*(1-p)*vy/rise
  az=2*start[2]*((vy/rise)**2-(1-p)*ay/rise)
  fy=100*(9.80665+ay);fz=100*az
  R=arm.moment_arms(q)
  tau,capacity,passive,fiber=arm.dynamics(q,dq,ddq,np.array([0,-fy/2,-fz/2]))
  # First educational model balances shoulder elevation + elbow flexion only.
  # Use original Fmax as a fixed reference, NOT a length/velocity-dependent
  # physiological capacity. Out-of-plane, wrist and girdle torques are supported.
  reference_force=np.array([m.getMaxIsometricForce() for m in arm.muscles])
  selected=[1,3]
  force,demand,feasible,residual,bounded_residual=recruit(R[selected],reference_force,np.zeros_like(passive),tau[selected])
  maxresidual=max(maxresidual,residual);feasibility.append(feasible)
  arm.set_pose(q);grip=arm.grip();max_position_error=max(max_position_error,float(np.linalg.norm(grip-np.array([start[0],start[1]+x,z]))))
  paths=[]
  for mus in arm.muscles:
   points=mus.getGeometryPath().getCurrentPath(arm.state)
   path=[points.get(i).getLocationInGround(arm.state).to_numpy().round(5).tolist() for i in range(points.getSize())]
   # Reduce long wrapping paths for compact visualization; preserve endpoints.
   if len(path)>18:path=[path[i] for i in np.unique(np.linspace(0,len(path)-1,18).astype(int))]
   paths.append(path)
  positions=arm.kinematics()
  frames.append({'time_s':float(t),'q_rad':q.tolist(),'bar_y_m':float(grip[1]),'bar_z_m':float(grip[2]),'bar_vy_m_s':float(vy),
    'bar_support_y_N':float(fy),'bar_support_z_N':float(fz),'grip':grip.tolist(),
    'joints':{k:v.round(5).tolist() for k,v in positions.items()},'paths':paths,
    'muscle_forces_N':force.round(4).tolist(),'reference_force_ratio':demand.round(5).tolist(),
    'reference_isometric_force_N':reference_force.round(4).tolist(),'passive_N':np.zeros_like(passive).tolist(),
    'required_joint_moments_Nm':tau.round(4).tolist(),'bounded_balance_residual_Nm':bounded_residual.round(4).tolist(),
    'bounded_solution_exists':feasible,'elbow_flexion_deg':float(np.rad2deg(q[3]))})
  allforce.append(force);allfiber.append(fiber)
  if index%30==0:print(f'Frame {index}/150 | max demand {max(demand):.2f} | feasible {feasible}',flush=True)
 forces=np.array(allforce);fiber=np.array(allfiber)
 impulses=np.zeros_like(forces);impulses[1:]=np.cumsum((forces[1:]+forces[:-1])/2*np.diff(times)[:,None],axis=0)
 # Mechanical fiber work is reported separately from force-time integral.
 fiber_velocity=np.gradient(fiber,times,axis=0)
 shortening=np.maximum(-fiber_velocity,0)*forces
 lengthening=np.maximum(fiber_velocity,0)*forces
 for index,frame in enumerate(frames):frame['muscle_impulse_Ns']=impulses[index].round(4).tolist()
 groups={}
 for key,(label,names) in GROUPS.items():
  ids=[arm.names.index(name) for name in names]
  f=forces[:,ids].sum(axis=1);imp=impulses[:,ids].sum(axis=1)
  groups[key]={'label':label,'muscles':names,'force_N':f.round(4).tolist(),'impulse_Ns':imp.round(4).tolist(),
               'peak_force_N':float(max(f)),'total_impulse_Ns':float(imp[-1]),'laterality':'one_arm'}
 summary={'model':'MoBL-ARMS 50 muscle compartments','source_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
 'method':'OpenSim 3D geometry and inverse dynamics; shoulder elevation + elbow flexion torque balance; minimum squared force/Fmax demand',
 'subject':config['subject'],'scale_factors':arm.scale_factors,'duration_s':2.5,'bar_mass_kg':100,
 'start_height_m':float(start[1]),'lockout_height_m':float(end[1]),'max_IK_error_m':ikerror,
 'max_sample_position_error_m':max_position_error,'max_equilibrium_error_Nm':maxresidual,
 'bounded_feasible_frame_fraction':sum(feasibility)/len(feasibility),'all_frames_bounded_feasible':all(feasibility),
 'force_interpretation':'Two-axis mechanical force demand, normalized by fixed original isometric Fmax. Not activation/EMG; above-reference demand is flagged. Not a full 3D equilibrium or a performance prediction.',
 'limitations':['Right arm solved, left mirrored with equal external load; no bar squeeze force.',
 'Original model Fmax unmodified. Body masses use 90kg male de Leva fractions; geometric dimensions prescribed.',
 'Shoulder girdle follows model rhythm as an externally supported moving base. Scapular virtual work is excluded from GH muscle demand. Trapezius/serratus forces and shoulder stability not estimated.',
 'Recruitment balances shoulder elevation and elbow flexion only. Out-of-plane shoulder, forearm rotation and wrist torques are externally supported.',
 'Fixed original isometric Fmax normalizes recruitment. Force-length-velocity and passive effects are not used in this first educational force allocation; no activation/fatigue/damage prediction.',
 'Visual muscle bellies are illustrative shapes along actual OpenSim paths, not segmented tissue meshes.',
 'Force-time integral is a loading measure, not hypertrophy-effective volume or predicted damage.'],
 'muscles':[{ 'name':m.getName(),'max_isometric_force_N':m.getMaxIsometricForce(),
              'peak_force_N':float(max(forces[:,i])),'impulse_Ns':float(impulses[-1,i])} for i,m in enumerate(arm.muscles)],
 'groups':{k:{a:b for a,b in v.items() if a not in ['force_N','impulse_Ns']} for k,v in groups.items()}}
 payload={'config':config,'summary':summary,'muscle_names':arm.names,'groups':groups,'frames':frames}
 (OUT/'muscle_motion.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 (OUT/'muscle_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 with (OUT/'muscle_forces.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.writer(f);writer.writerow(['time_s','bounded_feasible']+arm.names)
  for frame in frames:writer.writerow([frame['time_s'],frame['bounded_solution_exists']]+frame['muscle_forces_N'])
 assert max_position_error<.004
 assert np.isfinite(forces).all() and (forces>=-1e-6).all()
 assert maxresidual<.1
 print('PASS: muscle force balance, finite nonnegative forces, IK grip positions',flush=True)
 print('Bounded feasible fraction:',summary['bounded_feasible_frame_fraction'],flush=True)


if __name__=='__main__':main()

