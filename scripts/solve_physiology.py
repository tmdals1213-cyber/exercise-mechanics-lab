"""Required active+passive force, without a hard individual strength ceiling.

Native FLV gain weights force sharing. The unbounded effort coefficient is not
neural activation. An optional parameter-set capacity screen is audit-only and
does not establish the capacity of a real person.
"""
from pathlib import Path
import json,sys,math,shutil
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import osqp
from solve_fullbody import OUT,CONFIG


def problem(operators,index,mass,angle_deg=20.,activation_cap=np.inf,passive_enabled=True):
 R=operators['R'][index];C=operators['C'][index];gain=operators['gain_N'][index];passive=operators['passive_N'][index] if passive_enabled else np.zeros_like(gain)
 original_mass=CONFIG['bar_mass_kg'];target=operators['target'][index]+(mass-original_mass)*operators['bar_tau_per_kg'][index]
 nm=R.shape[1];nc=C.shape[1];n=nm+nc
 matrix=np.c_[R*gain[None,:],C];scale=np.maximum(np.linalg.norm(matrix,axis=1),20.)
 matrices=[sp.csc_matrix(matrix/scale[:,None])];lower=[(target-R@passive)/scale];upper=[lower[0].copy()]
 # Bounds are explicit; tangential foot forces can have either sign.
 bound=np.zeros((nm+8,n));bound[:nm,:nm]=np.eye(nm)
 for k in range(8):bound[nm+k,nm+3*k+2]=1
 matrices.append(sp.csc_matrix(bound));lower.append(np.zeros(nm+8));upper.append(np.r_[np.full(nm,activation_cap),np.full(8,np.inf)])
 friction=np.zeros((32,n))
 for k in range(8):
  for j in range(4):
   friction[k*4+j,nm+k*3+j//2]=1 if j%2==0 else -1
   friction[k*4+j,nm+k*3+2]=-CONFIG['ground_friction']
 matrices.append(sp.csc_matrix(friction));lower.append(np.full(32,-np.inf));upper.append(np.zeros(32))
 S=operators['shoulder_force_coeff'][index]
 offset=operators['shoulder_force_offset'][index]+(mass-original_mass)*operators['bar_force_per_kg'][index][None,:]/2
 cone_rows=[];cone_rhs=[]
 if angle_deg is not None:
  apothem=math.tan(math.radians(angle_deg))*math.cos(math.pi/16)
  for side in range(2):
   normal=operators['cone_axis'][index,side];reference=np.array([0.,0,1.])
   if abs(normal@reference)>.9:reference=np.array([0.,1,0])
   tangent=np.cross(normal,reference);tangent/=np.linalg.norm(tangent);tangent2=np.cross(normal,tangent)
   directions=[-normal]+[math.cos(phi)*tangent+math.sin(phi)*tangent2-apothem*normal for phi in np.linspace(0,2*math.pi,16,endpoint=False)]
   for direction in directions:
    row=np.zeros(n);row[:nm]=direction@(-S[side]*gain[None,:]);rhs=-direction@(offset[side]-S[side]@passive)
    factor=max(np.linalg.norm(row),1.);cone_rows.append(row/factor);cone_rhs.append(rhs/factor)
  matrices.append(sp.csc_matrix(np.array(cone_rows)));lower.append(np.full(len(cone_rows),-np.inf));upper.append(np.array(cone_rhs))
 A=sp.vstack(matrices,format='csc');lo=np.concatenate(lower);hi=np.concatenate(upper)
 P=sp.diags(np.r_[np.ones(nm),np.full(nc,.002)],format='csc')
 return {'A':A,'lo':lo,'hi':hi,'P':P,'gain':gain,'passive':passive,'target':target,'R':R,'C':C,'S':S,'offset':offset,'nm':nm,'n':n,'cone_rows':cone_rows}


def solve_frame(operators,index,mass,angle_deg=20.,activation_cap=np.inf,passive_enabled=True,confirm_infeasible=False):
 p=problem(operators,index,mass,angle_deg,activation_cap,passive_enabled)
 solver=osqp.OSQP();solver.setup(P=p['P'],q=np.zeros(p['n']),A=p['A'],l=p['lo'],u=p['hi'],verbose=False,
  eps_abs=1e-7,eps_rel=1e-7,max_iter=100000,polishing=True)
 result=solver.solve(raise_error=False)
 status=result.info.status;report={'frame':int(index),'time_s':float(operators['times'][index]),'bar_mass_kg':mass,
  'cone_half_angle_deg':angle_deg,'activation_cap':None if np.isinf(activation_cap) else activation_cap,
  'passive_enabled':passive_enabled,'status':status,'feasible':False}
 solution=None
 if result.x is not None and status.startswith('solved'):
  x=result.x;a=np.maximum(x[:p['nm']],0);force=p['passive']+p['gain']*a;contacts=x[p['nm']:].reshape(8,3)*1000
  error=float(abs(p['R']@force+p['C']@(contacts.ravel()/1000)-p['target']).max())
  reaction=p['offset']-p['S']@force;angles=[];compression=[]
  for side in range(2):
   normal=operators['cone_axis'][index,side];normal_force=float(reaction[side]@normal);magnitude=float(np.linalg.norm(reaction[side]));compression.append(normal_force)
   angles.append(float(math.degrees(math.acos(np.clip(normal_force/max(magnitude,1e-12),-1,1)))))
  max_violation=float(np.maximum(p['A']@x-p['hi'],p['lo']-p['A']@x).max())
  feasible=error<.01 and max_violation<1e-5 and (activation_cap==np.inf or a.max()<activation_cap+1e-5)
  if angle_deg is not None:feasible=feasible and min(compression)>-1e-5 and max(angles)<=angle_deg+.01
  report.update({'feasible':bool(feasible),'max_activation_or_effort':float(a.max()),'force_balance_error':error,
   'max_scaled_constraint_violation':max_violation,'shoulder_reaction_angle_deg':angles,'shoulder_compression_N':compression})
  if feasible:solution={'activation_or_effort':a,'force_N':force,'active_force_N':p['gain']*a,'passive_force_N':p['passive'],'contacts_N':contacts,'shoulder_reaction_world_N':reaction}
 if confirm_infeasible and not report['feasible']:
  equal=np.isfinite(p['lo'])&np.isfinite(p['hi'])&(abs(p['lo']-p['hi'])<1e-12)
  upper_rows=np.isfinite(p['hi'])&~equal;lower_rows=np.isfinite(p['lo'])&~equal
  Aub=sp.vstack([p['A'][upper_rows],-p['A'][lower_rows]],format='csc');bub=np.r_[p['hi'][upper_rows],-p['lo'][lower_rows]]
  check=linprog(np.zeros(p['n']),A_ub=Aub,b_ub=bub,A_eq=p['A'][equal],b_eq=p['lo'][equal],bounds=[(None,None)]*p['n'],method='highs')
  report['independent_linprog_status']=int(check.status);report['independent_linprog_message']=check.message
  if check.success:
   report['qp_numerical_issue_not_physical_infeasibility']=True
 return report,solution


def main():
 operators=np.load(OUT/'physiology_operators.npz');reports=[]
 if '--capacity-screen' not in sys.argv:
  reports=[];solutions=[]
  for i in range(len(operators['times'])):
   report,solution=solve_frame(operators,i,CONFIG['bar_mass_kg'],20.,np.inf,True,True)
   if solution is None:raise RuntimeError('Required-force equilibrium failed: '+str(report))
   reports.append(report);solutions.append(solution)
  legacy=np.load(OUT/'fullbody_motion.npz');values={k:legacy[k] for k in legacy.files};force=np.array([s['force_N'] for s in solutions])
  meta=json.loads((OUT/'fullbody_metadata.json').read_text('utf-8'));reference=np.array(meta['reference_force_N']);times=values['times']
  impulse=np.zeros_like(force);impulse[1:]=np.cumsum((force[1:]+force[:-1])/2*np.diff(times)[:,None],axis=0)
  values.update(force_N=force,active_force_N=np.array([s['active_force_N'] for s in solutions]),passive_force_N=np.array([s['passive_force_N'] for s in solutions]),
   normalized_active_effort=np.array([s['activation_or_effort'] for s in solutions]),reference_ratio=force/reference,impulse_Ns=impulse,
   contacts_N=np.array([s['contacts_N'] for s in solutions]),reserve=np.zeros_like(legacy['reserve']))
  directory=OUT/'physiology';directory.mkdir(exist_ok=True)
  np.savez_compressed(directory/'fullbody_motion.npz',**values)
  meta.update(method='required active+passive muscle force with native force-length-velocity efficiency; GH directional stability, no hard strength ceiling',
   force_model='active_passive_gh20_required_force',normalized_effort_definition='unbounded force-allocation coefficient; not a measured or bounded neural activation',
   strength_ceiling_enforced=False,gh_cone_half_angle_deg=20.,max_balance_error=max(r['force_balance_error'] for r in reports),max_nonroot_reserve=0.,
   limitations=['Native muscle law parameters retained, not individually calibrated.','No hard muscle-strength or neural activation ceiling; normalized effort is not a firing-rate estimate.',
    'GH force-direction stability is a 20-degree generic glenoid cone approximation, not every neural stabilization requirement.',
    'No explicit ligament/capsule material law, and no activation dynamics imposed on unbounded required-effort coefficients.',
    'Generic geometry and force-sharing cost do not uniquely predict an individual muscle recruitment pattern.'])
  (directory/'fullbody_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
  for name in ['bone_meshes.json','anatomy_landmarks.json']:shutil.copyfile(OUT/name,directory/name)
  reaction=np.array([s['shoulder_reaction_world_N'] for s in solutions]);angles=np.array([r['shoulder_reaction_angle_deg'] for r in reports])
  np.savez_compressed(directory/'physiology_solution.npz',times=times,force_N=force,active_force_N=values['active_force_N'],passive_force_N=values['passive_force_N'],
   normalized_active_effort=values['normalized_active_effort'],shoulder_reaction_world_N=reaction,gh_reaction_angle_deg=angles)
  summary={'status':'passed','model':'required force, active+passive FLV, GH stability','grip_width_cm':CONFIG['grip_width_m']*100,
   'strength_ceiling_enforced':False,'no_neural_activation_claim':True,'gh_cone_half_angle_deg':20.,'gh_max_reaction_angle_deg':float(angles.max()),
   'max_force_balance_error':meta['max_balance_error'],'minimum_active_force_N':float(values['active_force_N'].min()),
   'max_normalized_active_effort':float(values['normalized_active_effort'].max()),'frames':reports,
   'geometry_note':'Estimated generic glenoid centre from native scapular mesh; not patient-specific contact geometry',
   'stability_source':'https://doi.org/10.1371/journal.pone.0295003'}
  (directory/'physiology_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
  print('REQUIRED_PHYSIOLOGY_SOLUTION_COMPLETE',CONFIG['grip_width_m'],'max GH angle',angles.max(),flush=True)
  return
 for mass in [0.,20.,100.]:
  for frame in [0,15,30,45,60]:
   for angle in [None,20.]:
    report,solution=solve_frame(operators,frame,mass,angle,1.,True,True);reports.append(report)
    print('PHYS_CAPACITY',mass,'kg',frame,'cone',angle,report['status'],'LP',report.get('independent_linprog_status'),'max a',report.get('max_activation_or_effort'),flush=True)
 report={'status':'screening_complete','grip_width_cm':CONFIG['grip_width_m']*100,'muscle_law':'original MuJoCo active force-length-velocity plus passive force',
  'strength_scale':1.,'stability_half_angle_deg':20.,'stability_source':'https://doi.org/10.1371/journal.pone.0295003',
  'interpretation':'Any infeasible sampled frame rules out that prescribed path under these parameters. Feasibility of sampled frames is not full-trajectory or experimental validation.',
  'cases':reports}
 (OUT/'physiology_feasibility_screen.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
