"""Uniform-proportion OHP comparison at matched angles and timing, 3 mass policies.

Native-coordinate virtual work cancels a common length factor from muscle,
ground and bar moment arms. Body inertial terms retain one length factor.
Reference capacity uses fixed muscle mass fraction and fibre length ~ stature:
PCSA ratio = mass ratio / length ratio. No physiological prediction is claimed.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from solve_fullbody import Body,OUT,CONFIG,profile,allocate,NATIVE_HEIGHT_M

HEIGHTS=[1.65,1.80,1.95]
POLICIES={'geometric':'동일 비례 체형','same_bmi':'동일 BMI','same_mass':'동일 체중'}

def main():
 body=Body();m,d=body.m,body.d;data=np.load(OUT/'fullbody_motion.npz')
 meta=json.loads((OUT/'fullbody_metadata.json').read_text('utf-8'))
 base_height=meta['target_height_m'];base_mass=meta['subject_mass_kg']
 length_scale=base_height/NATIVE_HEIGHT_M;fps=CONFIG['fps'];times=data['times'];qs=data['qpos']
 velocity=np.zeros((len(times),m.nv))
 for i in range(1,len(times)-1):mujoco.mj_differentiatePos(m,velocity[i],2/fps,qs[i-1],qs[i+1])
 acceleration=np.gradient(velocity,1/fps,axis=0)
 velocity[0]=0;velocity[times>=2]=0;acceleration[0]=0;acceleration[times>=2]=0
 matrices=[];base_grips=data['physical_grips'];rise=base_grips[-1,:,2].mean()-base_grips[0,:,2].mean()
 shoulder=6+body.ind_names.index('shoulder_elv_r')
 for i,t in enumerate(times):
  body.pose(qs[i]);d.qvel[:]=velocity[i];mujoco.mj_forward(m,d);d.qacc[:]=acceleration[i]
  full=np.zeros(m.nv);mujoco.mj_rne(m,d,1,full)
  gravity=m.opt.gravity.copy();m.opt.gravity[:]=0
  inertia=np.zeros(m.nv);mujoco.mj_rne(m,d,1,inertia);m.opt.gravity[:]=gravity
  B=body.reduction(qs[i]);R=B.T@body.moments()
  C=np.concatenate([B.T@body.jacobian(bid,local)[0].T*1000 for bid,local in body.sole],axis=1)
  _,v,a=profile(float(t));progress=profile(float(t))[0]
  bar_a=np.array([0,-2*CONFIG['bar_start_forward_m']*(v*v-(1-progress)*a),a*rise])
  J=sum((body.jacobian(body.grip_body[side],body.grip_local)[0] for side in ['r','l']))/2
  bar_gravity=B.T@J.T@np.array([0,0,CONFIG['bar_mass_kg']*9.80665])
  bar_inertia=B.T@J.T@(CONFIG['bar_mass_kg']*bar_a)
  matrices.append((R,C,B.T@(full-inertia),B.T@inertia,bar_gravity,bar_inertia))
 result=[];force_arrays={};reference={}
 for policy,label in POLICIES.items():
  for height in HEIGHTS:
   relative=height/base_height;scale=height/NATIVE_HEIGHT_M
   mass_ratio=relative**(3 if policy=='geometric' else 2 if policy=='same_bmi' else 0)
   mass=base_mass*mass_ratio;capacity_ratio=mass_ratio/relative;f0=body.f0*capacity_ratio
   forces=[];reserves=[];errors=[];bar_moments=[]
   for R,C,grav,inertia,bar_grav,bar_inertia in matrices:
    target=mass_ratio*(grav+scale*inertia)+bar_grav+relative*bar_inertia
    force,ratio,contacts,reserve,error,_=allocate(R*f0[None,:],C,target,None,f0)
    forces.append(force);reserves.append(np.abs(reserve[6:]).max());errors.append(error)
    bar_moments.append(abs(bar_grav[shoulder])*scale)
   force=np.array(forces);key=f'{policy}_{round(height*100)}';force_arrays[key]=force;reference[key]=f0
   tri=[i for i,n in enumerate(body.names) if n.startswith('TRI')]
   delt=[i for i,n in enumerate(body.names) if n.startswith('DELT')]
   item={'key':key,'policy':policy,'label':label,'height_cm':height*100,'mass_kg':mass,'bmi':mass/height**2,
    'relative_length':relative,'relative_reference_strength':capacity_ratio,
    'assumed_muscle_cross_section_linear_ratio':float(np.sqrt(capacity_ratio)),
    'bar_travel_m':rise*relative,'bar_gravitational_work_J':CONFIG['bar_mass_kg']*9.80665*rise*relative,
    'peak_bar_only_shoulder_elv_generalized_moment_Nm':max(bar_moments),
    'peak_deltoid_sum_N':float(force[:,delt].sum(1).max()),'peak_triceps_sum_N':float(force[:,tri].sum(1).max()),
    'peak_single_actuator_reference_ratio':float((force/f0).max()),
    'max_balance_error':max(errors),'max_nonroot_reserve_equivalent':max(reserves)}
   result.append(item);print(key,'mass',round(mass,2),'bar work',round(item['bar_gravitational_work_J'],1),'solved',flush=True)
 np.savez_compressed(OUT/'height_comparison_forces.npz',times=times,**force_arrays,**{k+'_reference':v for k,v in reference.items()})
 report={'heights_cm':[165,180,195],'reference_height_cm':180,'reference_mass_kg':90,
  'bar_mass_kg':CONFIG['bar_mass_kg'],'ascent_s':2,'method':'native-coordinate virtual-work uniform similarity; 76 inverse-dynamics allocations per case',
  'controls':['same segment length proportions','same generalized joint trajectories','same two-second timing','grip width and bar path proportional to height','same 100 kg external load'],
  'capacity_assumption':'constant muscle mass fraction and fibre length proportional to height, Fref ratio = mass ratio / height ratio; assumed PCSA, not measured',
  'limits':['BMI and constant-mass cases change girth and reference capacity assumptions; internal moment-arm geometry remains uniformly similar.',
   'No force-length-velocity, fatigue, individual co-contraction or performance ceiling; over-capacity prescribed movements remain solved.',
   'Shoulder metric is the bar-only generalized moment for right shoulder_elv, not a measured three-axis joint resultant.',
   'Bar work is net gravitational work on the bar alone, excluding body-segment work and metabolic cost.'],
  'cases':result}
 (OUT/'height_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('HEIGHT_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__':main()
