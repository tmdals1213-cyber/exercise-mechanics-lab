"""Force traces and axis-specific opposing moments for two isolated OHP cases."""
from pathlib import Path
import json,csv
import numpy as np
from solve_fullbody import Body,OUT,CONFIG

GROUPS=[('delt_front','전면삼각근',['DELT1'],'shoulder_elv'),
 ('delt_middle','중간삼각근',['DELT2'],'shoulder_elv'),
 ('upper_pec','상부가슴',['PECM1'],'shoulder_elv'),
 ('triceps','삼두근',['TRI'],'elbow_flexion'),
 ('biceps','이두근',['BIC'],'elbow_flexion'),
 ('latissimus','광배근',['LAT'],'shoulder_elv'),
 ('rotator_cuff','회전근개',['SUPSP','INFSP','SUBSC','TMIN'],'shoulder_elv')]

def main():
 b=Body();m=b.m;data=np.load(OUT/'fullbody_motion.npz');meta=json.loads((OUT/'fullbody_metadata.json').read_text('utf-8'));metrics=np.load(OUT/'bar_joint_metrics.npz')
 names=meta['muscle_names'];times=data['times'];duration=meta.get('cycle_duration_s',2.);mask=times<=duration;forces=data['force_N'];s=b.length_scale
 traces={};group_metadata={};summaries={}
 for key,label,prefixes,joint in GROUPS:
  ids=[i for i,name in enumerate(names) if any(name.startswith(p) for p in prefixes)]
  traces[key+'_force_N']=forces[:,ids].sum(1);traces[key+'_opposing_torque_Nm']=[]
  traces[key+'_opposing_force_N']=[]
  if 'active_force_N' in data:
   traces[key+'_active_force_N']=data['active_force_N'][:,ids].sum(1)
   traces[key+'_passive_force_N']=data['passive_force_N'][:,ids].sum(1)
  group_metadata[key]={'label':label,'actuators':[names[i] for i in ids],'reference_coordinate':joint}
 traces['elbow_opposing_torque_Nm']=[]
 balance_error=0.
 for index,q in enumerate(data['qpos']):
  b.pose(q);R=b.reduction(q).T@b.moments()
  for key,label,prefixes,joint in GROUPS:
   torque=0.;opposed_force=0.
   for side,suffix in [('r',''),('l','_l')]:
    axis=meta['independent_coordinates'].index(joint+'_'+side);required=data['reduced_required_generalized_force'][index,axis]*s
    for aid,name in enumerate(names):
     if name not in group_metadata[key]['actuators']:continue
     if side=='l' and not name.endswith('_l'):continue
     if side=='r' and name.endswith('_l'):continue
     contribution=R[axis,aid]*forces[index,aid]*s
     if abs(required)>1e-5 and contribution*required<0:
      torque+=abs(contribution);opposed_force+=forces[index,aid]
   traces[key+'_opposing_torque_Nm'].append(torque);traces[key+'_opposing_force_N'].append(opposed_force)
  # All muscles crossing both elbow coordinates, not just biceps.
  opposing=0.
  for side in ['r','l']:
   axis=meta['independent_coordinates'].index('elbow_flexion_'+side);required=data['reduced_required_generalized_force'][index,axis]*s
   contributions=R[axis]*forces[index]*s
   positive=float(contributions[contributions>0].sum());negative=float(-contributions[contributions<0].sum())
   opposing+=min(positive,negative)
   error=abs(contributions.sum()-(required-data['reserve'][index,axis]*s));balance_error=max(balance_error,error)
  traces['elbow_opposing_torque_Nm'].append(opposing)
 for name in list(traces):traces[name]=np.array(traces[name])
 traces['shoulder_resultant_N']=np.linalg.norm(metrics['shoulder_reaction_world_N'][:,0],axis=1)
 traces['elbow_resultant_N']=np.linalg.norm(metrics['joint_reaction_world_N'][:,0],axis=1)
 traces['wrist_resultant_N']=np.linalg.norm(metrics['joint_reaction_world_N'][:,1],axis=1)
 for name,values in traces.items():
  assert np.isfinite(values).all() and values.min()>-1e-7,name
  summaries[name]={'mean_ascent':float(np.trapezoid(values[mask],times[mask])/duration),'max_ascent':float(values[mask].max())}
 for key,*_ in GROUPS:assert np.all(traces[key+'_opposing_force_N']<=traces[key+'_force_N']+1e-6)
 assert balance_error<.01,balance_error
 np.savez_compressed(OUT/'grip_traces.npz',times=times,**traces)
 report={'status':'passed','grip_width_cm':CONFIG['grip_width_m']*100,'force_model':meta.get('force_model','legacy_minimum_torque'),
  'strength_ceiling_enforced':False,'groups':group_metadata,'summaries':summaries,
  'statistics_interval_s':[0,duration],'mean_definition':'time integral over stated single-cycle interval divided by duration; display replays excluded',
  'antagonist_definition':'group force components oppose the required net moment of the stated coordinate; elbow opposing moment is min(sum flexor moments, sum extensor moment magnitudes), summed bilaterally',
  'max_elbow_moment_balance_error_Nm':balance_error,
  'joint_force_definition':'right-side transmitted resultant including allocated muscles; model estimate, not cartilage pressure',
  'limits':['No extra co-contraction or resistance was invented.','A muscle may oppose one coordinate while assisting another.','This minimum-effort allocation does not reproduce person-specific stabilizing co-contraction.','No strength ceiling; these prescribed-motion forces do not establish performance capability.']}
 (OUT/'grip_traces.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 with (OUT/'grip_traces.csv').open('w',encoding='utf-8-sig',newline='') as stream:
  writer=csv.writer(stream);writer.writerow(['time_s',*traces]);writer.writerows([[float(t),*[float(values[i]) for values in traces.values()]] for i,t in enumerate(times)])
 print('GRIP_TRACES_COMPLETE',CONFIG['grip_width_m'],flush=True)

if __name__=='__main__':main()
