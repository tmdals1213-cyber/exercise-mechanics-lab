"""Audit muscle operating ranges and unbounded force feasibility, not recruitment.

No strength ceiling, reserve actuator, or invented co-contraction is introduced.
An infeasible frame diagnoses this model / motion / passive-law combination.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco,myo_sim
from scipy.optimize import linprog
from jump_rig import JumpRig,OUT,ROOT

GROUPS={'quadriceps':('vas','recfem'),'hamstrings':('bflh','bfsh','semimem','semiten'),'gluteals':('glmax','glmed','glmin'),'calf':('gas','soleus'),'hip_flexors':('psoas','iliacus'),'trunk':('rect_abd','EO','IO','IL_','LTp'),'deltoid':('DELT',),'triceps':('TRI',),'biceps':('BIC',),'latissimus':('LAT',),'rotator_cuff':('SUPSP','INFSP','SUBSC','TMIN')}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--folder',default='vertical/acceleration_2400hz');ap.add_argument('--samples',type=int,default=81);a=ap.parse_args();folder=OUT/a.folder
 z=dict(np.load(folder/'motion.npz'));loads=dict(np.load(folder/'loads.npz'));rig=JumpRig();b=rig.b;m,d=rig.m,rig.d
 indices=np.unique(np.linspace(0,len(z['times'])-1,a.samples).round().astype(int));arrays={k:[] for k in ['times','length_m','velocity_m_s','normalized_fiber_length_surrogate','active_gain_N','passive_N','moment_arms_m','net_demand_Nm']};frames=[]
 lr=m.actuator_lengthrange;gp=m.actuator_gainprm;bp=m.actuator_biasprm;L0=(lr[:,1]-lr[:,0])/(gp[:,1]-gp[:,0]);LT=lr[:,0]-gp[:,0]*L0
 for index in indices:
  q=z['qpos'][index];v=z['qvel'][index];b.pose(q,v,z['qacc'][index]);B,_=rig.coordinates(q,z['u'][index]);R=(B.T@b.moments())[6:];length=d.actuator_length.copy();speed=d.actuator_velocity.copy();gain=[];passive=[]
  for j in range(m.nu):
   gain.append(max(0.,-mujoco.mju_muscleGain(float(length[j]),float(speed[j]),np.ascontiguousarray(lr[j]),float(m.actuator_acc0[j]),np.ascontiguousarray(gp[j,:9]))))
   passive.append(max(0.,-mujoco.mju_muscleBias(float(length[j]),np.ascontiguousarray(lr[j]),float(m.actuator_acc0[j]),np.ascontiguousarray(bp[j,:9]))))
  gain=np.array(gain);passive=np.array(passive);target=loads['net_joint_moment_Nm'][index];A=R*gain;rhs=target-R@passive;scale=np.maximum(np.linalg.norm(A,axis=1),1.)
  lp=linprog(np.ones(m.nu),A_eq=A/scale[:,None],b_eq=rhs/scale,bounds=(0,None),method='highs')
  residual=float(abs(A@lp.x-rhs).max()) if lp.success else None
  frames.append({'index':int(index),'time_s':float(z['times'][index]),'LP_feasible':bool(lp.success and residual<.01),'LP_status':int(lp.status),'physical_residual_Nm':residual,'zero_active_gain_elements':int((gain<1e-8).sum()),'minimum_inferred_fiber_length_ratio':float(((length-LT)/L0).min())})
  values={'times':z['times'][index],'length_m':length,'velocity_m_s':speed,'normalized_fiber_length_surrogate':(length-LT)/L0,'active_gain_N':gain,'passive_N':passive,'moment_arms_m':R,'net_demand_Nm':target}
  for key,value in values.items():arrays[key].append(value)
 arrays={key:np.asarray(value) for key,value in arrays.items()};groups={}
 for name,prefixes in GROUPS.items():
  ix=[i for i,n in enumerate(b.names) if n.startswith(prefixes)]
  groups[name]={'elements':len(ix),'zero_gain_at_any_sample':[b.names[i] for i in ix if (arrays['active_gain_N'][:,i]<1e-8).any()],'zero_gain_at_all_samples':[b.names[i] for i in ix if (arrays['active_gain_N'][:,i]<1e-8).all()],'inferred_fiber_length_ratio_range':[float(arrays['normalized_fiber_length_surrogate'][:,ix].min()),float(arrays['normalized_fiber_length_surrogate'][:,ix].max())] if ix else None,'maximum_sum_of_imported_passive_forces_N':float(arrays['passive_N'][:,ix].sum(1).max())}
 nonpositive=np.where((arrays['normalized_fiber_length_surrogate']<=0).any(0))[0];bad_lengths=[]
 for j in nonpositive:
  k=int(np.argmin(arrays['normalized_fiber_length_surrogate'][:,j]));bad_lengths.append({'element':b.names[j],'minimum_ratio':float(arrays['normalized_fiber_length_surrogate'][k,j]),'time_s':float(arrays['times'][k]),'MT_length_m':float(arrays['length_m'][k,j]),'inferred_L0_m':float(L0[j]),'inferred_rigid_tendon_length_m':float(LT[j])})
 from parametric_body import complete_coordinates
 native=myo_sim.load_spec('myofullbody').compile();nd=mujoco.MjData(native);nd.qpos[:]=complete_coordinates(native,native.qpos0);mujoco.mj_forward(native,nd);native_check=[]
 for entry in bad_lengths:
  j=native.actuator(entry['element']).id;nlr=native.actuator_lengthrange[j];ng=native.actuator_gainprm[j];nl0=(nlr[1]-nlr[0])/(ng[1]-ng[0]);nlt=nlr[0]-ng[0]*nl0
  native_check.append({'element':entry['element'],'unmodified_native_completed_neutral_fiber_ratio':float((nd.actuator_length[j]-nlt)/nl0),'negative_mapping_predates_subject_scaling':bool(nd.actuator_length[j]<nlt)})
 ambiguity=[]
 for requested_t in [.55,.92,1.58]:
  k=int(np.argmin(abs(arrays['times']-requested_t)));R=arrays['moment_arms_m'][k];gain=arrays['active_gain_N'][k];passive=arrays['passive_N'][k];A=R*gain;rhs=arrays['net_demand_Nm'][k]-R@passive;scale=np.maximum(np.linalg.norm(A,axis=1),1.)
  for group in ['quadriceps','hamstrings','latissimus']:
   ix=[i for i,n in enumerate(b.names) if n.startswith(GROUPS[group])];objective=np.zeros(m.nu);objective[ix]=gain[ix];factor=max(float(abs(objective).max()),1.)
   minimum=linprog(objective/factor,A_eq=A/scale[:,None],b_eq=rhs/scale,bounds=(0,None),method='highs');maximum=linprog(-objective/factor,A_eq=A/scale[:,None],b_eq=rhs/scale,bounds=(0,None),method='highs')
   ambiguity.append({'time_s':float(arrays['times'][k]),'group':group,'minimum_total_force_N_under_current_equations':float(objective@minimum.x+passive[ix].sum()) if minimum.success else None,'maximum_force_unbounded_under_current_equations':maximum.status==3,'maximum_LP_status':int(maximum.status),'interpretation':'Algebraic range only; no strength ceiling, GH stability or neural strategy. Unbounded means underdetermined equations, not infinite biological force.'})
 report={'status':'diagnostic_only_not_a_muscle_force_prediction','trajectory':a.folder,'sample_count':len(indices),'model_elements':m.nu,'missing_major_groups':b.graph()['missing_required_major_groups'],'all_sampled_frames_force_feasible':all(f['LP_feasible'] for f in frames),'infeasible_sample_count':sum(not f['LP_feasible'] for f in frames),'strength_ceiling_enforced':False,'zero_gain_count_any':int((arrays['active_gain_N']<1e-8).any(0).sum()),'zero_gain_count_all':int((arrays['active_gain_N']<1e-8).all(0).sum()),'nonpositive_inferred_fiber_elements':bad_lengths,'unmodified_native_mapping_crosscheck':native_check,'biological_operating_range_gate_passed':len(bad_lengths)==0,'muscle_force_release':False,'groups':groups,'frames':frames,'muscle_names':b.names,'definitions':{'force_law':'F_passive(length) + gain(length,velocity)*e, e>=0 with no upper bound','inferred_fiber_length':'MuJoCo rigid-tendon / zero-pennation surrogate, not imaged fascicle length','feasibility':'Existence of nonnegative unbounded effort satisfying current net moments and fixed imported passive force. LP objective is diagnostic; its solution is not a recruitment prediction.','zero_gain':'Model operating-range or shortening-speed issue, not proof of no human muscle activity.'},'limitations':['Jump motion realism is not validated.','No GH stability or prescribed co-contraction in this diagnostic LP.','No compliant tendon, activation time history, fatigue, tissue damage or measured individual muscle architecture.','Sampled feasibility does not validate unsampled frames.'],'trajectory_sha256':hashlib.sha256((folder/'motion.npz').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 report['force_sharing_ambiguity']=ambiguity
 np.savez_compressed(folder/'muscle_operating_audit.npz',**arrays);(folder/'muscle_operating_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['status','sample_count','model_elements','missing_major_groups','infeasible_sample_count','zero_gain_count_any','zero_gain_count_all']}))

if __name__=='__main__':main()
