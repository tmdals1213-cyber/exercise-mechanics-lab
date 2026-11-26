"""Conditional muscle tensions for unchanged native jump torques and poses.

The muscle-path model is analysis-only. It is never stepped and feeds no force
back into training. Passive floors use the standard lower-body/trunk paths;
unidentified arm/hand passive floors are omitted. No muscle strength ceiling.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np,mujoco
from controlled_broad_jump import build
from tension_reference import solve_history
from jump_monitor_data import ROOT,read,atomic
from tension_groups import REGISTRY,matches
OUT=ROOT/'deliverables/jump_rl_study_draft_v3'

def operator(m,d):
    R=np.zeros((m.nv,m.nu))
    for j in range(m.nu):
        a=int(d.moment_rowadr[j]);n=int(d.moment_rownnz[j]);R[d.moment_colind[a:a+n],j]=-d.actuator_moment[a:a+n]
    return R

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--output');ap.add_argument('--manifest');ap.add_argument('--case');args=ap.parse_args()
    if args.output:OUT=Path(args.output)
    OUT.mkdir(parents=True,exist_ok=True);manifest=read(Path(args.manifest)if args.manifest else ROOT/'deliverables/jump_rl_study_draft_v2/source_manifest.json');r,m,d,names,qa,da,_,reg=build(.000125,True,preserve_muscle_paths=True);muscles=[m.actuator(i).name for i in range(m.nu)];refs=m.actuator_gainprm[:,2].copy();assert np.all(refs>0);print('ANALYSIS_PATHS',m.nu,float(refs.min()),float(refs.max()),flush=True)
    groups=[dict(key=k,label=label,indices=[i for i,n in enumerate(muscles)if matches(n,prefix)])for k,label,prefix,_ in REGISTRY];groups=[g for g in groups if g['indices']]
    passive_ids=np.array([i for i,n in enumerate(muscles)if n.endswith(('_r','_l'))and not n.startswith(('LAT','DELT','TRI','BIC','PECM','SUPSP','INFSP','SUBSC','TMIN','TMAJ','CORB','BRA','BRD','ANC','FCR','FCU','ECR','ECU','FDS','FDP','EDC','OP','RI','LU_','UI_','EDM','EIP','EPL','EPB','FPL','APL','SUP','PT','PQ','PL'))],int)
    reports=[]
    for key in manifest['final_four_keys']:
        if args.case and key!=args.case:continue
        c=next(c for c in manifest['cases']if c['key']==key);folder=Path(c['folder']);z=np.load(folder/'motion.npz');rep=read(folder/'control_summary.json');p=rep['parameters'];native=mujoco.MjModel.from_binary_path(str(folder/'control_model.mjb'));assert (m.nq,m.nv)==(native.nq,native.nv)and np.array_equal(m.body_mass,native.body_mass);bottom=p['ready_s']+p['crouch_s']+p.get('prepush_lean_s',0.);ts=z['times'];grid=np.r_[np.arange(0,ts[-1],.04),ts[-1]];ids=set(int(np.argmin(abs(ts-t)))for t in grid);ids.update(np.flatnonzero(((ts>=bottom-.15)&(ts<=rep['takeoff_s']+.08))|((ts>=rep['touchdown_s']-.08)&(ts<=rep['touchdown_s']+.16))));ids=np.array(sorted(ids));R=[];passive=[];lengths=[];position_error=0.;fd_error=0.;root_error=0.
        for k,index in enumerate(ids):
            d.qpos[:]=z['qpos'][index];mujoco.mj_forward(m,d);position_error=max(position_error,float(abs(d.xpos-z['body_xpos'][index]).max()));A=operator(m,d);root_error=max(root_error,float(abs(A[:3]).max()));R.append(A[da]);L=d.actuator_length.copy();lengths.append(L);floor=np.zeros(m.nu)
            for j in passive_ids:floor[j]=max(0.,-mujoco.mju_muscleBias(float(L[j]),np.ascontiguousarray(m.actuator_lengthrange[j]),float(m.actuator_acc0[j]),np.ascontiguousarray(m.actuator_biasprm[j,:9])))
            passive.append(floor)
            if k in [0,len(ids)//3,len(ids)//2,len(ids)-1]:
                q=d.qpos.copy()
                for joint in range(len(qa)):
                    a=q.copy();b=q.copy();a[qa[joint]]+=1e-6;b[qa[joint]]-=1e-6;d.qpos[:]=a;mujoco.mj_forward(m,d);plus=d.actuator_length.copy();d.qpos[:]=b;mujoco.mj_forward(m,d);minus=d.actuator_length.copy();fd_error=max(fd_error,float(abs(-(plus-minus)/(2e-6)-A[da[joint]]).max()))
                d.qpos[:]=q;mujoco.mj_forward(m,d)
        assert position_error<1e-9 and root_error<1e-7 and fd_error<1e-4,(key,position_error,root_error,fd_error)
        R=np.array(R);passive=np.array(passive);target=z['joint_net_torque_Nm'][ids];times=ts[ids];solution=solve_history(times,R,target,np.ones_like(passive),passive,refs,temporal_weight=.0002,level_weight=1.,time_limit_s=180.);F=solution['force_N'];group_force=np.array([F[:,g['indices']].sum(1)/2 for g in groups]);primary={}
        for label,prefix in [('glutes',('glmax',)),('quads',('vas','recfem')),('calves',('gas','soleus'))]:
            indices=[i for i,n in enumerate(muscles)if n.endswith('_r')and n.startswith(prefix)];primary[label]=F[:,indices].sum(1)
        push=(times>=bottom)&(times<rep['takeoff_s']);report=dict(key=key,names=muscles,joint_names=names,groups=groups,primary_peaks_N={k:float(v[push].max())for k,v in primary.items()},checks=solution['checks'],position_error_m=position_error,root_virtual_work_error_m=root_error,finite_difference_moment_arm_error_m=fd_error,samples=len(times),source_motion=str(folder/'motion.npz'),source_sha256=hashlib.sha256((folder/'motion.npz').read_bytes()).hexdigest(),muscle_force_is_conditional=True,feeds_back_to_training=False,strength_ceiling=False,passive_policy='Native leg/trunk standard-model passive floors; arm/hand passive floors unidentified and omitted.',force_selection='Normalized force magnitude plus temporal regularization; one nonunique force-sharing solution, not EMG or metabolic energy.',path_profile=r.profile.get('published_knee_paths'))
        np.savez_compressed(OUT/f'{key}_muscles.npz',times=times,force_N=F,group_force_N=group_force,passive_force_N=passive,R=R,target=target,reference_N=refs,**primary);atomic(OUT/f'{key}_muscles.json',report);reports.append(report);print('MUSCLE_CASE',key,report['primary_peaks_N'],solution['checks']['moment_balance_max_error'],flush=True)
    atomic(OUT/'muscle_analysis.json',dict(cases={x['key']:x for x in reports},scope='Analysis-only force sharing on unchanged reduced mechanical poses and net torques; rigid-foot toe supports, out-of-plane supports and shoulder-girdle reactions remain unidentified.',human_muscle_forces_validated=False,relative_reference='Initial successful model push-off peak, per right-side muscle group.'))

if __name__=='__main__':main()
