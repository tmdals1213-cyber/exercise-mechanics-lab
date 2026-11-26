"""Full displayed force sharing: native paths plus thoracoscapular support.

The original14-DOF motion and11 net torque demands remain unchanged. Ten
analysis-only SC/AC virtual-rotation balances recover isometric girdle support
that cannot be inferred from frozen girdle coordinate motion alone.
"""
from pathlib import Path
from types import SimpleNamespace
import copy,json,hashlib,argparse
import numpy as np,mujoco
from controlled_broad_jump import build
from conservative_ellipsoid_paths import ConservativeEllipsoidPaths
from muscle_mechanics import subtree
from mechanics_engine_adapter import sample
from tension_reference import solve_history
from jump_monitor_data import ROOT,read,atomic
OUT=ROOT/'deliverables/jump_rl_study_draft_v4';OLD=ROOT/'deliverables/jump_rl_study_draft_v3'

def cut_projection(member,axes,anchors,W):
    sums=np.einsum('qb,bcn->qcn',member,W);F=sums[:,:3];T=sums[:,3:]-np.cross(anchors[:,None,:],F.transpose(0,2,1)).transpose(0,2,1);return np.einsum('qc,qcn->qn',axes,T)

def native_support(m,d,relevant,member,axes,anchors):
    S=np.zeros((len(axes),m.nu));pts=d.wrap_xpos.reshape(-1,3);objects=d.wrap_obj.ravel()
    for act in relevant:
        tid=int(m.actuator_trnid[act,0]);p=int(d.ten_wrapadr[tid]);end=p+int(d.ten_wrapnum[tid]);previous=None
        for j in range(int(m.tendon_adr[tid]),int(m.tendon_adr[tid]+m.tendon_num[tid])):
            kind=int(m.wrap_type[j]);obj=int(m.wrap_objid[j]);count=1 if kind==int(mujoco.mjtWrap.mjWRAP_SITE)else 2 if p<end and objects[p]==obj else 0;bid=int(m.site_bodyid[obj])if kind==int(mujoco.mjtWrap.mjWRAP_SITE)else int(m.geom_bodyid[obj])
            for _ in range(count):
                point=pts[p]
                if previous is not None and previous[0]!=bid:
                    other,prior=previous;delta=point-prior;L=np.linalg.norm(delta)
                    if L>1e-12:
                        force=m.actuator_gear[act,0]*delta/L;one=member[:,other]&~member[:,bid];two=member[:,bid]&~member[:,other];S[:,act]+=one*np.sum(axes*np.cross(prior-anchors,force),axis=1)-two*np.sum(axes*np.cross(point-anchors,force),axis=1)
                previous=(bid,point);p+=1
        assert p==end
    return S

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--case');ap.add_argument('--output');ap.add_argument('--manifest');ap.add_argument('--native-analysis');args=ap.parse_args()
    if args.output:OUT=Path(args.output)
    native_analysis=Path(args.native_analysis)if args.native_analysis else OLD
    OUT.mkdir(parents=True,exist_ok=True);manifest=read(Path(args.manifest)if args.manifest else OLD/'source_manifest.json');original,m,d,names,qa,da,_,reg=build(.000125,True,preserve_muscle_paths=True);names_native=[m.actuator(i).name for i in range(m.nu)];definitions=[]
    for side in['r','l']:
        for stem in['sternoclavicular_r2','sternoclavicular_r3','acromioclavicular_r1','acromioclavicular_r2','acromioclavicular_r3']:
            j=original.m.joint(stem+'_'+side).id;bid=int(original.m.jnt_bodyid[j]);R=original.d.xmat[bid].reshape(3,3);definitions.append(dict(name=stem+'_'+side,body=bid,axis_local=R.T@original.d.xaxis[j],anchor_local=R.T@(original.d.xanchor[j]-original.d.xpos[bid])))
    member=np.zeros((10,m.nbody),bool)
    for j,v in enumerate(definitions):member[j,subtree(m,v['body'])]=True
    relevant=[]
    for a in range(m.nu):
        tid=int(m.actuator_trnid[a,0]);bids=[]
        for j in range(int(m.tendon_adr[tid]),int(m.tendon_adr[tid]+m.tendon_num[tid])):
            kind=int(m.wrap_type[j]);obj=int(m.wrap_objid[j]);bids.append(int(m.site_bodyid[obj])if kind==int(mujoco.mjtWrap.mjWRAP_SITE)else int(m.geom_bodyid[obj]))
        if np.any(np.ptp(member[:,bids].astype(int),axis=1)):relevant.append(a)
    def pose(q,v=None,a=None):
        d.qpos[:]=q;d.qvel[:]=0 if v is None else v;mujoco.mj_forward(m,d);d.qacc[:]=0 if a is None else a
    proxy=SimpleNamespace(m=m,d=d,b=SimpleNamespace(pose=pose,q0=d.qpos.copy(),report=original.b.report));bridge=ConservativeEllipsoidPaths(proxy);bridge.operators();W0=bridge.unit_wrenches.copy();C0=d.xmat[m.body('chest_r').id].reshape(3,3).copy();p0=d.xpos[m.body('chest_r').id].copy();body_rel={bid:(C0.T@(d.xpos[bid]-p0),C0.T@d.xmat[bid].reshape(3,3))for bid in set(bridge.parts.values())};muscles=names_native+bridge.names;all_names=muscles+[f'scapular_contact_{i}'for i in range(4)];refs=np.r_[m.actuator_gainprm[:,2],bridge.refs,np.full(4,500.)];reports={}
    print('FULLBODY_SUPPORT',m.nu,len(bridge.names),len(relevant),len(refs),flush=True)
    for key in manifest['final_four_keys']:
        if args.case and args.case!=key:continue
        c=next(c for c in manifest['cases']if c['key']==key);f=Path(c['folder']);motion=np.load(f/'motion.npz');old=np.load(native_analysis/f'{key}_muscles.npz');ts=old['times'];indices=np.array([int(np.argmin(abs(motion['times']-t)))for t in ts]);R=[];target=[];passive=[];max_ne=0.;max_geometry=0.;max_pose=0.;root_closure=0.;contact_gaps=[]
        for k,index in enumerate(indices):
            q,v,acc=motion['qpos'][index],motion['qvel'][index],motion['qacc'][index];mechanics=sample(proxy,q,v,acc);max_ne=max(max_ne,mechanics['crosscheck_max']);max_pose=max(max_pose,float(abs(d.xpos-motion['body_xpos'][index]).max()));C=d.xmat[m.body('chest_r').id].reshape(3,3);pc=d.xpos[m.body('chest_r').id];rot=C@C0.T;trans=pc-rot@p0;W=np.zeros_like(W0);W[:,:3]=np.einsum('ij,bjn->bin',rot,W0[:,:3]);W[:,3:]=np.einsum('ij,bjn->bin',rot,W0[:,3:])+np.cross(trans,W[:,:3].transpose(0,2,1)).transpose(0,2,1)
            for bid,(x,B)in body_rel.items():max_geometry=max(max_geometry,float(abs(C.T@(d.xpos[bid]-pc)-x).max()),float(abs(C.T@d.xmat[bid].reshape(3,3)-B).max()))
            root_closure=max(root_closure,float(abs(W.sum(0)).max()));axes=np.array([d.xmat[x['body']].reshape(3,3)@x['axis_local']for x in definitions]);anchors=np.array([d.xpos[x['body']]+d.xmat[x['body']].reshape(3,3)@x['anchor_local']for x in definitions]);Sn=native_support(m,d,relevant,member,axes,anchors);Sg=cut_projection(member,axes,anchors,W);Sc=np.zeros((10,4))
            for j,point in enumerate(original.contacts):
                bid=point['body'];cb=point['chest'];Rc=d.xmat[cb].reshape(3,3);world=d.xpos[bid]+d.xmat[bid].reshape(3,3)@point['local'];center=d.xpos[cb]+Rc@point['centre'];Q=Rc@point['rotation'];x=Q.T@(world-center);grad=Q@(2*x/point['radii']**2);normal=grad/np.linalg.norm(grad);signed=member[:,bid].astype(float)-member[:,cb].astype(float);Sc[:,j]=signed*np.sum(axes*np.cross(world-anchors,normal),axis=1);contact_gaps.append(float(np.sum((x/point['radii'])**2)-1))
            own=mechanics['body_wrenches'];sums=member@own;support=np.sum(axes*(sums[:,3:]-np.cross(anchors,sums[:,:3])),axis=1);A=np.zeros((21,len(refs)));A[:11,:m.nu]=old['R'][k];A[11:]=np.c_[Sn,Sg,Sc];R.append(A);target.append(np.r_[old['target'][k],support]);passive.append(np.r_[old['passive_force_N'][k],np.zeros(len(bridge.names)+4)])
        assert max_ne<1e-5 and max_pose<1e-9 and max_geometry<1e-8 and root_closure<1e-7,(key,max_ne,max_pose,max_geometry,root_closure)
        R=np.array(R);target=np.array(target);passive=np.array(passive);result=solve_history(ts,R,target,np.ones_like(passive),passive,refs,temporal_weight=.0002,level_weight=1.,time_limit_s=240.);F=result['force_N'];p=read(f/'control_summary.json');pr=p['parameters'];push=(ts>=pr['ready_s']+pr['crouch_s']+pr.get('prepush_lean_s',0))&(ts<p['takeoff_s']);primary={}
        for group,prefix in [('glutes',('glmax',)),('quads',('vas','recfem')),('hamstrings',('bflh','bfsh','semimem','semiten')),('calves',('gas','soleus'))]:primary[group]=F[:,[i for i,n in enumerate(muscles)if n.endswith('_r')and n.startswith(prefix)]].sum(1)
        report=dict(key=key,names=muscles,all_force_names=all_names,muscle_count=len(muscles),primary_peaks_N={g:float(values[push].max())for g,values in primary.items()},checks=result['checks'],NE_crosscheck_max_SI=max_ne,position_error_m=max_pose,girdle_rigid_registration_error=max_geometry,girdle_unit_wrench_closure=root_closure,scapular_contact_gap_max=abs(np.array(contact_gaps)).max(),support_balance_coordinates=[v['name']for v in definitions],conditional_force_sharing=True,body_and_joint_torque_history_unchanged=True,force_ceiling_added=False,training_feedback=False,girdle_support='Ten analysis-only anatomical SC/AC rotation balances; four compressive scapulothoracic normals. Whole-model mass/inertia and measured arm accelerations determine support demand.',scope='One conditional allocation, not empirical individual muscle force; unsupported neck/other surfaces remain explicitly unidentified, never silently zero.',source=str(f/'motion.npz'),source_sha256=hashlib.sha256((f/'motion.npz').read_bytes()).hexdigest())
        np.savez_compressed(OUT/f'{key}_muscles.npz',times=ts,force_N=F[:,:len(muscles)],scapular_contact_N=F[:,len(muscles):],all_force_N=F,R=R,target=target,passive_force_N=passive,reference_N=refs,**primary);atomic(OUT/f'{key}_muscles.json',report);reports[key]=report;print('FULLBODY_CASE',key,report['primary_peaks_N'],'trapezius_peak',max(float(F[:,i].max())for i,n in enumerate(muscles)if n.startswith('Trapezius')),flush=True)
    if args.case:atomic(OUT/'fullbody_probe.json',reports)
    else:atomic(OUT/'muscle_analysis.json',dict(cases=reports,physical_torque_channels=11,analysis_support_balances=10,human_muscle_forces_validated=False,selection='Normalized force magnitude and continuity; native passive floors retained, added girdle passive floors unidentified. No muscle strength ceiling.'))

if __name__=='__main__':main()

