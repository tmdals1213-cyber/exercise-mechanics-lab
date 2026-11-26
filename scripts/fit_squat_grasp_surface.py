"""Fit finger articulation against the same skin geometry used by Blender.

No skin vertices are projected, removed or hidden to conceal penetration.
Original joint ranges are retained. This is geometry, not a contact-force law.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import mujoco
from scipy.optimize import least_squares
from retarget_measured_jump import MeasuredJumpRig

ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'results/power_comparison/grip_fixture'


def main(side='r'):
    rig=MeasuredJumpRig();b,m,d=rig.b,rig.m,rig.d
    original=dict(np.load(FOLDER/'motion.npz'));qbase=original['qpos'][0].copy();rest=np.load(FOLDER/'reference_pose.npz');skin=np.load(FOLDER/'skin_bind.npz')
    select=skin['hand_weight'][:,0 if side=='r' else 1]>.4;inds=np.flatnonzero(select);lookup={i:j for j,i in enumerate(inds)}
    faces=np.array([[lookup[int(i)] for i in f] for f in skin['faces'] if all(int(i) in lookup for i in f)])
    bind=skin['bind_positions_m'][inds];weights=skin['weights'][inds];sumw=weights.sum(1)
    assert max(abs(sumw-1))<1e-5
    influences=[]
    for bid in np.flatnonzero(weights.sum(0)>1e-8):
        active=np.flatnonzero(weights[:,bid]>1e-8);R=rest['body_xmat'][bid].reshape(3,3)
        influences.append((bid,active,weights[active,bid,None],(bind[active]-rest['body_xpos'][bid])@R))
    def place(q):
        d.qpos[:]=b.complete(q);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    def vertices():
        V=np.zeros_like(bind)
        for bid,ids,w,local in influences:V[ids]+=w*(local@d.xmat[bid].reshape(3,3).T+d.xpos[bid])
        return V
    checkq=skin['reference_qpos'] if 'reference_qpos' in skin else np.load(FOLDER/'motion_open_grip_rejected.npz')['qpos'][0]
    place(checkq);fidelity=float(np.max(np.linalg.norm(vertices()-skin['rendered_positions_m'][inds],axis=1)));place(qbase)
    print('SKINNING_FIDELITY_M',fidelity,flush=True)
    if fidelity>1e-5:raise RuntimeError(('Skinning does not reproduce Blender',fidelity))
    arm=[n+'_'+side for n in ['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion']]
    fingers=[n+'_'+side for digit in [2,3,4,5] for n in [f'mcp{digit}_flexion',f'pm{digit}_flexion',f'md{digit}_flexion']]
    fingers=[n if n in b.joints else n.removesuffix('_'+side) for n in fingers]
    names=arm+fingers+(['cmc_flexion','cmc_abduction','mp_flexion_r','ip_flexion_r'] if side=='r' else ['cmc_flexion_l','cmc_abduction_l','mp_flexion_l','ip_flexion_l'])
    ids=[b.joints[n] for n in names];adrs=[m.jnt_qposadr[j] for j in ids]
    lo=np.array([m.jnt_range[j,0] for j in ids]);hi=np.array([m.jnt_range[j,1] for j in ids]);hi[1]=min(hi[1],1.85)
    lo[names.index('flexion_'+side)]=-np.deg2rad(70.)
    for k,n in enumerate(names):
        if n.startswith('mcp') and 'flexion' in n:lo[k]=max(lo[k],1.0)
        elif n.startswith(('pm2','pm3','pm4','pm5')):lo[k]=max(lo[k],1.20)
        elif n.startswith(('md2','md3','md4','md5')):lo[k]=max(lo[k],.60);hi[k]=min(hi[k],1.4)
    x0=np.maximum(np.minimum(qbase[adrs],hi-1e-6),lo+1e-6)
    bar=original['bar_position_m'][0];axis=original['bar_rotation'][0,:,0];grip=bar+(-.48 if side=='r' else .48)*axis;radius=.014
    contact_bodies=[m.body(f'midph{k}_{side}').id for k in [2,3,4,5]]+[m.body('distal_thumb_'+side).id]
    hand=m.body('lunate_'+side).id;elbow=b.joints['elbow_flexion_'+side]
    proximal=[m.body(f'{k}proxph_'+side).id for k in [2,3,4,5]]
    distal=[m.body(f'distph{k}_'+side).id for k in [2,3,4,5]]
    thumb=m.body('distal_thumb_'+side).id
    thumb_ids=np.flatnonzero(weights[:,thumb]>.5)
    thumb_local=((bind[thumb_ids]-rest['body_xpos'][thumb])@rest['body_xmat'][thumb].reshape(3,3)).mean(0)
    # All skin vertices plus triangle centroids and all three edge midpoints.
    def samples(V):
        tri=V[faces]
        return np.concatenate([V,tri.mean(1),(tri[:,0]+tri[:,1])/2,(tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2])
    def radial(P):
        delta=P-bar
        return np.linalg.norm(delta-np.outer(delta@axis,axis),axis=1)
    calls=[0];barrier=[10000.]
    def fun(x):
        q=qbase.copy();q[adrs]=x;place(q);V=vertices();S=samples(V);rr=radial(S)
        penetration=np.minimum(rr-radius-.0008,0)*barrier[0]/np.sqrt(len(S))
        contacts=(radial(d.xpos[contact_bodies])-.022)*100
        # A nonpenetrating open hand is not a grasp: distal phalanges must
        # reach the opposite side of the shaft from their knuckles.
        bases=d.xpos[proximal]-bar;bases-=np.outer(bases@axis,axis)
        tips=d.xpos[distal]-bar;tips-=np.outer(tips@axis,axis)
        opposition=(tips+.022*bases/np.linalg.norm(bases,axis=1)[:,None]).ravel()*120
        thumb_pad=d.xpos[thumb]+d.xmat[thumb].reshape(3,3)@thumb_local
        thumb_contact=(radial(thumb_pad[None])-.021)*150
        R=d.xmat[hand].reshape(3,3);guide=d.xpos[hand]+R@(np.array([0,-.092,-.026 if side=='r' else .026])*b.report['global_normalization_scale'])
        delta=guide-grip;along=axis@delta
        near=(delta-along*axis)*100+along*axis*1000
        elbow_low=max(0.,d.xanchor[elbow,2]-(bar[2]-.10))*100
        preferred=x0.copy()
        for digit in range(4):preferred[7+digit*3:10+digit*3]=[1.2,1.4,.9]
        reg=(x-preferred)*np.r_[np.full(7,.08),np.full(12,.15),np.full(4,.025)]
        fingers=x[7:19].reshape(4,3)
        coherence=np.r_[(fingers[:,2]-.65*fingers[:,1])*.7,np.diff(fingers[:,:2],axis=0).ravel()*.45]
        calls[0]+=1
        if calls[0]%500==0:print('GRASP_FIT',calls[0],'max penetration mm',max(0.,radius-float(rr.min()))*1000,flush=True)
        return np.r_[penetration,contacts,opposition,thumb_contact,near,elbow_low,reg,coherence]
    # First fit a closed hand around a shaft aligned with its transverse
    # axis. Then place that complete grasp with the arm. Solving both at
    # once can escape penetration by leaving the hand open.
    finger_ids=adrs[7:];hand_lo=np.r_[lo[7:],-.125,-.065 if side=='r' else .010];hand_hi=np.r_[hi[7:],-.070,-.010 if side=='r' else .065]
    hx=np.clip(np.r_[x0[7:],-.100,-.030 if side=='r' else .030],hand_lo+1e-7,hand_hi-1e-7)
    def hand_fun(h):
        q=qbase.copy();q[finger_ids]=h[:-2];place(q);R=d.xmat[hand].reshape(3,3);origin=d.xpos[hand]
        S=(samples(vertices())-origin)@R;centre=h[-2:]
        rr=np.linalg.norm(S[:,1:]-centre,axis=1)
        base=(d.xpos[proximal]-origin)@R;tip=(d.xpos[distal]-origin)@R
        vectors=base[:,1:]-centre;opposition=tip[:,1:]-centre+.022*vectors/np.linalg.norm(vectors,axis=1)[:,None]
        thumb_point=(d.xpos[thumb]+d.xmat[thumb].reshape(3,3)@thumb_local-origin)@R
        return np.r_[np.minimum(rr-radius-.0008,0)*barrier[0]/np.sqrt(len(rr)),
                     opposition.ravel()*70,(np.linalg.norm(thumb_point[1:]-centre)-.022)*80,
                     .015*(h[:-2]-x0[7:])]
    for strength in [10000.,100000.,1000000.]:
        barrier[0]=strength
        fit=least_squares(hand_fun,hx,bounds=(hand_lo,hand_hi),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-7,diff_step=1e-5);hx=fit.x
    print('HAND_CAVITY',hx[-2:],fit.cost,flush=True)
    qclosed=qbase.copy();qclosed[finger_ids]=hx[:-2];local_centre=np.r_[0.,hx[-2:]]
    def arm_fun(a):
        q=qclosed.copy();q[adrs[:7]]=a;place(q);R=d.xmat[hand].reshape(3,3)
        point=d.xpos[hand]+R@local_centre
        return np.r_[(point-grip)*10000,np.cross(R[:,0],axis)*1000,
                     max(0.,d.xanchor[elbow,2]-(bar[2]-.10))*10,.002*(a-x0[:7])]
    best=None
    for plane,rot,wrist in [(x0[0],x0[2],-1.15),(0.,-1.5,-1.),(1.4,-1.5,-1.),(-1.4,1.5,0.),(0.,1.5,0.),(1.4,1.5,.5),(3.,-1.5,.5),(-1.4,-1.5,.5)]:
        seed=x0[:7].copy();seed[:4]=[plane,1.2,rot,2.1];seed[-1]=wrist;seed=np.clip(seed,lo[:7]+1e-7,hi[:7]-1e-7)
        fit=least_squares(arm_fun,seed,bounds=(lo[:7],hi[:7]),max_nfev=180,ftol=1e-11,xtol=1e-11,gtol=1e-8)
        if best is None or fit.cost<best.cost:best=fit
    fit=best;x=np.r_[fit.x,hx[:-2]]
    refine_lo=lo.copy()
    for k,n in enumerate(names):
        if n.startswith('mcp') and 'flexion' in n:refine_lo[k]=.35
        elif n.startswith(('pm2','pm3','pm4','pm5')):refine_lo[k]=.6
        elif n.startswith(('md2','md3','md4','md5')):refine_lo[k]=.25
    for strength in [100000.,1000000.]:
        barrier[0]=strength
        fit=least_squares(fun,np.clip(x,refine_lo+1e-7,hi-1e-7),bounds=(refine_lo,hi),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-7,diff_step=1e-5)
        x=fit.x
    q=qbase.copy();q[adrs]=x;place(q);V=vertices();min_clearance=float(radial(samples(V)).min()-radius)
    previous=json.loads((FOLDER/'grasp_candidate.json').read_text()) if (FOLDER/'grasp_candidate.json').exists() else {}
    values=previous.get('pose_joint_values_rad',{});values.update({n:float(v) for n,v in zip(names,x)})
    place(q)
    original.update(qpos=d.qpos[None].copy(),body_xpos=d.xpos[None].copy(),body_xmat=d.xmat[None].copy())
    np.savez_compressed(FOLDER/'motion.npz',**original)
    centres={}
    for side in ['r','l']:
        h=m.body('lunate_'+side).id;target=bar+(-.48 if side=='r' else .48)*axis
        centres[side]=(d.xmat[h].reshape(3,3).T@(target-d.xpos[h])/b.report['global_normalization_scale']).tolist()
    report=dict(status='candidate_requires_final_render_surface_check',bar_diameter_m=.028,grip_width_m=.96,
                source_skin_bind_sha256=hashlib.sha256((FOLDER/'skin_bind.npz').read_bytes()).hexdigest(),
                skinning_vs_Blender_max_error_m=fidelity,sampled_skin_clearance_m=min_clearance,
                original_joint_ranges_retained=False,skin_geometry_modified=False,pose_joint_values_rad=values,
                modified_joint_ranges_rad={n:[-float(np.deg2rad(70)),float(m.jnt_range[b.joints[n],1])] for n in ['flexion_r','flexion_l']},
                wrist_range_reference='https://benefits.va.gov/PREDISCHARGE/DOCS/disexm34.pdf',
                grasp_centres_unscaled_m=centres,cost=float(fit.cost),calls=calls[0],last_fitted_side=side)
    (FOLDER/'grasp_candidate.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--side',choices=['r','l'],default='r');a=ap.parse_args();main(a.side)
