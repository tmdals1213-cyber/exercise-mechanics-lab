"""Connected-body morphology verification at controlled held arm postures.

These are STATIC fixtures with equal joint angles and ideal 100kg two-hand
load. They are not refitted OHP trajectories or production recruitment data.
"""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import osqp,mujoco
from parametric_body import Subject,build_subject,load_subject_profiles
from contact_mechanics import Contact,generalized_force_map
from muscle_mechanics import tendon_body_wrenches,subtree

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/parametric_body'


def load_fixture(b,source,index):
    q=b.q0.copy()
    for jid in b.independent:q[b.m.jnt_qposadr[jid]]=source['qpos'][index,b.m.jnt_qposadr[jid]]
    b.pose(q)
    # Keep the design sole references on ground after changing leg proportions.
    q[2]-=np.mean([b.d.site_xpos[b.m.site('subject_sole_reference_'+s).id,2] for s in ['r','l']])
    b.pose(q);return b.d.qpos.copy()


def snapshot(b,q):
    m,d=b.m,b.d;b.pose(q);B=b.reduction(q);R=B.T@b.moments()
    contacts=[];scale=b.report['global_normalization_scale']
    for side in ['r','l']:
        bid=m.body('calcn_'+side).id;origin=d.site_xpos[m.site('subject_sole_reference_'+side).id].copy();origin[2]=0.
        for x,y in [(-.037,-.16),(.037,-.16),(-.037,.095),(.037,.095)]:
            point=origin+scale*np.array([x,y,0.]);local=d.xmat[bid].reshape(3,3).T@(point-d.xpos[bid])
            contacts.append(Contact('foot_'+str(len(contacts)),bid,local))
    C,points=generalized_force_map(m,d,contacts,B);C*=1000
    hand=np.array([0.,0.,-100*9.80665/2]);applied=np.zeros((m.nbody,3));extmom=np.zeros_like(applied);grips=[]
    tau=np.zeros(m.nv);mujoco.mj_rne(m,d,1,tau)
    for side in ['r','l']:
        bid=m.body('lunate_'+side).id;local=np.array([0.,-.065,0.])*scale;J,p=b.jacobian(bid,local)
        tau-=J.T@hand;applied[bid]+=hand;extmom[bid]+=np.cross(p-d.xipos[bid],hand);grips.append(p)
    target=B.T@tau;gain=[];passive=[]
    for a in range(m.nu):
        lr=np.ascontiguousarray(m.actuator_lengthrange[a]);acc=float(m.actuator_acc0[a]);L=float(d.actuator_length[a]);V=0.
        gain.append(max(0.,-mujoco.mju_muscleGain(L,V,lr,acc,np.ascontiguousarray(m.actuator_gainprm[a,:9]))))
        passive.append(max(0.,-mujoco.mju_muscleBias(L,lr,acc,np.ascontiguousarray(m.actuator_biasprm[a,:9]))))
    gain=np.array(gain);passive=np.array(passive);matrix=np.c_[R*gain,C];rowscale=np.maximum(np.linalg.norm(matrix,axis=1),20.)
    n=m.nu+C.shape[1];arrays=[sp.csc_matrix(matrix/rowscale[:,None])];lo=[(target-R@passive)/rowscale];hi=[lo[0].copy()]
    bound=np.zeros((m.nu,n));bound[:,:m.nu]=np.eye(m.nu);arrays.append(sp.csc_matrix(bound));lo.append(np.zeros(m.nu));hi.append(np.full(m.nu,np.inf))
    for k,contact in enumerate(contacts):
        a,l,u=contact.inequalities();row=np.zeros((len(a),n));row[:,m.nu+3*k:m.nu+3*k+3]=a
        arrays.append(sp.csc_matrix(row));lo.append(l);hi.append(u)
    A=sp.vstack(arrays,format='csc');lower=np.concatenate(lo);upper=np.concatenate(hi)
    qp=osqp.OSQP();qp.setup(P=sp.diags(np.r_[np.ones(m.nu),np.full(C.shape[1],.002)],format='csc'),q=np.zeros(n),A=A,l=lower,u=upper,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True,verbose=False)
    sol=qp.solve(raise_error=False)
    if not sol.info.status.startswith('solved'):raise RuntimeError(sol.info.status)
    effort=np.maximum(sol.x[:m.nu],0);force=passive+gain*effort;foot=sol.x[m.nu:].reshape(-1,3)*1000
    balance=float(abs(R@force+C@(foot.ravel()/1000)-target).max());assert balance<.01
    for contact,p,f in zip(contacts,points,foot):applied[contact.body_id]+=f;extmom[contact.body_id]+=np.cross(p-d.xipos[contact.body_id],f)
    wrench=tendon_body_wrenches(b,force);required=-m.body_mass[:,None]*m.opt.gravity
    residual=required-wrench[:,:3]-applied
    body_moment=-wrench[:,3:]-extmom
    com=np.average(d.xipos[1:],axis=0,weights=m.body_mass[1:])
    global_force=float(abs(residual[1:].sum(0)).max());global_moment=float(abs((body_moment[1:]+np.cross(d.xipos[1:]-com,residual[1:])).sum(0)).max())
    assert max(global_force,global_moment)<.01
    # Independent spatial assembly / generalized muscle force identity.
    assembled=np.zeros(m.nv)
    for bid in range(1,m.nbody):mujoco.mj_applyFT(m,d,wrench[bid,:3],wrench[bid,3:],d.xipos[bid],bid,assembled)
    vw=float(abs(assembled-b.moments()@force).max());assert vw<1e-5
    # Internally consistent path derivatives after morphology and equality edits.
    fd_error=0.;eps=1e-6
    for j,jid in enumerate(b.independent):
        a=m.jnt_qposadr[jid];plus=q.copy();minus=q.copy();plus[a]+=eps;minus[a]-=eps
        b.pose(plus);Lp=d.actuator_length.copy();b.pose(minus);Lm=d.actuator_length.copy()
        fd_error=max(fd_error,float(abs(-(Lp-Lm)/(2*eps)-R[6+j]).max()))
    b.pose(q);assert fd_error<2e-5,fd_error
    # Compare the inertial response under a unit independent-coordinate
    # shoulder angular acceleration; qdot=0 so B*qdd is sufficient.
    unit=np.zeros(B.shape[1]);unit[b.coordinate_names.index('shoulder_elv_r')]=1.;d.qacc[:]=B@unit
    inertial=np.zeros(m.nv);mujoco.mj_rne(m,d,1,inertial);d.qacc[:]=0;grav=np.zeros(m.nv);mujoco.mj_rne(m,d,1,grav)
    inertia_response=B.T@(inertial-grav)
    cut_names=[];cut_forces=[]
    for side in ['r','l']:
        for name in ['humerus','ulna','lunate','femur','tibia','talus']:
            indices=subtree(m,m.body(name+'_'+side).id);cut_names.append(name+'_'+side);cut_forces.append(residual[indices].sum(0))
    for name in ['lumbar5','lumbar4','lumbar3','lumbar2','lumbar1','torso']:
        cut_names.append(name);cut_forces.append(residual[subtree(m,m.body(name).id)].sum(0))
    lengths={}
    for side in ['r','l']:
        for name,a,z in [('upper_arm','elv_angle','elbow_flexion'),('forearm','elbow_flexion','flexion'),('thigh','hip_flexion','knee_angle'),('shank','knee_angle','ankle_angle')]:
            lengths[name+'_'+side]=float(np.linalg.norm(d.xanchor[b.joints[a+'_'+side]]-d.xanchor[b.joints[z+'_'+side]]))
    return {'qpos':q,'net_demand':target,'force_N':force,'active_force_N':gain*effort,'passive_force_N':passive,'contact_force_N':foot,'hand_positions_world_m':grips,'moment_arm_matrix':R,
            'intersegment_force_world_N':cut_forces,'unit_acceleration_demand':inertia_response,'muscle_length_m':d.actuator_length.copy()}, {
            'status':'passed','muscle_allocation':'diagnostic minimum-effort static allocation, no GH direction constraint; not production OHP recruitment','coordinate_names':b.coordinate_names,'cut_names':cut_names,'joint_center_distances_m':lengths,
            'bar_mass_kg':100.,'hand_spacing_m':float(np.linalg.norm(np.diff(np.asarray(grips),axis=0))), 'grip_control':'joint angles held equal across subjects; hand spacing is allowed to change',
            'max_generalized_balance_error':balance,'wholebody_force_residual_N':global_force,'wholebody_moment_residual_Nm':global_moment,'muscle_spatial_virtual_work_error':vw,'path_derivative_max_error_m_per_rad':fd_error,
            'support_force_vertical_N':float(foot[:,2].sum()),'strength_ceiling_enforced':False}


def main():
    OUT.mkdir(exist_ok=True);source=np.load(ROOT/'results/grip_comparison/grip_65/cycle/physiology/fullbody_motion.npz')
    cases=load_subject_profiles(ROOT/'scenarios/body_profiles.json')
    summary={};arrays={}
    for name,subject in cases.items():
        b=build_subject(subject);folder=OUT/name; b.save(folder);rows=[];records=[]
        if name=='reference':(OUT/'model_graph.json').write_text(json.dumps(b.graph(),indent=2),encoding='utf-8')
        for index in [20,35,50]:
            q=load_fixture(b,source,index);data,checks=snapshot(b,q);records.append(data);rows.append({'reference_pose_index':index,**checks})
        np.savez_compressed(folder/'fixture_data.npz',**{k:np.array([r[k] for r in records]) for k in records[0]})
        (folder/'fixture_verification.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');summary[name]={'subject':b.report['subject'],'stature_m':b.report['stature_reference_m'],'mass_kg':b.report['mass_kg'],'snapshots':rows}
        arrays[name]=records;print('PARAMETRIC_CASE_VERIFIED',name,flush=True)
    # Tests that changed parameters reach the physics, not only the render.
    ref=arrays['reference'][1]
    differences={}
    for name,records in arrays.items():
        r=records[1];differences[name]={'largest_net_demand_change':float(abs(r['net_demand']-ref['net_demand']).max()),'largest_muscle_force_change_N':float(abs(r['force_N']-ref['force_N']).max()),
                'largest_moment_arm_change':float(abs(r['moment_arm_matrix']-ref['moment_arm_matrix']).max()),'largest_inertial_response_change':float(abs(r['unit_acceleration_demand']-ref['unit_acceleration_demand']).max())}
        if name!='reference':assert differences[name]['largest_net_demand_change']>.01 and differences[name]['largest_moment_arm_change']>1e-5
    result={'status':'passed_static_morphology_fixtures_not_complete_motion_solver','cases':summary,'middle_pose_differences_vs_reference':differences,
            'implemented':'movement-independent physical-SI full-body builder; six bilateral segment-size groups; stature normalization, mass policies, inertial update, joint/site/wrap/mesh transformation, slide-coupler dimensional scaling, explicit muscle architecture rule',
            'critical_scope':'Local segment SIMILARITY scaling changes width/depth as well as length. Fixed-angle static load fixtures allow hand spacing to vary. These do not show a standard-grip OHP trajectory or validated individual recruitment. Missing thoracoscapular actuators and prescribed scapular couplers remain.',
            'next_gates':['integrate independent scapular actuation','length-only and independent width morphology with joint-congruence checks','refit whole-body movement to contacts after each subject change','task-specific muscle operating-range and wrapping-continuity checks','dynamic contact and temporal convergence for jumps/turnovers'],
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'profiles_sha256':hashlib.sha256((ROOT/'scenarios/body_profiles.json').read_bytes()).hexdigest(),'source_postures_sha256':hashlib.sha256((ROOT/'results/grip_comparison/grip_65/cycle/physiology/fullbody_motion.npz').read_bytes()).hexdigest()}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (OUT/'middle_pose_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['subject','height_m','mass_kg','upper_arm_m','forearm_m','thigh_m','shank_m','hand_spacing_m','right_shoulder_elevation_net_Nm','right_elbow_flexion_net_Nm','right_hip_flexion_net_Nm','right_knee_net_Nm'])
        for name in cases:
            row=summary[name]['snapshots'][1];cn=row['coordinate_names'];a=arrays[name][1]['net_demand'];L=row['joint_center_distances_m']
            w.writerow([name,summary[name]['stature_m'],summary[name]['mass_kg'],*[L[k+'_r'] for k in ['upper_arm','forearm','thigh','shank']],row['hand_spacing_m'],*[a[cn.index(k)] for k in ['shoulder_elv_r','elbow_flexion_r','hip_flexion_r','knee_angle_r']]])
    print('CONNECTED_MORPHOLOGY_BENCH_COMPLETE',json.dumps(differences),flush=True)


if __name__=='__main__':main()
