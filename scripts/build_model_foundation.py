"""New, isolated OHP analysis revision: contact cone + all-muscle/body telemetry.

Run once per grip using EXERCISE_SCENARIO and --grip 65/80.
Existing renders, cached source forces and movements are never overwritten.
"""
import argparse, csv, hashlib, importlib.metadata, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import osqp, mujoco, myo_sim
from solve_fullbody import Body, CONFIG, ROOT
from solve_physiology import problem
from analyze_bar_and_joints import tendon_body_wrenches, subtree
from contact_mechanics import Contact, generalized_force_map, verify_contracts


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve(operators, i, contacts, angle=20., weights=None):
    p=problem(operators,i,CONFIG['bar_mass_kg'],angle,np.inf,True)
    # Replace ONLY the legacy 4-face box friction block, retaining equality,
    # nonnegative muscle effort and GH constraints. Contacts are in kN in QP.
    start=p['R'].shape[0]+p['nm']+len(contacts)
    rows=[]; lo=[]; hi=[]
    for k,c in enumerate(contacts):
        a,l,u=c.inequalities(); block=np.zeros((len(a),p['n']))
        block[:,p['nm']+3*k:p['nm']+3*k+3]=a
        rows.append(block); lo.extend(l); hi.extend(u)
    A=sp.vstack([p['A'][:start], sp.csc_matrix(np.vstack(rows)), p['A'][start+4*len(contacts):]],format='csc')
    lower=np.r_[p['lo'][:start],lo,p['lo'][start+4*len(contacts):]]
    upper=np.r_[p['hi'][:start],hi,p['hi'][start+4*len(contacts):]]
    P=p['P'] if weights is None else sp.diags(np.r_[weights,np.full(p['n']-p['nm'],.002)],format='csc')
    qp=osqp.OSQP(); qp.setup(P=P,q=np.zeros(p['n']),A=A,l=lower,u=upper,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True)
    r=qp.solve(raise_error=False)
    if not r.info.status.startswith('solved'):
        return None,{'status':r.info.status,'frame':int(i)}
    e=np.maximum(r.x[:p['nm']],0); active=p['gain']*e; f=p['passive']+active
    cf=r.x[p['nm']:].reshape(-1,3)*1000
    error=float(abs(p['R']@f+p['C']@(cf.ravel()/1000)-p['target']).max())
    violation=float(np.maximum(A@r.x-upper,lower-A@r.x).max())
    if error>.01 or violation>1e-5: raise RuntimeError(('Invalid solution',i,error,violation))
    return {'force_N':f,'active_force_N':active,'passive_force_N':p['passive'],'normalized_active_effort':e,'contacts_N':cf}, {'status':r.info.status,'frame':int(i),'balance_error':error,'constraint_violation':violation}


def cuts(m):
    out=[]
    for side in ['r','l']:
        for name,body,joint in [('shoulder','humerus','elv_angle'),('elbow','ulna','elbow_flexion'),('wrist','lunate','flexion'),('hip','femur','hip_flexion'),('knee','tibia','knee_angle'),('ankle','talus','ankle_angle')]:
            out.append((name+'_'+side,m.body(body+'_'+side).id,m.joint(joint+'_'+side).id))
    # All lumbar intersegment cuts; anchored to that body's origin. These are
    # aggregate intersegment resultants including prescribed/coupled effects.
    for name in ['lumbar5','lumbar4','lumbar3','lumbar2','lumbar1','torso']:
        out.append((name,m.body(name).id,None))
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--grip',type=int,choices=[65,80],required=True); args=ap.parse_args()
    b=Body(); m,d=b.m,b.d; s=b.length_scale
    source=ROOT/f'results/grip_comparison/grip_{args.grip}/cycle/physiology'
    if abs(CONFIG['grip_width_m']-args.grip/100)>1e-10: raise ValueError('Scenario / grip mismatch')
    dest=ROOT/f'results/model_foundation/grip_{args.grip}'; dest.mkdir(parents=True,exist_ok=True)
    motion=np.load(source/'fullbody_motion.npz'); cached=np.load(source.parent/'physiology_operators.npz')
    op={k:cached[k].copy() for k in cached.files}
    metadata=json.loads((source/'fullbody_metadata.json').read_text('utf-8'))
    # Legacy cache used native body g=9.81 but bar g=9.80665. Use ONE physical
    # gravity for the new revision, correcting both generalized demand and the
    # affine intersegment force offsets before allocating muscle forces.
    old_gravity=m.opt.gravity.copy(); gravity=np.array([0.,0.,-9.80665])
    if not np.allclose(old_gravity,gravity,atol=1e-12,rtol=0):
        ratio=gravity[2]/old_gravity[2]
        for i,q in enumerate(motion['qpos']):
            b.pose(q); G=np.zeros(m.nv); mujoco.mj_rne(m,d,0,G)
            op['target'][i] += b.reduction(q).T@G*(ratio-1)
        for key,body_names in [('shoulder_force_offset',['humerus_r','humerus_l']),('joint_force_offset',['humerus_r','ulna_r','lunate_r','humerus_l','ulna_l','lunate_l'])]:
            for j,name in enumerate(body_names):
                op[key][:,j] += m.body_mass[subtree(m,m.body(name).id)].sum()*(old_gravity-gravity)
    m.opt.gravity[:]=gravity
    np.savez_compressed(dest/'analysis_operators.npz',**op)
    contacts=[Contact('foot_'+str(i),bid,local.copy(),friction=CONFIG['ground_friction']) for i,(bid,local) in enumerate(b.sole)]
    solved=[]; qp_checks=[]
    for i in range(len(motion['times'])):
        sol,check=solve(op,i,contacts)
        if sol is None: raise RuntimeError(check)
        solved.append(sol); qp_checks.append(check)
    values={k:motion[k].copy() for k in motion.files}
    values['reduced_required_generalized_force']=op['target'].copy()
    for k in solved[0]: values[k]=np.array([v[k] for v in solved])
    F=values['force_N']; values['reference_ratio']=F/np.array(metadata['reference_force_N'])
    values['impulse_Ns']=np.zeros_like(F)
    values['impulse_Ns'][1:]=np.cumsum((F[1:]+F[:-1])/2*np.diff(values['times'])[:,None],axis=0)
    np.savez_compressed(dest/'fullbody_motion.npz',**values)
    selected=cuts(m); parts=[subtree(m,bid) for _,bid,_ in selected]
    keys=['musculotendon_length_m','musculotendon_velocity_m_s','normalized_fiber_length','normalized_fiber_velocity_per_s','muscle_generalized_force','muscle_moment_arm_m','joint_force_world_N','joint_moment_world_Nm','joint_force_distal_frame_N','joint_moment_distal_frame_Nm','joint_anchor_world_m','contacts_position_world_m','contact_power_W','body_com_world_m','body_com_acceleration_world_m_s2','com_world_m','com_acceleration_world_m_s2','global_force_residual_N','global_moment_residual_Nm','net_generalized_force','net_generalized_power_W','native_active_gain_N','native_passive_force_N','moment_arm_power_residual_W']
    out={k:[] for k in keys}; maximum_native_force_error=0.; maximum_native_moment_error=0.; max_contact_map_error=0.
    rng=m.actuator_gainprm[:,:2]; L0=(m.actuator_lengthrange[:,1]-m.actuator_lengthrange[:,0])/(rng[:,1]-rng[:,0]); LT=m.actuator_lengthrange[:,0]-L0*rng[:,0]
    assert np.all(L0>0)
    physical_scale=np.r_[np.ones(3),np.full(len(metadata['independent_coordinates'])-3,s)]
    for i,q in enumerate(values['qpos']):
        b.pose(q); d.qvel[:]=values['qvel_native'][i]; mujoco.mj_forward(m,d); d.qacc[:]=values['qacc_native'][i]
        B=b.reduction(q); moment=b.moments(); red=B.T@moment
        C,points=generalized_force_map(m,d,contacts,B)
        max_contact_map_error=max(max_contact_map_error,float(abs(C*1000-op['C'][i]).max()))
        forces=tendon_body_wrenches(b,F[i]); required=np.zeros((m.nbody,3)); native_required=np.zeros_like(required); torques=np.zeros_like(required); native_torques=np.zeros_like(required); acc=np.zeros_like(required)
        for bid in range(1,m.nbody):
            J=np.zeros((3,m.nv)); Jr=np.zeros_like(J); Jdot=np.zeros_like(J); Jrdot=np.zeros_like(J)
            mujoco.mj_jacBodyCom(m,d,J,Jr,bid); mujoco.mj_jacDot(m,d,Jdot,Jrdot,d.xipos[bid],bid)
            a=J@d.qacc+Jdot@d.qvel; acc[bid]=s*a
            required[bid]=m.body_mass[bid]*(s*a-m.opt.gravity)
            native_required[bid]=m.body_mass[bid]*(a-m.opt.gravity)
            R=d.ximat[bid].reshape(3,3); I=R@np.diag(m.body_inertia[bid])@R.T; w=Jr@d.qvel; alpha=Jr@d.qacc+Jrdot@d.qvel
            native_torques[bid]=I@alpha+np.cross(w,I@w); torques[bid]=s*s*native_torques[bid]
        ext=np.zeros((m.nbody,3)); ext_torque=np.zeros_like(ext); native_ext_torque=np.zeros_like(ext)
        hand=-CONFIG['bar_mass_kg']*(values['bar_acceleration_m_s2'][i]-m.opt.gravity)/2
        for side in ['r','l']:
            bid=b.grip_body[side]; ext[bid]+=hand
            native_ext_torque[bid]+=np.cross(b.grip(side)-d.xipos[bid],hand)
        for c,p,f in zip(contacts,points,values['contacts_N'][i]):
            ext[c.body_id]+=f; native_ext_torque[c.body_id]+=np.cross(p-d.xipos[c.body_id],f)
        ext_torque=s*native_ext_torque
        residual=required-forces[:,:3]-ext; residual_torque=torques-s*forces[:,3:]-ext_torque
        native_residual=native_required-forces[:,:3]-ext; native_residual_torque=native_torques-forces[:,3:]-native_ext_torque
        reaction=[]; moment_reaction=[]; local_f=[]; local_m=[]; anchors=[]
        for (_,bid,jid),part in zip(selected,parts):
            anchor=d.xanchor[jid] if jid is not None else d.xpos[bid]
            rf=residual[part].sum(0); rm=(residual_torque[part]+np.cross(s*(d.xipos[part]-anchor),residual[part])).sum(0)
            reaction.append(rf); moment_reaction.append(rm); anchors.append(s*anchor)
            rotation=d.xmat[bid].reshape(3,3); local_f.append(rotation.T@rf); local_m.append(rotation.T@rm)
        if i in [0,30,60,90,138]:
            d.xfrc_applied[:,:3]=ext; d.xfrc_applied[:,3:]=native_ext_torque
            d.actuator_force[:]=-F[i]; d.qfrc_constraint[:]=0; mujoco.mj_rnePostConstraint(m,d)
            for (_,bid,jid),part in zip(selected,parts):
                origin=d.subtree_com[m.body_rootid[bid]]
                nf=native_residual[part].sum(0)
                nm=(native_residual_torque[part]+np.cross(d.xipos[part]-origin,native_residual[part])).sum(0)
                maximum_native_force_error=max(maximum_native_force_error,float(abs(nf-d.cfrc_int[bid,3:]).max()))
                maximum_native_moment_error=max(maximum_native_moment_error,float(abs(nm-d.cfrc_int[bid,:3]).max()))
        com=np.average(d.xipos[1:]*s,axis=0,weights=m.body_mass[1:]); coma=np.average(acc[1:],axis=0,weights=m.body_mass[1:])
        muscle_gf=red.T*F[i,:,None]*physical_scale[None,:]
        gforce=(op['target'][i]+op['bar_tau_per_kg'][i]*0)*physical_scale
        independent_velocity=np.r_[d.qvel[:3]*s,d.qvel[3:6],[d.qvel[m.jnt_dofadr[j]] for j in b.independent]]
        gain=[]; passive=[]
        for a in range(m.nu):
            lr=np.ascontiguousarray(m.actuator_lengthrange[a]); gp=np.ascontiguousarray(m.actuator_gainprm[a,:9]); bp=np.ascontiguousarray(m.actuator_biasprm[a,:9])
            gain.append(max(0,-mujoco.mju_muscleGain(float(d.actuator_length[a]),float(d.actuator_velocity[a]),lr,float(m.actuator_acc0[a]),gp))*s*s)
            passive.append(max(0,-mujoco.mju_muscleBias(float(d.actuator_length[a]),lr,float(m.actuator_acc0[a]),bp))*s*s)
        mt_power=-F[i]*s*d.actuator_velocity
        row={'musculotendon_length_m':s*d.actuator_length.copy(),'musculotendon_velocity_m_s':s*d.actuator_velocity.copy(),'normalized_fiber_length':(d.actuator_length-LT)/L0,'normalized_fiber_velocity_per_s':d.actuator_velocity/L0,
             'muscle_generalized_force':muscle_gf,'muscle_moment_arm_m':(red[3:].T*s),'joint_force_world_N':reaction,'joint_moment_world_Nm':moment_reaction,'joint_force_distal_frame_N':local_f,'joint_moment_distal_frame_Nm':local_m,'joint_anchor_world_m':anchors,
             'contacts_position_world_m':s*points,'contact_power_W':[float(f@(b.jacobian(c.body_id,c.local_point)[0]@d.qvel*s)) for c,f in zip(contacts,values['contacts_N'][i])],
             'body_com_world_m':s*d.xipos.copy(),'body_com_acceleration_world_m_s2':acc,'com_world_m':com,'com_acceleration_world_m_s2':coma,'global_force_residual_N':residual[1:].sum(0),
             'global_moment_residual_Nm':(residual_torque[1:]+np.cross(s*d.xipos[1:]-com,residual[1:])).sum(0),'net_generalized_force':gforce,'net_generalized_power_W':gforce*independent_velocity,
             'native_active_gain_N':gain,'native_passive_force_N':passive,'moment_arm_power_residual_W':muscle_gf@independent_velocity-mt_power}
        for k in keys: out[k].append(row[k])
        if i%30==0: print('TELEMETRY',args.grip,i,flush=True)
    out={k:np.asarray(v) for k,v in out.items()}
    out.update(times=values['times'],phase_code=values['phase_code'],force_N=F,active_force_N=values['active_force_N'],passive_force_N=values['passive_force_N'],normalized_active_effort=values['normalized_active_effort'],contacts_N=values['contacts_N'],musculotendon_power_W=-F*out['musculotendon_velocity_m_s'])
    np.savez_compressed(dest/'telemetry.npz',**out)
    friction_violation=float((np.linalg.norm(values['contacts_N'][:,:,:2],axis=2)-CONFIG['ground_friction']*values['contacts_N'][:,:,2]).max())
    identity=float(abs(F-values['active_force_N']-out['native_passive_force_N']).max())
    gain_identity=float(abs(out['native_active_gain_N']*values['normalized_active_effort']-values['active_force_N']).max())
    # A joint force-frame rotation must preserve magnitude.
    frame_error=float(abs(np.linalg.norm(out['joint_force_world_N'],axis=2)-np.linalg.norm(out['joint_force_distal_frame_N'],axis=2)).max())
    checks={'status':'passed','gravity_world_m_s2':gravity.tolist(),'legacy_body_gravity_world_m_s2':old_gravity.tolist(),'frames':len(F),'musculotendon_actuators':m.nu,'monitored_intersegment_cuts':len(selected),'native_RNE_force_crosscheck_N':maximum_native_force_error,'native_RNE_moment_crosscheck_Nm':maximum_native_moment_error,
            'contact_generalized_map_error':max_contact_map_error,'global_force_residual_N':float(abs(out['global_force_residual_N']).max()),'global_moment_residual_Nm':float(abs(out['global_moment_residual_Nm']).max()),
            'muscle_power_virtual_work_error_W':float(abs(out['moment_arm_power_residual_W']).max()),'native_passive_identity_error_N':identity,'native_gain_identity_error_N':gain_identity,'joint_frame_rotation_error_N':frame_error,
            'maximum_circular_friction_violation_N':friction_violation,'max_QP_balance_error':max(c['balance_error'] for c in qp_checks),'contact_contract_checks':verify_contracts(),'scope':'Numerical verification, not experimental validation.'}
    assert maximum_native_force_error<1e-5 and maximum_native_moment_error<1e-5,checks
    assert checks['global_force_residual_N']<.01 and checks['global_moment_residual_Nm']<.01,checks
    assert max_contact_map_error<1e-7 and identity<1e-7 and gain_identity<1e-7 and frame_error<1e-7 and friction_violation<1e-4,checks
    assert checks['muscle_power_virtual_work_error_W']<1e-4,checks
    (dest/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    names=metadata['muscle_names']; coords=metadata['independent_coordinates']; times=values['times']; duration=times[-1]-times[0]
    mapping=json.loads((source/'atlas_mapping.json').read_text('utf-8'))
    mapped={a for x in mapping for a in x['actuators']}; unmapped=[n for n in names if n not in mapped]
    zero_gain=out['native_active_gain_N']<=1e-8
    outside=(out['normalized_fiber_length']<m.actuator_gainprm[None,:,4]) | (out['normalized_fiber_length']>m.actuator_gainprm[None,:,5])
    audit={'render_meshes':len(mapping),'render_meshes_with_actuator_mapping':sum(bool(x['actuators']) for x in mapping),'actuators_without_render_surface':unmapped,
           'zero_active_gain_at_any_frame':[names[i] for i in np.where(zero_gain.any(0))[0]],'zero_active_gain_at_all_frames':[names[i] for i in np.where(zero_gain.all(0))[0]],
           'outside_active_length_support_at_any_frame':[names[i] for i in np.where(outside.any(0))[0]],'source_force_change_max_N':float(abs(F-motion['force_N']).max()),
           'native_inelastic_tendon':True,'native_scalar_muscle_law_not_3D_tissue_stress':True}
    (dest/'coverage_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    provenance={'schema':'exercise-mechanics.telemetry/1.0','grip_cm':args.grip,'subject':CONFIG['subject'],'runtime':{p:importlib.metadata.version(p) for p in ['myo-sim','mujoco','numpy','scipy','osqp']},
      'sources':{'motion_sha256':sha(source/'fullbody_motion.npz'),'operators_sha256':sha(source.parent/'physiology_operators.npz'),'scenario_sha256':sha(Path(__import__('os').environ['EXERCISE_SCENARIO']))},
      'implementation_sha256':{name:sha(ROOT/'scripts'/name) for name in ['build_model_foundation.py','contact_mechanics.py','solve_fullbody.py','solve_physiology.py','analyze_bar_and_joints.py']},
      'model_source_files_sha256':{str(p.relative_to(Path(myo_sim.__file__).parent)):sha(p) for p in Path(myo_sim.__file__).parent.rglob('*') if p.is_file() and p.suffix in ['.xml','.py','.md']},
      'world_frame':'Native MyoSim right-handed axes, Z up; all exported lengths and wrenches physical SI units.','gravity_world_m_s2':gravity.tolist(),
      'muscles':names,'bodies':metadata['body_names'],'coordinates':coords,'joint_cuts':[{'name':n,'body':m.body(bid).name,'anchor_joint':m.joint(jid).name if jid is not None else 'body origin'} for n,bid,jid in selected],
      'phase_labels':['ascent','top hold','descent','bottom hold'],'equilibrium':'M(q)qdd+h(q,qd)=R(q)F+J(q)^T f_ext, reduced by prescribed coordinate couplings.',
      'force_allocation':'minimum sum(e^2) plus small contact-force regularizer; F=Fpassive+native_FLV_gain*e; e>=0 unbounded; 20deg GH cone; 16-face inscribed foot-friction cone',
      'units':{'muscle_generalized_force':'N in first three root translations, Nm in remaining rotational coordinates','muscle_moment_arm_m':'34 rotational coordinates, coordinates[3:]','net_generalized_force':'body inertial/gravity plus bar demand BEFORE subtracting ground; N first3, Nm others','net_generalized_power_W':'that demand times corresponding independent-coordinate velocity; not muscle or metabolic power','joint_force_world_N':'proximal-on-distal aggregate intersegment constraint resultant','joint_moment_world_Nm':'aggregate intersegment moment about saved anchor, including coupled/prescribed effects; not cartilage contact torque','normalized_fiber_length':'MuJoCo inferred constant-tendon-length fiber surrogate / L0','normalized_fiber_velocity_per_s':'dL/dt / L0','musculotendon_power_W':'positive shortening delivers mechanical power; rigid-tendon surrogate, not metabolic/fiber material power'},
      'limitations':['Generic scaled reference male, not a measured 180cm/90kg person or calibrated muscularity.','No hard strength ceiling, no neural activation estimate; no activation dynamics or fatigue integration.','Native inelastic-tendon FLV; no compliant tendon, explicit pennation mechanics or three-dimensional soft-tissue stress.','51 prescribed couplers and fixed fingers/head; reduced force sharing does not determine every passive constraint force uniquely.','Intersegment cut loads aggregate constraint effects and do not identify individual ligaments, disc pressure or cartilage pressure.','No individual PCSA/optimal-length calibration. Visual bulk is not muscle capacity.','30Hz source trajectory: landing/impact peaks cannot be inferred from this dataset.','Pull-up, muscle-up, squat and jump movement solvers are not implemented.']}
    (dest/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    with (dest/'muscle_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['actuator','mean_total_N','peak_total_N','mean_active_N','mean_passive_N','min_length_m','max_length_m','peak_abs_velocity_m_s','positive_MT_work_J','negative_MT_work_J','force_time_integral_Ns'])
        for j,n in enumerate(names):
            pw=out['musculotendon_power_W'][:,j]
            w.writerow([n,np.trapezoid(F[:,j],times)/duration,F[:,j].max(),np.trapezoid(values['active_force_N'][:,j],times)/duration,np.trapezoid(values['passive_force_N'][:,j],times)/duration,out['musculotendon_length_m'][:,j].min(),out['musculotendon_length_m'][:,j].max(),abs(out['musculotendon_velocity_m_s'][:,j]).max(),np.trapezoid(np.maximum(pw,0),times),np.trapezoid(np.minimum(pw,0),times),np.trapezoid(F[:,j],times)])
    with (dest/'joint_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['intersegment_cut','mean_resultant_N','peak_resultant_N','mean_resultant_moment_Nm','peak_resultant_moment_Nm'])
        for j,(n,_,_) in enumerate(selected):
            jf=np.linalg.norm(out['joint_force_world_N'][:,j],axis=1); jm=np.linalg.norm(out['joint_moment_world_Nm'][:,j],axis=1)
            w.writerow([n,np.trapezoid(jf,times)/duration,jf.max(),np.trapezoid(jm,times)/duration,jm.max()])
    print('FOUNDATION_COMPLETE',args.grip,json.dumps(checks),flush=True)


if __name__=='__main__': main()
