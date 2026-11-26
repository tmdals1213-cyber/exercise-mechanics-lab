"""Trace integration resolution, force-allocation geometry and arm free bodies.

No model parameters are modified. LP minima and singular values are diagnostics,
not recruitment predictions or measured single-muscle lever arms.
"""
import json,hashlib,numpy as np,mujoco
from pathlib import Path
from scipy.optimize import linprog
from jump_rig import JumpRig,ROOT
from muscle_mechanics import subtree
from tension_reference import solve_causal_history
OUT=ROOT/'results/tension_visualization'

def main():
    result={'status':'diagnostic_only_load_comparison_rejected','physics':'Newton-Euler with inertia; no biomechanical exception to force/moment balance','ground_reaction_and_COP':'prescribed force profile and separately optimized COP, not measured or solved by a coupled physical contact model','render_fps':30,'force_derivatives_from_30fps_video':False,'resolution':[],'operator_diagnostics':[],'arm_free_body':[]}
    for hz in [600,1200,2400]:
        f=ROOT/f'results/jump_validation/vertical/acceleration_{hz}hz';z=np.load(f/'motion.npz');l=np.load(f/'loads.npz');s=json.loads((f/'setup.json').read_text('utf-8'));names=s['coordinate_names'][6:];ix=[i for i,n in enumerate(names) if n.startswith(('elv_angle','shoulder_elv','shoulder_rot'))];tau=l['net_joint_moment_Nm'];a=z['udot'][:,6:];u=z['u'][:,6:]
        result['resolution'].append({'physics_hz':hz,'dt_ms':1000/hz,'shoulder_generalized_moment_abs_peak_Nm':float(abs(tau[:,ix]).max()),'shoulder_coordinate_acceleration_abs_peak_rad_s2':float(abs(a[:,ix]).max()),'shoulder_coordinate_speed_abs_peak_rad_s':float(abs(u[:,ix]).max()),'GRF_vertical_peak_N':float(l['contact_force_world_N'].sum(1)[:,2].max()),'source_sha256':s['source_sha256']})
    assert all(r['source_sha256']==result['resolution'][0]['source_sha256'] for r in result['resolution'])
    result['same_code_in_resolution_comparison']=True
    p=ROOT/'results/jump_validation/vertical/acceleration_2400hz';motion=np.load(p/'motion.npz');geo=np.load(OUT/'geometry.npz');op=np.load(OUT/'operators.npz');raw=np.load(OUT/'standard.npz');d=json.loads((OUT/'data.json').read_text('utf-8'));setup=json.loads((p/'setup.json').read_text('utf-8'));coords=setup['coordinate_names'][6:];r=JumpRig();m,b=r.m,r.b
    armrows=[i for i,n in enumerate(coords) if n.startswith(('elv_angle','shoulder_','elbow_','pro_sup_','deviation_','flexion_'))]
    for frame in [0,40,41,47]:
        R=op['R'][frame];T=op['target'][frame];G=op['gain'][frame];P=op['passive'][frame];F=solve_causal_history([0.],R[None],T[None],G[None],P[None],op['reference'],None,0.,1.)['force_N'][0];sc=np.maximum(np.linalg.norm(R,axis=1),1e-3);bounds=[(float(v),float(v) if g<1e-8 else None) for v,g in zip(P,G)];lp=linprog(np.ones(m.nu),A_eq=R/sc[:,None],b_eq=T/sc,bounds=bounds,method='highs');sv=np.linalg.svd(R[armrows][:,G>=1e-8],compute_uv=False);top=np.argsort(F)[-5:][::-1]
        item={'display_frame':frame,'time_s':float(raw['times'][frame]),'no_GH_no_history_all_element_force_sum_N':float(F.sum()),'minimum_all_element_force_sum_LP_N':float(lp.fun) if lp.success else None,'max_target_generalized_moment_Nm':float(abs(T).max()),'arm_block_smallest_singular_value_m':float(sv[-1]),'arm_block_condition_ratio':float(sv[0]/max(sv[-1],1e-30)),'singular_value_is_not_one_muscle_moment_arm':True,'largest_elements':[{'name':b.names[i],'force_N':float(F[i]),'passive_N':float(P[i]),'largest_absolute_generalized_moment_arm_m':float(abs(R[:,i]).max())} for i in top]}
        if frame==40:
            keys={'deltoid','pectoralis','triceps','biceps','rotator_cuff','latissimus','arm_accessory','forearm','hand_intrinsics'};upper=[i for g in d['groups'] if g['key'] in keys for i in g['indices']];objective=np.zeros(m.nu);objective[upper]=1;ul=linprog(objective,A_eq=R/sc[:,None],b_eq=T/sc,bounds=bounds,method='highs');aid=b.names.index('DELT2');row=int(np.argmax(abs(R[:,aid])));moment=float(R[row,aid]*F[aid]);item['minimum_upper_force_sum_LP_N']=float(ul.fun) if ul.success else None;item['upper_sum_includes_LAT_for_the_lower_bound']=True;item['torque_cancellation']={'coordinate':coords[row],'target_Nm':float(T[row]),'DELT2_force_N':float(F[aid]),'DELT2_moment_arm_m':float(R[row,aid]),'DELT2_moment_contribution_Nm':moment,'all_other_elements_contribution_Nm':float(T[row]-moment),'GH_and_history_constraints_used':False}
        result['operator_diagnostics'].append(item)
        j=int(geo['source_indices'][frame]);q=motion['qpos'][j];v=motion['qvel'][j];a=motion['qacc'][j];b.pose(q,v,a)
        for side in ['r','l']:
            part=subtree(m,m.body('humerus_'+side).id);centre=r.d.xanchor[b.joints['shoulder_elv_'+side]].copy();force=np.zeros(3);moment=np.zeros(3);mass=0.;maxalpha=0.
            for bid in part:
                mb=float(m.body_mass[bid])
                if mb<=0:continue
                J=np.zeros((3,m.nv));Jr=np.zeros_like(J);Jdot=np.zeros_like(J);Jrdot=np.zeros_like(J);mujoco.mj_jacBodyCom(m,r.d,J,Jr,bid);mujoco.mj_jacDot(m,r.d,Jdot,Jrdot,r.d.xipos[bid],bid)
                acc=J@a+Jdot@v;omega=Jr@v;alpha=Jr@a+Jrdot@v;rot=r.d.ximat[bid].reshape(3,3);I=rot@np.diag(m.body_inertia[bid])@rot.T;f=mb*(acc-m.opt.gravity);force+=f;moment+=np.cross(r.d.xipos[bid]-centre,f)+I@alpha+np.cross(omega,I@omega);mass+=mb;maxalpha=max(maxalpha,float(np.linalg.norm(alpha)))
            result['arm_free_body'].append({'time_s':float(raw['times'][frame]),'side':side,'mass_kg':mass,'net_proximal_force_norm_N':float(np.linalg.norm(force)),'total_required_proximal_moment_norm_Nm':float(np.linalg.norm(moment)),'maximum_segment_angular_acceleration_norm_rad_s2':maxalpha,'moment_definition':'Total inertial-minus-gravity wrench about the shoulder centre, before separating passive/joint-coupling/muscle contributions; not a muscle-tension sum.'})
    result['conclusion']='Refining the integrator does not remove the abnormal upstream dynamics. Poor torque-generation geometry at a flight pose persists even after removing GH and history constraints. This is not explained by force summation or long time steps alone; verify the contact/motion task, segment dynamics and shoulder geometry/couplings before estimating human muscle loads.'
    result['observed_refinement_does_not_remove_peaks']=True
    result['not_a_claim_of_complete_time_step_convergence']=True
    result['input_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'motion.npz',p/'loads.npz',OUT/'operators.npz',OUT/'standard.npz']}
    (OUT/'balance_pipeline_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
