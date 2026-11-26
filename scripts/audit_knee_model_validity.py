"""Compare knee geometry fidelity separately from numerical force balance.

The original lineage model and an improved OpenSim model are comparators, not
human force measurements. No canonical path, motion or muscle force is changed.
"""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,opensim as osim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retarget_measured_jump import MeasuredJumpRig
from scipy.optimize import linprog
from tension_reference import reaction_cone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/knee_model_validity'
MUSCLES=['vaslat_r','vasmed_r','vasint_r','recfem_r']

def source_model(path,label):
    tree=ET.parse(path)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag=='attached_geometry':parent.remove(child)
    target=OUT/(label+'_without_display_meshes.osim');tree.write(target,encoding='utf-8',xml_declaration=True)
    model=osim.Model(str(target));state=model.initSystem()
    for c in model.getCoordinateSet():c.setValue(state,0.,False)
    coord=model.getCoordinateSet().get('knee_angle_r');muscles=[model.getMuscles().get(n) for n in MUSCLES];couplers=[]
    for node in tree.findall('.//CoordinateCouplerConstraint'):
        coeff=node.find('coupled_coordinates_function/LinearFunction/coefficients');assert coeff is not None
        slope,offset=map(float,coeff.text.split());scale=float(node.findtext('scale_factor'));couplers.append((model.getCoordinateSet().get(node.findtext('dependent_coordinate_name')),model.getCoordinateSet().get(node.findtext('independent_coordinate_names')),slope*scale,offset*scale))
    def evaluate(theta):
        coord.setValue(state,float(theta),False)
        for dep,ind,slope,offset in couplers:dep.setValue(state,slope*ind.getValue(state)+offset,False)
        model.realizePosition(state)
        if abs(coord.getValue(state)-theta)>1e-7:raise RuntimeError(('Reference coordinate moved',label,theta,coord.getValue(state)))
        return np.array([m.computeMomentArm(state,coord) for m in muscles]),np.array([m.getLength(state) for m in muscles])
    return evaluate

def main():
    OUT.mkdir(parents=True,exist_ok=True);osim.Logger.setLevelString('error');rig=MeasuredJumpRig();b,m,d=rig.b,rig.m,rig.d;g=b.report['global_normalization_scale'];angle_index=m.jnt_qposadr[b.joints['knee_angle_r']];col=6+b.ind_names.index('knee_angle_r');ids=[b.names.index(n) for n in MUSCLES]
    def native(theta):
        q=b.q0.copy();q[angle_index]=theta;b.pose(q);B,_=rig.coordinates(q,np.zeros(rig.n));R=(B.T@b.moments())[col,ids];return R/g,d.actuator_length[ids].copy()/g
    sources={'MyoSim converted (scale removed)':native,
             'OpenSim Rajagopal 2016':source_model(ROOT/'datasets/knee_model_reference/Rajagopal2016.osim','rajagopal2016'),
             'OpenSim revised 2023':source_model(ROOT/'datasets/knee_model_reference/RajagopalLaiUhlrich2023.osim','revised2023'),
             'CMJ Catelli generic':source_model(ROOT/'datasets/cmj_calisti2025/source/catelli_3dof_model.osim','catelli')}
    deg=np.arange(0.,120.01,3.);theta=np.deg2rad(deg);curves={};derivatives={};h=1e-5
    for label,evaluate in sources.items():
        rows=[];checks=[]
        for angle in theta:
            arm,_=evaluate(angle);rows.append(arm)
            if angle>h and angle<theta[-1]-h:
                _,plus=evaluate(angle+h);_,minus=evaluate(angle-h);fd=-(plus-minus)/(2*h);checks.append(float(abs(fd-arm).max()))
        curves[label]=np.array(rows);derivatives[label]=dict(max_length_derivative_error_m=max(checks),scope='Numerical path derivative including each model coordinate constraints; not anatomical validation.')
        print('KNEE_MODEL_CURVE',label,'FD_error',max(checks),flush=True)
    z=np.load(ROOT/'results/measured_jump_analysis/s10_CMJ_t1/motion.npz');audit=json.loads((ROOT/'results/measured_jump_analysis/s10_CMJ_t1/quadriceps_peak_audit.json').read_text());k=int(np.argmin(abs(z['times']-audit['peak_time_s'])));angle=float(z['qpos'][k,angle_index]);peak={label:{n:float(-evaluate(angle)[0][j]*100) for j,n in enumerate(MUSCLES)} for label,evaluate in sources.items()}
    reference=curves['OpenSim Rajagopal 2016'];converted=curves['MyoSim converted (scale removed)'];maxdiff=abs(reference-converted).max(axis=0)*100
    ts=np.load(ROOT/'results/measured_jump_analysis/s10_CMJ_t1/tensions.npz');summary=json.loads((ROOT/'results/measured_jump_analysis/s10_CMJ_t1/tension_summary.json').read_text());kt=int(np.argmin(abs(ts['times']-audit['peak_time_s'])));keep=[i for i,n in enumerate(summary['coordinate_names']) if not n.startswith('mtp_angle_')];quad=[j for j,n in enumerate(summary['names']) if n.startswith(('vas','recfem'))];R=ts['R'][kt,keep];target=ts['target'][kt,keep];refs=ts['reference_N'];C,bound=reaction_cone(ts['S'][kt],ts['offset'][kt],ts['axes'][kt],20.,16);lower_bounds=[]
    for label,floor in [('baseline passive floor',ts['passive'][kt]),('zero passive diagnostic',np.zeros(len(refs)))]:
        A=R*refs;scale=np.maximum(np.linalg.norm(A,axis=1),1);U=C*refs;us=np.maximum(np.linalg.norm(U,axis=1),1);cost=np.zeros(len(refs));cost[quad]=refs[quad]/2;answer=linprog(cost,A_eq=A/scale[:,None],b_eq=(target-R@floor)/scale,A_ub=U/us[:,None],b_ub=(bound-C@floor)/us,bounds=(0,None),method='highs');assert answer.success;F=floor+refs*answer.x
        lower_bounds.append(dict(case=label,quadriceps_lower_bound_mean_per_side_N=float(F[quad].sum()/2),maximum_any_element_N=float(F.max()),moment_balance_error_Nm=float(abs(R@F-target).max()),scope='A mathematical lower bound under existing geometry and constraints, not physiological recruitment. It shifts large loads to other muscles and is not applied.'))
    xml=ROOT/'.venv/Lib/site-packages/myo_sim/models/leg/README.md';notes=xml.read_text(encoding='utf-8');assert 'Vastus muscle moment arms' in notes
    result=dict(status='geometry_fidelity_review_incomplete',current_peak_knee_flexion_deg=float(np.rad2deg(angle)),moment_arm_definition='Signed knee extension leverage, -dL/d(knee flexion), cm. MyoSim divided by uniform subject length scale for lineage comparison.',peak_extension_leverage_cm=peak,converted_vs_original_max_absolute_difference_cm=dict(zip(MUSCLES,maxdiff.tolist())),numerical_derivative_checks=derivatives,upstream_known_limitation=dict(file=str(xml),sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),finding='MyoSim leg README explicitly lists relatively large vastus knee moment-arm differences from OpenSim, of a few cm, associated with dependent knee constraints and knee extensor force differences.'),canonical_forces_changed=False,canonical_motion_changed=False,human_force_validated=False,comparison_limits=['Same named flexion coordinate and zero other joint coordinates, not matched subject-specific anatomy.','The 2023 and Catelli comparators intentionally differ from the original lineage.','A geometry discrepancy is evidence to investigate, not proof that a particular human force should equal another model force.','The signed leverage is a whole coupled-path derivative, not simply the distance from one tendon line to a fixed hinge.'],references=['https://github.com/opensim-org/opensim-models/blob/master/Models/Rajagopal/README.txt','https://pmc.ncbi.nlm.nih.gov/articles/PMC5989715/'])
    (OUT/'geometry_audit.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/'moment_arm_curves.npz',knee_flexion_deg=deg,model_labels=np.array(list(curves)),muscle_names=np.array(MUSCLES),moment_arms_m=np.array(list(curves.values())))
    result.update(force_sharing_lower_bound_diagnostics=lower_bounds,canonical_motion_sha256=hashlib.sha256((ROOT/'results/measured_jump_analysis/s10_CMJ_t1/motion.npz').read_bytes()).hexdigest(),canonical_tensions_sha256=hashlib.sha256((ROOT/'results/measured_jump_analysis/s10_CMJ_t1/tensions.npz').read_bytes()).hexdigest(),finding='At about 96 degrees, converted vastus leverage is larger, not smaller, than the original and revised OpenSim comparators. The high quadriceps force cannot be assigned to an excessively short converted moment arm on this comparison alone. The source models themselves are not experimental ground truth.')
    (OUT/'geometry_audit.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,ax in enumerate(axes.flat):
        for label,values in curves.items():ax.plot(deg,-100*values[:,j],label=label,lw=2)
        ax.axvline(np.rad2deg(angle),color='#555',ls='--',lw=1);ax.set_title(MUSCLES[j]);ax.set_xlabel('Knee flexion [deg]');ax.set_ylabel('Extension leverage [cm]');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Knee extensor geometry audit: converted model vs source lineage',fontsize=16);fig.savefig(OUT/'moment_arm_comparison.png',dpi=170);plt.close(fig)
    rows=''.join('<tr><td>'+label+'</td>'+''.join(f'<td>{peak[label][n]:.2f}</td>' for n in MUSCLES)+'</tr>' for label in peak)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Knee model validity</title><style>body{{max-width:1050px;margin:30px auto;padding:20px;background:#0d1726;color:#edf3fb;font:17px/1.6 system-ui}}a{{color:#8ce2d5}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #52687a;padding:8px}}.scroll{{overflow:auto}}</style><a href="../measured_jump_visualization/index.html">Jump analysis</a><h1>Knee model validity audit</h1><p>Model fidelity and realistic results require anatomical and empirical checks beyond numerical balance. The current 10.23 kN quadriceps peak remains a conditional estimate under review.</p><p>The installed MyoSim leg documentation identifies a known vastus moment-arm discrepancy from its source OpenSim model. Here the current converted geometry is compared with the original lineage and two separate improved references.</p><img src="moment_arm_comparison.png" alt="Four knee extensor leverage curves"><h2>Leverage at current peak knee flexion: {np.rad2deg(angle):.1f}°</h2><p>Centimeters. MyoSim uniform subject scale removed; this is not a personal anatomical measurement.</p><div class="scroll"><table><tr><th>Model</th>{''.join('<th>'+n+'</th>' for n in MUSCLES)}</tr>{rows}</table></div><p>Numerical derivative agreement only establishes consistency within each model. It does not establish human anatomy or human force accuracy. The actual source muscle paths, dependent knee translations/rotations and patellar motion need a matched conversion review before correction.</p><p><a href="geometry_audit.json">Audit data and limits</a> · <a href="moment_arm_curves.npz">Curve data</a> · <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5989715/">Lai et al. 2017: anatomy and anomalous co-activation</a> · <a href="https://github.com/opensim-org/opensim-models/blob/master/Models/Rajagopal/README.txt">Official OpenSim model lineage</a></p></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8');print('KNEE_VALIDITY_RESULT',json.dumps(dict(peak_knee_deg=result['current_peak_knee_flexion_deg'],peak_extension_leverage_cm=peak,max_original_difference_cm=result['converted_vs_original_max_absolute_difference_cm'])),flush=True)
    text=(OUT/'index.html').read_text(encoding='utf-8');text=text.replace('<h2>Leverage at current peak','<p><strong>Finding:</strong> At this flexed posture the converted vastus leverage is larger than the original and revised references. A short converted lever arm is not established as the cause of the high force. Force-sharing assumptions and coupled-joint demand require further checks.</p><h2>Leverage at current peak');(OUT/'index.html').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
