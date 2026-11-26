"""Task-specific path/virtual-work verification and independent model screening.

Published reference-model moment arms are a cross-model screen, not measured
individual forces or an acceptance interval fitted to the present output.
"""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco,opensim as osim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retarget_measured_jump import MeasuredJumpRig
from high_flex_profile import apply
ROOT=Path(__file__).resolve().parents[1]
STEMS=['vasint','vaslat','vasmed','recfem','bflh','bfsh','semimem','semiten','gasmed','gaslat','soleus','tibant','glmax1','glmax2','glmax3']


def main():
    p=argparse.ArgumentParser();p.add_argument('--jump',default='results/athletic_box_jump/yun_160_supported_cycle_v2');p.add_argument('--label',default='yun_160_240_paths');p.add_argument('--published-patella',action='store_true');args=p.parse_args();out=ROOT/'results/muscle_validation'/args.label;out.mkdir(parents=True,exist_ok=True)
    r=MeasuredJumpRig(87,1.78);apply(r);
    if args.published_patella:
        from published_patella_profile import apply as apply_patella
        apply_patella(r)
    b,m,d=r.b,r.m,r.d;names=b.names;selected=[i for i,n in enumerate(names) if n.endswith('_r') and n[:-2] in STEMS];coordinates=['hip_flexion_r','knee_angle_r','ankle_angle_r','flex_extension','shoulder_elv_r'];rows=[];source_stats={};worst=0.
    arch=ET.parse(ROOT/'models/rmr/TSM_subject_noWeight.osim');lat=arch.find('.//Millard2012EquilibriumMuscle[@name="LatissimusDorsi_S"]');lr=m.actuator_lengthrange.copy();g=b.report['global_normalization_scale'];l0=float(lat.findtext('optimal_fiber_length'))*g;lt=float(lat.findtext('tendon_slack_length'))*g
    for j,name in enumerate(names):
        if name in ['LAT1','LAT1_l']:lr[j]=lt+m.actuator_gainprm[j,:2]*l0
    lopt=(lr[:,1]-lr[:,0])/(m.actuator_gainprm[:,1]-m.actuator_gainprm[:,0]);slack=lr[:,0]-m.actuator_gainprm[:,0]*lopt
    for label,folder in [('box_jump',ROOT/args.jump),('squat_240',ROOT/'results/power_comparison/yun_sungbin/squat_240')]:
        z=dict(np.load(folder/'motion.npz'));indices=np.unique(np.r_[np.linspace(0,len(z['times'])-1,25).round().astype(int),int(np.argmax(z['qpos'][:,m.jnt_qposadr[b.joints['knee_angle_r']]]))]);lengths=[];passives=[];ratios=[];maxfd={c:0. for c in coordinates};forceR=[]
        for index in indices:
            q=b.complete(z['qpos'][index]);u=z['u'][index] if 'u' in z else np.r_[z['qvel'][index,:6],z['qvel'][index,r.dofs]];B,_=r.set_state(q,u);b.pose(q,B@u);R=(B.T@b.moments())[6:];lengths.append(d.actuator_length.copy());ratios.append((d.actuator_length-slack)/lopt);passives.append(np.array([max(0.,-mujoco.mju_muscleBias(float(d.actuator_length[j]),np.ascontiguousarray(lr[j]),float(m.actuator_acc0[j]),np.ascontiguousarray(m.actuator_biasprm[j,:9]))) for j in range(m.nu)]))
            for coordinate in coordinates:
                k=b.ind_names.index(coordinate);adr=r.ids[k];values=[]
                for epsilon in [1e-5,5e-6]:
                    ls=[]
                    for sign in [-1,1]:qp=q.copy();qp[adr]+=sign*epsilon;b.pose(qp);ls.append(d.actuator_length.copy())
                    fd=-(ls[1]-ls[0])/(2*epsilon);values.append(fd)
                error=abs(R[k]-values[-1]);j=int(error.argmax());maxfd[coordinate]=max(maxfd[coordinate],float(error[j]));worst=max(worst,float(error[j]));rows.append(dict(case=label,time_s=float(z['times'][index]),coordinate=coordinate,maximum_error_m=float(error[j]),element=names[j],finite_difference_refinement_max_m=float(abs(values[0]-values[1]).max())))
            forceR.append(R[:,selected]);print('PATH_AUDIT',label,int(index),flush=True)
        ratio=np.array(ratios);pas=np.array(passives);source_stats[label]=dict(source_motion_sha256=hashlib.sha256((folder/'motion.npz').read_bytes()).hexdigest(),sample_count=len(indices),maximum_virtual_work_error_m=maxfd,maximum_passive_element_N=float(pas.max()),maximum_passive_element=names[int(pas.max(0).argmax())],nonpositive_fiber_mapping_elements=[names[i] for i in np.flatnonzero(ratio.min(0)<=0)],passive_peak_by_element_N=dict(zip(names,np.round(pas.max(0),6).tolist())),normalized_fiber_length_min=dict(zip(names,np.round(ratio.min(0),6).tolist())),normalized_fiber_length_max=dict(zip(names,np.round(ratio.max(0),6).tolist())),knee_flexion_max_deg=float(np.rad2deg(z['qpos'][:,m.jnt_qposadr[b.joints['knee_angle_r']]].max())))
        np.savez_compressed(out/(label+'_path_samples.npz'),times=z['times'][indices],length_m=lengths,normalized_fiber_length=ratio,passive_force_N=pas,reduced_selected_moment_arms_m=forceR,indices=indices)
    # Same joint-angle sweep, two independent published high-flexion models.
    # Registration and anthropometry differ, so deviations are not force error.
    models={}
    for key,relative in [('Catelli','datasets/cmj_calisti2025/source/catelli_3dof_model.osim'),('RajagopalLaiUhlrich2023','models/high_flex/RajagopalLaiUhlrich2023.osim')]:
        osim.Logger.setLevelString('error');model=osim.Model(str(ROOT/relative));state=model.initSystem();models[key]=(model,state,relative)
    angles=np.array([0,30,60,90,110,120,130,140,145.]);realized={key:[] for key in models};sweeps={key:[] for key in ['native',*models]};neutral=b.q0.copy()
    for degrees in angles:
        q=neutral.copy();q[m.jnt_qposadr[b.joints['hip_flexion_r']]]=np.deg2rad(60);q[m.jnt_qposadr[b.joints['knee_angle_r']]]=np.deg2rad(degrees);B,_=r.set_state(q,np.zeros(r.n));b.pose(q);R=(B.T@b.moments())[6:];k=b.ind_names.index('knee_angle_r');sweeps['native'].append(R[k,selected])
        for key,(model,state,_) in models.items():
            coords=model.updCoordinateSet();coords.get('hip_flexion_r').setValue(state,np.deg2rad(60),True);coords.get('knee_angle_r').setValue(state,np.deg2rad(degrees),True);model.realizePosition(state);knee=coords.get('knee_angle_r');realized[key].append(float(np.rad2deg(knee.getValue(state))));sweeps[key].append([model.getMuscles().get(names[j]).computeMomentArm(state,knee) for j in selected])
    sweeps={k:np.array(v) for k,v in sweeps.items()};np.savez_compressed(out/'knee_reference_sweep.npz',knee_flexion_deg=angles,names=np.array([names[j] for j in selected]),**sweeps)
    comparison=[]
    for j,index in enumerate(selected):
        if names[index][:-2] not in ['vasint','vaslat','vasmed','recfem','bflh','bfsh','semimem','semiten','gasmed','gaslat']:continue
        comparison.append(dict(element=names[index],native_range_mm=(sweeps['native'][:,j]*1000).tolist(),Catelli_range_mm=(sweeps['Catelli'][:,j]*1000).tolist(),RajagopalLaiUhlrich2023_range_mm=(sweeps['RajagopalLaiUhlrich2023'][:,j]*1000).tolist(),sign_disagreement_with_both_references=bool(np.any((abs(sweeps['native'][:,j])>.002)&(np.sign(sweeps['native'][:,j])!=np.sign(sweeps['Catelli'][:,j]))&(np.sign(sweeps['native'][:,j])!=np.sign(sweeps['RajagopalLaiUhlrich2023'][:,j]))))))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,name in zip(axes.flat,['vasint','vaslat','vasmed','recfem']):
        j=[names[i] for i in selected].index(name+'_r')
        for key,data in sweeps.items():ax.plot(angles,-data[:,j]*1000,'o-',label=key)
        ax.set(title=name,xlabel='Knee flexion [deg]',ylabel='Extension moment arm [mm]');ax.axvspan(120,145,color='#df9e56',alpha=.12);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Knee muscle paths: cross-model screening; not subject validation');fig.savefig(out/'quadriceps_moment_arms.png',dpi=160);plt.close(fig)
    report=dict(status='virtual_work_verified_reference_screen_available' if worst<1e-4 else 'path_derivative_review_required',maximum_virtual_work_error_m=worst,virtual_work_tolerance_m=1e-4,scope='Native path geometry and generic rigid-tendon mapping; not independent muscle-force validation.',LAT1_mapping_repair='Existing project repair from RMR/Seth2019 standard upper-lat architecture; same in both cases.',cases=source_stats,derivative_checks=rows,knee_reference_screen=comparison,reference_model_files={key:dict(file=relative,sha256=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()) for key,(_,_,relative) in models.items()},human_muscle_force_validated=False,force_clipping=False,reference_realized_knee_angles_deg=realized,reference_angle_warning='Compare at realized reference angles; OpenSim assembly can clamp coordinates outside the source model range.')
    (out/'path_audit.json').write_text(json.dumps(report,indent=2));print('MUSCLE_PATH_RESULT',report['status'],worst,flush=True)


if __name__=='__main__':main()



