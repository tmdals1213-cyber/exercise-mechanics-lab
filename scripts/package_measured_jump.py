"""Deliver the existing anatomical view with measured-motion diagnostics."""
import json,os,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/os.environ.get('EXERCISE_JUMP_VISUALS','results/measured_jump_visualization')

def main():
    data=json.loads((OUT/'data.json').read_text());t=np.array(data['times']);F=np.array(data['group_tension_N']);J=np.array(data['joint_force_N']);groups=data['groups'];to=data['takeoff_s'];bottom=data['countermovement_end_s'];source=ROOT/os.environ.get('EXERCISE_JUMP_SOURCE','results/measured_jump_analysis/s10_CMJ_t1');motion=np.load(source/'motion.npz');tension=np.load(source/'tensions.npz');verification=json.loads((source/'verification.json').read_text());solver=json.loads((source/'tension_summary.json').read_text())
    def series(keys):return F[[i for i,g in enumerate(groups) if g['key'] in keys]].sum(0)/2
    rows=[('Gluteals',['gluteals']),('Quadriceps',['quadriceps']),('Hamstrings',['hamstrings']),('Calf',['gastrocnemius','soleus']),('Abdominal wall',['abdominals']),('Lumbar / back',['spinal_extensors','quadratus_lumborum']),('Latissimus',['latissimus']),('Trapezius',['trapezius'])]
    summary=dict(status='conditional_measured_motion_analysis',trial='s10_CMJ_t1',height_m=data['height_m'],mass_kg=data['mass_kg'],muscle_elements=data['muscle_count'],force_samples=len(t),physics_hz=125,measured_GRF_peak_N=float(motion['force_world_N'][:,2].max()),verification=verification,kinematic_retarget=json.loads((source/'kinematic_retarget.json').read_text()),force_solver_checks=solver['checks'],muscle_force_validated=False,landing_included=False,root_residual_applied=False,means_scope='Quiet lead-in through flight, excluding landing; repeats are display replays.',group_statistics=[],joint_statistics=[])
    phase_masks={'countermovement':t<bottom,'propulsion':(t>=bottom)&(t<to),'flight':t>=to}
    for label,keys in rows:
        y=series(keys);i=int(y.argmax());summary['group_statistics'].append(dict(group=label,mean_per_side_N=float(np.trapezoid(y,t)/(t[-1]-t[0])),peak_per_side_N=float(y[i]),peak_time_s=float(t[i]),phase_peak_N={key:float(y[mask].max()) if mask.any() else None for key,mask in phase_masks.items()}))
    for key in ['shoulder','elbow','wrist','hip','knee','ankle']:
        y=J[[i for i,j in enumerate(data['joints']) if j['key'].startswith(key+'_')]].mean(0);summary['joint_statistics'].append(dict(joint=key,mean_per_side_N=float(np.trapezoid(y,t)/(t[-1]-t[0])),peak_per_side_N=float(y.max())))
    fig,ax=plt.subplots(2,2,figsize=(14,8),layout='constrained');a=ax[0,0];a.plot(motion['times'],motion['force_world_N'][:,2],label='Measured force plates',color='#00838f');a.plot(motion['times'],motion['force_world_N'][:,2]+motion['root_force_residual_N'][:,2],label='Implied by retargeted motion',color='#df6b22');a.set_title('Both-feet vertical ground reaction');a.set_ylabel('N');a.legend(fontsize=9)
    for a,selected,title in [(ax[0,1],rows[:4],'Lower-body muscle-group tension'),(ax[1,0],rows[4:],'Trunk and back muscle-group tension')]:
        for label,keys in selected:a.plot(t,series(keys),label=label,lw=2)
        a.set_title(title);a.set_ylabel('Mean per side [N]');a.legend(fontsize=9)
    a=ax[1,1]
    for key in ['hip','knee','ankle','shoulder']:
        y=J[[i for i,j in enumerate(data['joints']) if j['key'].startswith(key+'_')]].mean(0);a.plot(t,y,label=key.title(),lw=2)
    a.set_title('Joint cut reaction after muscle forces');a.set_ylabel('Mean per side [N]');a.legend(fontsize=9)
    for a in ax.flat:
        a.axvspan(bottom,to,color='#e9b54e',alpha=.1);a.axvline(to,color='#555',ls='--',lw=1);a.set_xlabel('Time from replay start [s]');a.grid(alpha=.2);a.set_xlim(t[0],t[-1])
    fig.suptitle('Measured CMJ → full-body muscle model · 180 cm / 88 kg',fontsize=18);fig.text(.5,-.025,'Conditional tension allocation. Muscle-group scalar sums and joint cut reactions are different quantities. Landing excluded.',ha='center',fontsize=10);fig.savefig(OUT/'analysis_curves.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2));shutil.copyfile(source/'verification.json',OUT/'retarget_checks.json');shutil.copyfile(source/'tension_summary.json',OUT/'tension_summary.json');shutil.copyfile(ROOT/'assets/fullbody/CREDITS.md',OUT/'CREDITS.md')
    shutil.copyfile(source/'kinematic_retarget.json',OUT/'kinematic_retarget.json')
    if (source/'quadriceps_peak_audit.json').exists():shutil.copyfile(source/'quadriceps_peak_audit.json',OUT/'quadriceps_peak_audit.json')
    page=(ROOT/'scripts/measured_jump_viewer.html').read_text(encoding='utf-8');(OUT/'index.html').write_text(page,encoding='utf-8')
    if (ROOT/'results/knee_model_validity/index.html').exists():
        page=page.replace('<details><summary>High quadriceps estimate</summary>','<p><a href="../knee_model_validity/index.html">Knee model validity: converted geometry versus source models</a></p><details><summary>High quadriceps estimate</summary>');(OUT/'index.html').write_text(page,encoding='utf-8')
    if (ROOT/'results/power_comparison/index.html').exists():
        target='comparison.html' if (ROOT/'results/power_comparison/comparison.html').exists() else 'index.html'
        page=page.replace('<a href="analysis_curves.png">All curves</a>', f'<a href="analysis_curves.png">All curves</a> · <a href="../power_comparison/{target}">Power comparison · natural-speed replay</a>');(OUT/'index.html').write_text(page,encoding='utf-8')
    print('MEASURED_JUMP_PACKAGE',json.dumps({k:summary[k] for k in ['status','force_samples','measured_GRF_peak_N','group_statistics','joint_statistics']}),flush=True)

if __name__=='__main__':main()
