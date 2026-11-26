"""Check physical residuals, sampling sensitivity and the displayed summaries."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from case_contours import ROWS,overview

ROOT=Path(__file__).resolve().parents[1]

def main(kg):
    folder=ROOT/f'results/power_comparison/squat_{kg}' if kg else ROOT/'results/power_comparison/cmj_analysis';source=folder/'analysis' if kg else folder;visual=source/'visualization'
    if kg:meta=json.loads((folder/'summary.json').read_text())
    else:
        setup=json.loads((source/'setup.json').read_text());meta=dict(phase_times_s=dict(start_ascent=setup['countermovement_end_s'],end_ascent=setup['takeoff_s']))
    summary=json.loads((source/'tension_summary.json').read_text());data=json.loads((visual/'data.json').read_text());fine=np.load(source/'tensions.npz');coarse=np.load(source/('batch_30hz.npz' if (source/'batch_30hz.npz').exists() else 'tensions_30hz.npz'))
    t=fine['times'];F=fine['force_N'];passive=fine['passive_force_N'];selected=[i for i,n in enumerate(summary['coordinate_names']) if not n.startswith('mtp_angle_')]
    residual=fine['target']-np.einsum('tqn,tn->tq',fine['R'],F);max_error=float(abs(residual[:,selected]).max())
    means=np.trapezoid(overview(data),np.array(data['times']),axis=1)/(data['times'][-1]-data['times'][0]);peak=overview(data).max(1)
    rows=[];convergence=[];lever=[]
    for j,(label,keys) in enumerate(ROWS):
        ids=[idx for g in data['groups'] if g['key'] in keys for idx in g['indices']]
        y=F[:,ids].sum(1)/2;yc=coarse['force_N'][:,ids].sum(1)/2;yi=np.interp(coarse['times'],t,y)
        rms=float(np.sqrt(np.mean((yc-yi)**2))/max(100.,np.sqrt(np.mean(yi**2))));pd=float(abs(yc.max()-y.max())/max(100.,y.max()))
        convergence.append(dict(group=label,relative_curve_RMS_difference=rms,relative_peak_difference=pd,low_force_normalizer_N=100))
        rows.append(dict(group=label,mean_per_side_N=float(means[j]),peak_per_side_N=float(peak[j]),peak_time_s=float(t[np.argmax(y)])))
    for side in ['r','l']:
        g=next(g for g in data['groups'] if g['key']=='quadriceps' and g['side']==side);ids=g['indices'];q=summary['coordinate_names'].index('knee_angle_'+side);force=F[:,ids].sum(1);k=int(force.argmax());contribution=float(fine['R'][k,q,ids]@F[k,ids])
        lever.append(dict(side=side,time_s=float(t[k]),quadriceps_sum_N=float(force[k]),quadriceps_knee_moment_Nm=contribution,
                          signed_effective_moment_arm_m=contribution/force[k],effective_moment_arm_m=abs(contribution)/force[k],required_net_knee_moment_Nm=float(fine['target'][k,q]),
                          other_muscle_knee_moment_Nm=float(fine['R'][k,q]@F[k]-contribution),passive_quadriceps_sum_N=float(passive[k,ids].sum())))
    gates=dict(moment_balance=max_error<.01,nonnegative=bool(F.min()>-1e-7),passive_floor=bool((F-passive).min()>-1e-6),
               overview_sample_sensitivity=all(max(r['relative_curve_RMS_difference'],r['relative_peak_difference'])<.10 for r in convergence),
               final_source_time_included=abs(t[-1]-np.load(source/'motion.npz')['times'][-1])<1e-9)
    gates={k:bool(v) for k,v in gates.items()}
    report=dict(status='passed' if all(gates.values()) else 'needs_review',gates=gates,moment_balance_max_error_Nm=max_error,
                force_sampling_Hz=60,comparison_sampling_Hz=30,selection_mode=summary.get('selection_mode'),convergence=convergence,convergence_gate_is_numerical_choice=.10,
                muscle_tension_is_independently_validated=False,strength_cap_used=False,scope='One conditional force-sharing solution, not EMG, stress or damage.',
                mean_scope='One complete model cycle; replay holds excluded.',muscle_overview=rows,quadriceps_moment_audit=lever,
                source_motion_sha256=summary['source_motion_sha256'],force_file_sha256=hashlib.sha256((source/'tensions.npz').read_bytes()).hexdigest(),
                warnings=['Bar load enters the rigid upper back. Hand/back pressure sharing is not solved.','MTP intrinsic/fascial support remains an unidentified internal joint moment.','Same-model sampling consistency is not biological validation.'])
    (folder/'tension_audit.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    for y,(label,_) in zip(overview(data),ROWS):axes[0].plot(data['times'],y/1000,label=label)
    jf=np.array(data['joint_force_N'])
    for stem in ['shoulder','elbow','wrist','hip','knee','ankle']:
        ids=[i for i,j in enumerate(data['joints']) if j['key'].startswith(stem+'_')];axes[1].plot(data['times'],jf[ids].mean(0)/1000,label=stem.title())
    for ax,title in zip(axes,['Conditional muscle-group tension','Conditional joint-region reaction; includes selected muscle forces']):
        ax.set(title=title,xlabel='Physical time [s]',ylabel='Mean per side [kN]');ax.legend(ncol=3,fontsize=9);ax.grid(alpha=.2);ax.axvspan(meta['phase_times_s']['start_ascent'],meta['phase_times_s']['end_ascent'],color='#d4a243',alpha=.08)
    title=f'High-bar squat / {kg} kg' if kg else 'CMJ measured-motion reference'
    fig.suptitle(title+' · conditional sharing, not measured recruitment',fontsize=16);fig.savefig(folder.parent/(f'squat_{kg}_conditional_tension_curves.png' if kg else 'cmj_conditional_tension_curves.png'),dpi=150);plt.close(fig)
    print(json.dumps(report['gates']),flush=True)
    if not all(gates.values()):raise RuntimeError(('Tension checks require review',gates))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kg',type=int,choices=[0,60,100,140],default=100);main(p.parse_args().kg)
