"""Compare depth under fixed load, body, timing and high-bar posture strategy."""
from pathlib import Path
import json,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/power_comparison'
def main():
 out=BASE/'squat_depth_pilot';out.mkdir(exist_ok=True);records=[];series=[]
 for case in ['high_bar_100kg_knee60','high_bar_100kg_knee90','high_bar_100kg_full']:
  folder=BASE/case;r=json.loads((folder/'summary.json').read_text());z=np.load(folder/'motion.npz');assert r['all_mechanics_gates_passed']
  combined=(r['body_mass_kg']*z['com']+r['bar_mass_kg']*z['bar_position_m'])/(r['body_mass_kg']+r['bar_mass_kg'])
  err=float(abs(combined-z['system_COM_m']).max());assert err<1e-10
  still=z['times']<r['phase_times_s']['start_descent']-.03
  static_err=float(np.max(abs(z['GRF_N'][still,2]-(r['body_mass_kg']+r['bar_mass_kg'])*9.80665)));assert static_err<.01
  coords=r['coordinates'];moments=np.array(r['peak_net_joint_moments_Nm'])
  row=dict(case=case,bottom_knee_flexion_deg=r['depth']['bottom_knee_flexion_deg'],bar_mass_kg=r['bar_mass_kg'],body_mass_kg=r['body_mass_kg'],height_m=r['height_m'],bar_travel_m=r['bar_travel_m'],peak_GRF_N=r['peak_GRF_N'],
    peak_hip_net_moment_Nm=float(moments[coords.index('hip_flexion_r')]),peak_knee_net_moment_Nm=float(moments[coords.index('knee_angle_r')]),peak_ankle_net_moment_Nm=float(moments[coords.index('ankle_angle_r')]),
    peak_bar_power_W=r['peak_bar_power_W'],combined_COM_identity_max_error_m=err,static_weight_balance_max_error_N=static_err,
    max_COP_COM_horizontal_distance_m=float(np.linalg.norm(z['COP_m'][:,:2]-z['system_COM_m'][:,:2],axis=1).max()),all_mechanics_gates_passed=True)
  records.append(row);series.append((r,z))
 with (out/'comparison.csv').open('w',newline='',encoding='utf-8')as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 report=dict(status='mechanically_checked_prescribed_high_bar_depth_pilot',conditions=dict(height_m=1.8,body_mass_kg=88,bar_mass_kg=100,descent_s=1.25,hold_s=.15,ascent_s=.9),
   interpretation='Net joint moments, GRF and bar power are mechanical outputs. Individual muscle tensions have not been computed for these new cases.',
   balance='Horizontal combined body-plus-bar COM is fitted to the support centre in the prescribed posture family. COP is calculated from the dynamic ground wrench, and can differ from COM projection.',
   validation='Equation, engine, contact, grasp and energy checks passed; no claim of measured human-motion or muscle-force accuracy.',rows=records)
 (out/'comparison.json').write_text(json.dumps(report,indent=2))
 fig,axes=plt.subplots(2,2,figsize=(12,8));colors=['#2c7c9a','#e99c30','#a55799']
 for row,(r,z),c in zip(records,series,colors):
  name=f'{row["bottom_knee_flexion_deg"]:.1f} deg';t=z['times'];coords=r['coordinates']
  axes[0,0].plot(t,z['bar_position_m'][:,2],label=name,color=c);axes[0,1].plot(t,z['GRF_N'][:,2],label=name,color=c)
  axes[1,0].plot(t,z['net_joint_moments_Nm'][:,coords.index('knee_angle_r')],label=name,color=c)
  axes[1,1].plot(t,z['COP_m'][:,1]-z['system_COM_m'][:,1],label=name,color=c)
 for ax,label in zip(axes.ravel(),['Bar height (m)','Total vertical GRF (N)','Right knee net moment (Nm)','COP - combined COM, fore-aft (m)']):ax.set(xlabel='Time (s)',ylabel=label);ax.grid(alpha=.25);ax.legend(fontsize=9)
 fig.suptitle('High-bar depth pilot | 180 cm / 88 kg / 100 kg | Same cadence, prescribed balance strategy');fig.tight_layout();fig.savefig(out/'depth_comparison.png',dpi=140);plt.close(fig)
 cells=[]
 for r in records:cells.append('<tr>'+''.join(f'<td>{r[k]:.2f}</td>'for k in ['bottom_knee_flexion_deg','bar_travel_m','peak_GRF_N','peak_hip_net_moment_Nm','peak_knee_net_moment_Nm','peak_ankle_net_moment_Nm'])+'</tr>')
 (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>High-bar depth pilot</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:35px auto;padding:20px;background:#101923;color:#e7f1f6}a{color:#6ce0d4}img{width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #40505f}table{border-collapse:collapse}small{color:#b0c0d0}</style><h1>High-bar squat: depth pilot</h1><p>180 cm / 88 kg / 100 kg bar. Descent 1.25 s; hold 0.15 s; ascent 0.90 s. Depth is anatomical knee flexion, not a thigh-parallel label.</p><table><tr><th>Knee flexion (deg)</th><th>Bar travel (m)</th><th>Peak GRF (N)</th><th>Hip (Nm)</th><th>Knee (Nm)</th><th>Ankle (Nm)</th></tr>'''+''.join(cells)+'''</table><p>Moments: peak absolute net moments on the right side. These are not individual muscle tensions.</p><img src="depth_comparison.png" alt="Time histories at three squat depths"><p>Body and bar mass positions both enter the combined COM. The posture family holds its horizontal projection at the support centre. Dynamic COP is a separate computed result.</p><small>Mechanical consistency checks passed; human movement and muscle forces are not experimentally validated for these cases.</small><p><a href="comparison.json">Checks and values</a> | <a href="comparison.csv">CSV</a> | <a href="../../../docs/modeling_references.html">Primary reference library</a></p></html>''',encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
