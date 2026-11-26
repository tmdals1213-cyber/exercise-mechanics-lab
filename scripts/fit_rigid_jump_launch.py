"""Physics-informed initialization of launch; no launch state is injected.

Only parameters that can causally affect ground propulsion enter this block.
The final full-jump policy retains every posture/timing output.
"""
import json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from controlled_broad_jump import ROOT,run,build

KEYS=['hip_extension_duration_fraction','knee_delay','ankle_delay','arm_delay_s','prepush_lean_s','hip_top','pitch_top','hip_bottom','knee_bottom','shoulder_top','push_s','trunk_bottom']
BOUNDS=np.array([[.4,1.1],[0,.6],[0,.8],[-.2,.12],[0,.25],[-.2,.4],[.8,1.2],[1.6,1.9],[1.05,1.4],[1.3,2.],[.20,.36],[-.25,0.]],float)
class Ready(Exception):pass
def main():
 source=ROOT/'results/controlled_broad_jump/rigid_joint_launch_timing_v25/candidate_03/control_summary.json';base=json.loads(source.read_text())['parameters'];base.update(seconds=3.,stop_after_takeoff_s=.025,record_energy_ledger=False);base.update(hip_extension_duration_fraction=.6,knee_delay=.35);prepared=build(.0005);out=ROOT/'results/controlled_broad_jump/rigid_launch_fit_v28';out.mkdir(exist_ok=True);low=BOUNDS[:,0];span=BOUNDS[:,1]-low;x0=(np.array([base[k]for k in KEYS])-low)/span;history=[];best=None;start=time.time()
 def f(x):
  nonlocal best
  p=base.copy();p.update(dict(zip(KEYS,map(float,low+span*x))));r,z=run(p,dt=.0005,prepared=prepared);v=r.get('takeoff_velocity_m_s',[0,0,0]);vx=-v[1];vz=v[2];H=r.get('takeoff_H_kg_m2_s',[100])[0];C=r.get('takeoff_COM_m',[0,0,0]);pitch=r.get('takeoff_root_pitch_rad',2.);peak=r['peak_GRF_BW'];a=z['qpos'][:,3:];limits=prepared[1].jnt_range[3:];rom=float(np.maximum(a-limits[:,1],limits[:,0]-a).max());proxy=-1.;lead=0.
  if r['takeoff_s']is not None and C[2]>.75:
   contact_vz=-math.sqrt(vz*vz+2*9.80665*(C[2]-.75));T=(vz-contact_vz)/9.80665;lead=((.25*.75*vx-H/87)+.75*(-vx))/contact_vz;proxy=-C[1]+vx*T+lead+min(pt[2][1]for pt in prepared[6])
  residual=np.r_[(vx-2.5)/.7,(vz-2.2)/.7,H/12,3*max(2.12-proxy,0),2*max(.85-C[2],0),.4*(pitch-.65),max(peak-5.,0),10*max(rom-.025,0),.025*(x-x0)]
  if r['takeoff_s']is None or r['fell']:residual[:3]+=10
  score=float(residual@residual);row=dict(trial=len(history),score=score,vx=vx,vz=vz,Hx=H,COM=C,root_pitch=pitch,balanced_landing_distance_proxy_m=proxy,foot_lead_proxy_m=lead,peak_propulsion_BW=peak,max_ROM_excess_rad=rom,parameters=p);history.append(row)
  if best is None or score<best['score']:best=row;(out/'best_launch.json').write_text(json.dumps(best,indent=2))
  with(out/'trials.jsonl').open('a')as logfile:logfile.write(json.dumps(row)+'\n')
  if len(history)%5==1:print('RIGID_LAUNCH_FIT',len(history),round(score,3),'v',round(vx,2),round(vz,2),'H',round(H,1),'proxy',round(proxy,2),flush=True)
  if vx>1.8 and vz>1.7 and abs(H)<15 and proxy>2.12 and .1<lead<.65 and peak<5.5 and rom<.04 and not r['fell']:
   (out/'ready_launch.json').write_text(json.dumps(row,indent=2));raise Ready()
  return residual
 try:result=least_squares(f,x0,bounds=(np.zeros(len(x0)),np.ones(len(x0))),jac='2-point',diff_step=.03,max_nfev=60,ftol=1e-4,xtol=1e-4,gtol=1e-4)
 except Ready:pass
 (out/'summary.json').write_text(json.dumps(dict(best=best,physical_evaluations=len(history),wall_s=time.time()-start,ready=(out/'ready_launch.json').exists(),goal='Develop a balanced initial launch using actual native outputs; proxy only ranks candidates and is never applied to the body.'),indent=2));print('RIGID_LAUNCH_FIT_DONE',len(history),best['score'],flush=True)
if __name__=='__main__':main()
