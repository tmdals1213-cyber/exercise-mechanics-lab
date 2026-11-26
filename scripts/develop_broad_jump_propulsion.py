"""Search a plausible launch seed with native ground contact, never a launch kick.

Velocity targets are initialization objectives, not injected initial states.
This is controller development, not a measurement of human maximum performance.
"""
import argparse,json,math
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import differential_evolution
from controlled_broad_jump import ROOT,DEFAULT,build,run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='propulsion_development_v1');ap.add_argument('--iterations',type=int,default=3);a=ap.parse_args();out=ROOT/'results/controlled_broad_jump'/a.label;out.mkdir(parents=True,exist_ok=True);base=DEFAULT.copy();base.update(json.loads((ROOT/'results/controlled_broad_jump/broad_jump_seed_refined_v4/candidate_01/control_summary.json').read_text())['parameters']);base.update(seconds=2.5,stop_after_takeoff_s=.045);prepared=build(.001);rows=[]
 keys=['lean_m','pitch_top','hip_top','push_s','ankle_delay','arm_delay_s','leg_kp','ankle_kp'];bounds=[(.28,.65),(.55,1.15),(-.20,.55),(.20,.40),(.20,.75),(-.16,.04),(500,1400),(200,650)]
 def save(rep,z):
  p=out/'best_launch';p.mkdir(exist_ok=True);np.savez_compressed(p/'control_motion.npz',**z);(p/'control_summary.json').write_text(json.dumps(rep,indent=2));(p/'joint_names.json').write_text(json.dumps(prepared[3]));(p/'body_names.json').write_text(json.dumps([prepared[1].body(i).name for i in range(prepared[1].nbody)]));mujoco.mj_saveModel(prepared[1],str(p/'control_model.mjb'))
 def f(x):
  p=base.copy();p.update(dict(zip(keys,map(float,x))));rep,z=run(p,dt=.001,prepared=prepared);v=rep.get('takeoff_velocity_m_s',[0,0,0]);vx=-v[1];vz=v[2];H=rep.get('takeoff_H_kg_m2_s',[100])[0];C=rep.get('takeoff_COM_m',[0,0,0]);native=z['physics_force_world_N'];cost=((vx-2.9)/1.)**2+((vz-2.0)/.8)**2+.003*H**2+5*max(.80-C[2],0)**2
  if rep['takeoff_s'] is None or rep['fell']:cost+=30
  # Keep explicit diagnostics; these are not hard human force limits.
  row=dict(trial=len(rows),cost=cost,vx=vx,vz=vz,speed=math.hypot(vx,vz),angle_deg=math.degrees(math.atan2(vz,max(vx,.001))),Hx=H,COM=C,peak_propulsion_BW=float(native[:,2].max()/(87*9.80665)),status=rep['status'],parameters={k:p[k]for k in keys});rows.append(row)
  if cost<=min(r['cost']for r in rows):save(rep,z)
  (out/'search.json').write_text(json.dumps(rows,indent=2));print('PROPULSION',len(rows),round(cost,3),'v',round(vx,2),round(vz,2),'H',round(H,1),'F',round(row['peak_propulsion_BW'],1),flush=True);return cost
 seeds=np.array([[base[k]for k in keys],[.45,.9,.05,.27,.5,-.08,850,400],[.55,1.05,.1,.26,.6,-.1,1100,500],[.4,.8,-.1,.24,.5,-.1,1000,450]],float)
 rng=np.random.default_rng(107);population=np.vstack([seeds,np.array([rng.uniform(lo,hi,20)for lo,hi in bounds]).T]);result=differential_evolution(f,bounds,init=population,maxiter=a.iterations,seed=107,polish=False,tol=.01,mutation=(.4,.9),recombination=.7);best=min(rows,key=lambda r:r['cost']);(out/'development_summary.json').write_text(json.dumps(dict(status='launch_seed_search_completed',best=best,physical_trials=len(rows),targets_are_not_applied_as_state=True,goal='Develop native launch near literature-scale speed, then separately solve flight and landing.',human_maximum_claim=False,reference='Wakai & Linthorne 2005, Table 2: observed preferred-angle speed 3.24-3.55 m/s in five active males; not a population norm.'),indent=2));print('PROPULSION_SEARCH_DONE',best,flush=True)
if __name__=='__main__':main()
