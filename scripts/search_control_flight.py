from pathlib import Path
import argparse,json,numpy as np,mujoco
from scipy.optimize import differential_evolution
from controlled_broad_jump import ROOT,DEFAULT,build,run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='flight_search_v1');ap.add_argument('--base',required=True);ap.add_argument('--iterations',type=int,default=8);a=ap.parse_args();out=ROOT/'results/controlled_broad_jump'/a.label;out.mkdir(parents=True,exist_ok=True);prepared=build(.001);base=DEFAULT.copy();base.update(json.loads(Path(a.base).read_text())['parameters']);base.update(controller='pd',seconds=4.5,momentum_feedback=True,landing_feedback=True);rows=[]
 keys=['flight_hip','flight_knee','flight_shoulder','prepare_hip','prepare_knee','prepare_shoulder','prepare_trunk','prepare_landing_s'];cache={}
 bounds=[(.3,1.8),(.5,1.9),(-.4,1.5),(.3,2.1),(.1,1.3),(-.4,1.5),(-.5,.4),(.12,.4)]
 x0=np.array([base[k]for k in keys]);bounds=[(max(lo,v-.16),min(hi,v+.16))for (lo,hi),v in zip(bounds,x0)]
 def objective(x):
  p=base.copy();p.update(dict(zip(keys,map(float,x))));rep,z=run(p,dt=.001,prepared=prepared,flight_cache=cache);dist=rep.get('first_contact_heel_distance_m',-1);capture=rep.get('landing_capture_error_m',2);pitch=abs(rep.get('landing_torso_pitch_rad',2));speed=np.linalg.norm(rep['final_COM_velocity_m_s']);score=4*capture+1.5*pitch+(2 if rep['fell'] else 0)+.15*min(speed,6)-(4 if rep['stable_landing'] else 0)-.05*dist;rep.update(trial=len(rows),search_cost=score);rows.append(rep);(out/'search.json').write_text(json.dumps(rows,indent=2))
  if len(rows)==1 or score<=min(r['search_cost']for r in rows):
   best=out/'best';best.mkdir(exist_ok=True);np.savez_compressed(best/'control_motion.npz',**z);(best/'control_summary.json').write_text(json.dumps(rep,indent=2));(best/'joint_names.json').write_text(json.dumps(prepared[3]));(best/'body_names.json').write_text(json.dumps([prepared[1].body(i).name for i in range(prepared[1].nbody)]));mujoco.mj_saveModel(prepared[1],str(best/'control_model.mjb'))
  print('FLIGHT',len(rows)-1,rep['status'],'d',round(dist,2),'capture',round(capture,2),'tilt',round(pitch,2),'cost',round(score,2),flush=True);return score
 differential_evolution(objective,bounds,popsize=4,maxiter=a.iterations,seed=31,polish=False,tol=.01,updating='immediate',x0=np.clip(x0,[x[0]for x in bounds],[x[1]for x in bounds]));print('SEARCH_READY',out,flush=True)
if __name__=='__main__':main()
