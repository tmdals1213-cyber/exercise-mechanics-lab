from pathlib import Path
import json,numpy as np,mujoco
from scipy.optimize import differential_evolution
from controlled_broad_jump import ROOT,DEFAULT,build,run
def main():
 source=ROOT/'results/controlled_broad_jump/standing_long_jump_seed_v1/candidate_05';out=ROOT/'results/controlled_broad_jump/broad_jump_seed_flight_v2';out.mkdir(parents=True,exist_ok=True);base=DEFAULT.copy();base.update(json.loads((source/'control_summary.json').read_text())['parameters']);base.update(seconds=4.8,flight_transition_s=.18,prepare_transition_s=.14);prepared=build(.001);cache={};rows=[]
 keys=['flight_hip','flight_knee','flight_shoulder','prepare_hip','prepare_knee','prepare_shoulder','prepare_trunk','prepare_landing_s'];bounds=[(.4,1.7),(.5,1.6),(-.3,1.7),(.6,2.0),(.1,.9),(-.4,1.7),(-.5,.4),(.04,.20)]
 def f(x):
  p=base.copy();p.update(dict(zip(keys,map(float,x))));rep,z=run(p,dt=.001,prepared=prepared,flight_cache=cache);dist=rep.get('first_contact_heel_distance_m',-1);capture=rep.get('landing_capture_error_m',2);pitch=abs(rep.get('landing_torso_pitch_rad',2));speed=np.linalg.norm(rep['final_COM_velocity_m_s']);cost=4*capture+1.5*pitch+(2 if rep['fell']else 0)+.15*min(speed,6)-(5 if rep['stable_landing']else 0)-.1*dist;rep.update(search_cost=cost,trial=len(rows));rows.append(rep);(out/'search.json').write_text(json.dumps(rows,indent=2))
  if len(rows)==1 or cost<=min(r['search_cost']for r in rows):
   pth=out/'best';pth.mkdir(exist_ok=True);np.savez_compressed(pth/'control_motion.npz',**z);(pth/'control_summary.json').write_text(json.dumps(rep,indent=2));(pth/'joint_names.json').write_text(json.dumps(prepared[3]));(pth/'body_names.json').write_text(json.dumps([prepared[1].body(i).name for i in range(prepared[1].nbody)]));mujoco.mj_saveModel(prepared[1],str(pth/'control_model.mjb'))
  print('BROAD_FLIGHT',len(rows)-1,rep['status'],'d',round(dist,2),'capture',round(capture,2),'tilt',round(pitch,2),'cost',round(cost,2),flush=True);return cost
 def stop(x,convergence):return any(r['stable_landing']for r in rows)
 differential_evolution(f,bounds,popsize=4,maxiter=3,seed=55,polish=False,callback=stop,tol=.01);print('BROAD_FLIGHT_READY',out,flush=True)
if __name__=='__main__':main()
