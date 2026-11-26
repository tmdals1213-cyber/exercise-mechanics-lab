from pathlib import Path
import argparse,json,numpy as np,mujoco
from controlled_broad_jump import ROOT,DEFAULT,build,run
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='landing_v1');ap.add_argument('--base',required=True);ap.add_argument('--count',type=int,default=32);a=ap.parse_args();out=ROOT/'results/controlled_broad_jump'/a.label;out.mkdir(parents=True,exist_ok=True);prepared=build(.001);rng=np.random.default_rng(82);rows=[]
 base=json.loads(Path(a.base).read_text())['parameters']
 for k in range(a.count):
  p=DEFAULT.copy();p.update(base);p.update(momentum_feedback=True,controller='pd',seconds=4.5,prepare_hip=1.8,prepare_knee=.8,prepare_landing_s=.22,flight_hip=1.1,flight_knee=1.35,landing_hip=1.3,landing_knee=1.3)
  if k:p.update(prepare_hip=float(rng.uniform(.8,2.1)),prepare_knee=float(rng.uniform(.2,1.3)),prepare_landing_s=float(rng.uniform(.12,.38)),flight_hip=float(rng.uniform(.5,1.7)),flight_knee=float(rng.uniform(.8,1.8)),landing_hip=float(rng.uniform(1.1,1.8)),landing_knee=float(rng.uniform(1.1,1.8)))
  rep,z=run(p,dt=.001,prepared=prepared);distance=rep.get('first_contact_heel_distance_m',-1);speed=np.linalg.norm(rep['final_COM_velocity_m_s']);score=distance-(2 if rep['fell'] else 0)+(3 if rep['stable_landing'] else 0)-.1*min(speed,5);rep.update(trial=k,search_score=score);rows.append(rep);(out/'sweep.json').write_text(json.dumps(rows,indent=2))
  if k==0 or score>=max(x['search_score']for x in rows):
   best=out/'best';best.mkdir(exist_ok=True);np.savez_compressed(best/'control_motion.npz',**z);(best/'control_summary.json').write_text(json.dumps(rep,indent=2));(best/'joint_names.json').write_text(json.dumps(prepared[3]));(best/'body_names.json').write_text(json.dumps([prepared[1].body(i).name for i in range(prepared[1].nbody)]));mujoco.mj_saveModel(prepared[1],str(best/'control_model.mjb'))
  print('LANDING',k,rep['status'],'distance',round(distance,2),'endV',round(speed,2),'score',round(score,2),flush=True)
if __name__=='__main__':main()
