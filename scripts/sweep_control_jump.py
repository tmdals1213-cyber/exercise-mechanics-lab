from pathlib import Path
import argparse,json,numpy as np
from controlled_broad_jump import ROOT,DEFAULT,build,run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='sweep_v1');ap.add_argument('--count',type=int,default=32);ap.add_argument('--seed',type=int,default=12);ap.add_argument('--base');a=ap.parse_args();out=ROOT/'results/controlled_broad_jump'/a.label;out.mkdir(parents=True,exist_ok=True);prepared=build(.001);rng=np.random.default_rng(a.seed);rows=[]
 for k in range(a.count):
  p=DEFAULT.copy()
  if a.base:p.update(json.loads(Path(a.base).read_text())['parameters']);p.update(controller='pd',seconds=4.5)
  if k:
   p.update(hip_bottom=float(rng.uniform(1.50,1.95)),ankle_bottom=float(rng.uniform(.55,.9)),trunk_bottom=float(rng.uniform(-.3,-.08)),lean_m=float(rng.uniform(.15,.35)),pitch_bottom=float(rng.uniform(.6,1.0)),pitch_top=float(rng.uniform(.3,.9)),push_s=float(rng.uniform(.22,.38)),ankle_delay=float(rng.uniform(.15,.65)),balance_gain=float(rng.uniform(.5,1.5)),pitch_gain=float(rng.uniform(0,300)),shoulder_top=float(rng.uniform(.9,1.8)),hip_top=float(rng.uniform(-.05,.45)))
  report,z=run(p,dt=.001,prepared=prepared)
  v=report.get('takeoff_velocity_m_s',[0,0,0]);H=report.get('takeoff_H_kg_m2_s',[100,0,0]);distance=report.get('first_contact_heel_distance_m',-1)
  score=distance-abs(H[0])*.015-(.8 if report['fell'] else 0.)+(1.5 if report['stable_landing'] else 0.)-.10*min(np.linalg.norm(report['final_COM_velocity_m_s']),5)
  report.update(trial=k,search_score=score);rows.append(report);(out/'sweep.json').write_text(json.dumps(rows,indent=2))
  if k==0 or score>=max(x['search_score']for x in rows):
   best=out/'best';best.mkdir(exist_ok=True);np.savez_compressed(best/'control_motion.npz',**z);(best/'control_summary.json').write_text(json.dumps(report,indent=2));(best/'joint_names.json').write_text(json.dumps(prepared[3]));(best/'body_names.json').write_text(json.dumps([prepared[1].body(i).name for i in range(prepared[1].nbody)]));import mujoco;mujoco.mj_saveModel(prepared[1],str(best/'control_model.mjb'))
  print('TRIAL',k,report['status'],'v',np.round(v,2),'H',round(H[0],1),'distance',round(distance,2),'score',round(score,2),flush=True)
 print('BEST',max(rows,key=lambda r:r['search_score'])['trial'],flush=True)
if __name__=='__main__':main()
