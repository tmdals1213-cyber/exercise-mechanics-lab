import argparse,json,numpy as np,mujoco
from scipy.optimize import differential_evolution
from controlled_broad_jump import ROOT,DEFAULT,build,run
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='takeoff_physical_v1');ap.add_argument('--iterations',type=int,default=2);a=ap.parse_args();out=ROOT/'results/controlled_broad_jump'/a.label;out.mkdir(parents=True,exist_ok=True);prepared=build(.001);rows=[];base=DEFAULT.copy();base.update(seconds=1.85)
 keys=['hip_bottom','ankle_bottom','push_s','ankle_delay','ankle_top','shoulder_top','pitch_top','lean_m'];bounds=[(1.4,1.9),(.4,.85),(.25,.42),(.3,.8),(-.65,-.1),(.6,1.7),(.2,.7),(.12,.32)]
 def f(x):
  p=base.copy();p.update(dict(zip(keys,map(float,x))));rep,z=run(p,dt=.001,prepared=prepared);v=rep.get('takeoff_velocity_m_s',[0,0,0]);H=rep.get('takeoff_H_kg_m2_s',[100,0,0])[0];pitch=rep.get('takeoff_root_pitch_rad',2);vx=-v[1];vz=v[2];cost=.12*abs(H)+4*max(.7-vx,0)+max(vx-2.3,0)+4*max(2.0-vz,0)+2*max(vz-3.5,0)+3*max(abs(pitch-.4)-.25,0)+(4 if rep['takeoff_s']is None else 0);rep.update(search_cost=cost,trial=len(rows));rows.append(rep);(out/'search.json').write_text(json.dumps(rows,indent=2))
  if len(rows)==1 or cost<=min(r['search_cost']for r in rows):
   best=out/'best';best.mkdir(exist_ok=True);np.savez_compressed(best/'control_motion.npz',**z);(best/'control_summary.json').write_text(json.dumps(rep,indent=2));(best/'joint_names.json').write_text(json.dumps(prepared[3]));(best/'body_names.json').write_text(json.dumps([prepared[1].body(i).name for i in range(prepared[1].nbody)]));mujoco.mj_saveModel(prepared[1],str(best/'control_model.mjb'))
  print('TAKEOFF',len(rows)-1,'v',np.round(v,2),'H',round(H,1),'pitch',round(pitch,2),'cost',round(cost,2),flush=True);return cost
 differential_evolution(f,bounds,popsize=4,maxiter=a.iterations,seed=63,polish=False,tol=.01,updating='immediate');print('TAKEOFF_SEARCH_READY',out,flush=True)
if __name__=='__main__':main()
