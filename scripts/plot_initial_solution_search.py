from pathlib import Path
import argparse,json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='two_metre_initial_solution_v18');a=ap.parse_args();source=ROOT/'results/controlled_broad_jump'/a.label;rows=[json.loads(s)for s in(source/'trials.jsonl').read_text().splitlines()];definition=json.loads((source/'search_definition.json').read_text());population=definition.get('population',32);dimensions=definition.get('all_parameters_jointly_varied',43);out=ROOT/'deliverables/standing_long_jump_learning';out.mkdir(exist_ok=True)
 for row in rows:
  row['stable']=row.get('robust_stable',row.get('stable',False))
  if row['stable']and(row.get('refinement')or{}).get('valid_distance_m')is not None:row['valid_distance_m']=min(row['valid_distance_m'],row['refinement']['valid_distance_m'])
 valid=[r for r in rows if r.get('stable')];invalid=[r for r in rows if not r.get('stable')and r.get('first_heel_m')is not None];fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[3,1]});fig.set_facecolor('#0d1726')
 for ax in axes:ax.set_facecolor('#0d1726');ax.tick_params(colors='#a6bcd1');ax.xaxis.label.set_color('#a6bcd1');ax.yaxis.label.set_color('#a6bcd1');ax.grid(alpha=.13);ax.spines[['top','right']].set_visible(False)
 axes[0].scatter([r['trial']+1 for r in invalid],[r['first_heel_m']for r in invalid],marker='x',s=12,c='#735957',alpha=.45,label='First contact, failed landing');axes[0].scatter([r['trial']+1 for r in valid],[r['valid_distance_m']for r in valid],s=26,c='#60b7dd',label='Stable supported landing');best=-np.inf;path=[]
 for r in rows:
  if r.get('stable'):best=max(best,r['valid_distance_m'])
  path.append(best if np.isfinite(best)else np.nan)
 axes[0].plot(np.arange(1,len(rows)+1),path,color='#ef9558',lw=2,label='Best valid');axes[0].axhline(2.,ls='--',color='#7cd2a5',lw=1,label='2 m initialization target');axes[0].set_ylim(.8,2.65);axes[0].set_ylabel('Foot contact distance (m)');legend=axes[0].legend(loc='lower right',facecolor='#172739',edgecolor='none',labelcolor='#edf3fb');gens=(len(rows)+population-1)//population;rates=[np.mean([r.get('stable',False)for r in rows[g*population:(g+1)*population]])for g in range(gens)];axes[1].bar(np.arange(gens)*population+population/2,np.array(rates)*100,width=population*.75,color='#60b7dd');axes[1].set(xlabel='Native physics trial',ylabel='Stable (%)',ylim=(0,100));fig.suptitle(f'Initial-motion search / {dimensions} jointly varied posture and timing parameters',color='#edf3fb',fontsize=14);fig.tight_layout();fig.savefig(out/'initial_solution_search.png',dpi=180);plt.close(fig);print('INITIAL_SEARCH_PLOT',len(rows),best,flush=True)
if __name__=='__main__':main()
