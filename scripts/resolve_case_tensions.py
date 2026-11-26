"""Reuse physical path operators; solve a time-consistent whole-history QP."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from tension_reference import solve_history,reaction_cone

ROOT=Path(__file__).resolve().parents[1]

def resolve(folder,z,summary,label):
    ts=z['times'];names=summary['coordinate_names'];keep=[i for i,n in enumerate(names) if not n.startswith('mtp_angle_')]
    stability=[reaction_cone(S,o,a,20.,16) for S,o,a in zip(z['S'],z['offset'],z['axes'])]
    print('RESOLVE_BATCH',folder.name,label,len(ts),flush=True)
    s=solve_history(ts,z['R'][:,keep],z['target'][:,keep],np.ones_like(z['passive']),z['passive'],z['reference_N'],stability,temporal_weight=.002,level_weight=1.)
    F=s['force_N'];s['unidentified_joint_support_moment_Nm']=z['target']-np.einsum('tqn,tn->tq',z['R'],F)
    s['GH_reaction_N']=z['offset']-np.einsum('tscn,tn->tsc',z['S'],F)
    z.update(s);out=folder/('batch_30hz.npz' if label=='30Hz' else 'tensions.npz');np.savez_compressed(out,**z)
    result=summary.copy();result.update(selection_mode='batch',sample_count=len(ts),checks=s['checks'],
        force_sharing='Whole-history integral of normalized force magnitude and derivative squared. Not physiological effort or metabolic energy.',
        force_file_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    result['peak_element_N']=float(F.max());result['peak_element']=result['names'][int(np.unravel_index(F.argmax(),F.shape)[1])]
    setup=json.loads((folder/'setup.json').read_text());result['scope']=setup['scope'];result['movement_kind']=setup.get('movement_kind','jump')
    result['unidentified_support_peak_Nm']={n:float(abs(s['unidentified_joint_support_moment_Nm'][:,i]).max()) for i,n in enumerate(names) if n.startswith('mtp_angle_')}
    (folder/('batch_summary_30hz.json' if label=='30Hz' else 'tension_summary.json')).write_text(json.dumps(result,indent=2))
    print('RESOLVE_PASSED',label,result['checks']['moment_balance_max_error'],flush=True)

def main(relative):
    folder=ROOT/relative;coarse=dict(np.load(folder/'tensions_30hz.npz'));fine=dict(np.load(folder/'operators_60hz.npz'));summary=json.loads((folder/'operator_summary.json').read_text())
    if coarse['times'][-1]<fine['times'][-1]-1e-9:
        n=len(coarse['times'])
        for k,v in list(coarse.items()):
            if k!='reference_N' and v.ndim and len(v)==n:coarse[k]=np.concatenate([v,fine[k][-1:]])
    resolve(folder,coarse,summary,'30Hz');resolve(folder,fine,summary,'60Hz')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);main(p.parse_args().folder)
