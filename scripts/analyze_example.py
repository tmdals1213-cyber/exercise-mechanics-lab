"""Re-run the small registered example without the historical rollout corpus."""
from pathlib import Path
import argparse,gzip,json,hashlib
import numpy as np,mujoco
from jump_monitor_data import ROOT,read,atomic
from controlled_broad_jump import run

def load_example(folder,dt):
    folder=Path(folder);meta=read(folder/'prepared_model.json');archive=folder/'registered_model.mjb.gz';content=gzip.decompress(archive.read_bytes());assert hashlib.sha256(content).hexdigest()==meta['model_sha256']
    # A VFS name avoids native Windows filename encoding issues for Korean paths.
    m=mujoco.MjModel.from_binary_path('registered_model.mjb',assets={'registered_model.mjb':content});m.opt.timestep=dt;d=mujoco.MjData(m);names=meta['joint_names'];qa=np.array([m.jnt_qposadr[m.joint(n).id]for n in names]);da=np.array([m.jnt_dofadr[m.joint(n).id]for n in names]);points=[(p['body'],np.array(p['local']),np.array(p['initial_world']),p['forefoot'])for p in meta['contact_points']];return(None,m,d,names,qa,da,points,meta['registration'])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--example',default=str(ROOT/'examples/long_jump'));ap.add_argument('--parameters');ap.add_argument('--dt',type=float,default=.0005);ap.add_argument('--set',action='append',default=[],metavar='KEY=JSON_VALUE');ap.add_argument('--output',default=str(ROOT/'results/example_analysis'));args=ap.parse_args();folder=Path(args.example);p=read(Path(args.parameters)if args.parameters else folder/'parameters.json');p=p.get('parameters',p)
    for value in args.set:
        key,raw=value.split('=',1)
        if key not in p:raise ValueError('Unknown parameter: '+key)
        p[key]=json.loads(raw)
    output=Path(args.output);prepared=load_example(folder,args.dt);report,z=run(p,output,dt=args.dt,prepared=prepared);from landing_learning_objective import score
    scored=score(report,z,prepared[1]);atomic(output/'analysis_score.json',scored);print(json.dumps(dict(status=report['status'],stable=scored['stable'],retained_m=scored['retained_distance_m'],first_contact_m=scored['distance_m'],peak_GRF_BW=report['peak_GRF_BW'],output=str(output)),indent=2))

if __name__=='__main__':main()
