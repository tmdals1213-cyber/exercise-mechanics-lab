"""Export one compact, reproducible mechanical input, not an outcome corpus."""
import gzip,json,hashlib,argparse
from pathlib import Path
import numpy as np,mujoco
from jump_monitor_data import ROOT,read,atomic
from controlled_broad_jump import build

def export(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);validation=read(ROOT/'results/controlled_broad_jump/landing_stability_v3/verified_best.json');row=min(validation['cases'],key=lambda r:r['retained_distance_m']);source=Path(row['folder']);prepared=build(row['dt_s'],True);_,base,_,names,qa,da,points,registration=prepared;model=source/'control_model.mjb';native=mujoco.MjModel.from_binary_path(str(model));assert np.array_equal(native.body_mass,base.body_mass)and native.nq==base.nq;content=model.read_bytes();(destination/'registered_model.mjb.gz').write_bytes(gzip.compress(content,compresslevel=6,mtime=0));atomic(destination/'parameters.json',row['parameters']);atomic(destination/'prepared_model.json',dict(model_sha256=hashlib.sha256(content).hexdigest(),compressed_sha256=hashlib.sha256((destination/'registered_model.mjb.gz').read_bytes()).hexdigest(),mujoco_version=mujoco.__version__,joint_names=names,contact_points=[dict(body=int(b),local=l.tolist(),initial_world=p.tolist(),forefoot=bool(f))for b,l,p,f in points],registration=registration,scope='Registered reduced rigid body and contacts; no anatomical surface mesh or muscle-path force sharing is represented by its shape.',source_id=row['id'],source_dt_s=row['dt_s'],expected_retained_m=row['retained_distance_m'],checked_min_retained_m=min(r['retained_distance_m']for r in validation['cases']),checked_timesteps_s=sorted(r['dt_s']for r in validation['cases']),input_is_not_a_prescribed_COM_path=True));print('REGISTERED_EXAMPLE_EXPORTED',destination,(destination/'registered_model.mjb.gz').stat().st_size,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--destination',required=True);export(ap.parse_args().destination)
