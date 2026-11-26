from pathlib import Path
import argparse,json
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from retarget_measured_jump import MeasuredJumpRig

p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();folder=Path(a.folder)
z=np.load(folder/'motion.npz');h=np.load(folder/'ground_momentum_targets.npz');s=json.loads((folder/'setup.json').read_text());r=MeasuredJumpRig(s['mass_kg'],s['height_m']);b,m,d=r.b,r.m,r.d
target=CubicSpline(h['times'],h['H']);ids=np.flatnonzero(z['mode']<2);error=z['angular_momentum'][ids]-target(z['times'][ids])
worst=np.argsort(np.linalg.norm(z['root_residual'][ids,3:],axis=1))[-8:]
for k in worst:
 i=ids[k];print('WORST',float(z['times'][i]),'root',z['root_residual'][i,3:],'Herror',error[k],'foot_v',float(z['foot_contact_speed_m_s'][i]))
print('H_ERROR',np.max(abs(error),axis=0),'RM',np.quantile(np.linalg.norm(z['root_residual'][ids,3:],axis=1),[.5,.9,.95,.99,1]))
q=z['qpos'][ids[len(ids)//2]];u=z['u'][ids[len(ids)//2]];B,c=r.coordinates(q,u);v=B@u;eps=1e-5
qp=q.copy();qm=q.copy();mujoco.mj_integratePos(m,qp,v,eps);mujoco.mj_integratePos(m,qm,v,-eps);qp=b.complete(qp);qm=b.complete(qm);fd=np.zeros(m.nv);mujoco.mj_differentiatePos(m,fd,2*eps,qm,qp)
print('MAP_V_ERROR',np.max(abs(fd-v)))
