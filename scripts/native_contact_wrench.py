"""Native contact resultants; isolated from model-building dependencies."""
import numpy as np,mujoco

def actual_contact_wrench(model,data):
    count=data.ncon
    if not count:return np.zeros(3),np.zeros(3),np.full(3,np.nan)
    contacts=data.contact;g1=contacts.geom1[:count];g2=contacts.geom2[:count];b1=model.geom_bodyid[g1];b2=model.geom_bodyid[g2];ids=np.flatnonzero((b1==0)!=(b2==0))
    if not len(ids):return np.zeros(3),np.zeros(3),np.full(3,np.nan)
    local=np.zeros((len(ids),6))
    for j,ci in enumerate(ids):mujoco.mj_contactForce(model,data,int(ci),local[j])
    frames=contacts.frame[:count].reshape(-1,3,3)[ids];positions=contacts.pos[:count][ids];sign=np.where(b1[ids]==0,1.,-1.)
    forces=np.einsum('nji,nj->ni',frames,local[:,:3])*sign[:,None];couples=np.einsum('nji,nj->ni',frames,local[:,3:])*sign[:,None];force=forces.sum(0);moment=(np.cross(positions,forces)+couples).sum(0);pressure=(positions*np.maximum(forces[:,2],0)[:,None]).sum(0)
    return force,moment,pressure/force[2] if force[2]>1. else np.full(3,np.nan)
