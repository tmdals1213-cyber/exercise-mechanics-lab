"""Independent SI rigid-body wrench assembly; no engine dynamics calls.

Wrench order is [Fx,Fy,Fz,Mx,My,Mz], world axes, moment about world origin.
Inputs are physical segment COM states and inertia tensors in world axes.
"""
import numpy as np

def body_wrenches(mass,com,inertia_world,acceleration,omega,alpha,gravity):
    mass=np.asarray(mass);com=np.asarray(com);omega=np.asarray(omega)
    f=mass[:,None]*(np.asarray(acceleration)-np.asarray(gravity))
    angular=np.einsum('bij,bj->bi',inertia_world,alpha)+np.cross(omega,np.einsum('bij,bj->bi',inertia_world,omega))
    return np.c_[f,angular+np.cross(com,f)]

def external_wrenches(body_count,contacts):
    w=np.zeros((body_count,6))
    for body,point,force,moment in contacts:
        w[body,:3]+=force;w[body,3:]+=np.cross(point,force)+moment
    return w

def accumulate(parent,local_wrench):
    result=np.array(local_wrench,dtype=float,copy=True)
    for i in range(len(parent)-1,0,-1):result[int(parent[i])]+=result[i]
    return result

def at_point(wrench,point):
    w=np.array(wrench,copy=True);w[3:]-=np.cross(point,w[:3]);return w

def energy(mass,com,inertia_world,velocity,omega,gravity):
    return float(.5*np.sum(np.asarray(mass)[:,None]*np.asarray(velocity)**2)+.5*np.einsum('bi,bij,bj->',omega,inertia_world,omega)-np.sum(np.asarray(mass)[:,None]*np.asarray(com)*np.asarray(gravity)))
