"""Spatial muscle path force assembly without exercise/configuration imports."""
import numpy as np
import mujoco


def tendon_body_wrenches(body,forces):
    m,d=body.m,body.d;wrenches=np.zeros((m.nbody,6));points=d.wrap_xpos.reshape(-1,3);objects=d.wrap_obj.ravel()
    def add(bid,point,force):
        wrenches[bid,:3]+=force;wrenches[bid,3:]+=np.cross(point-d.xipos[bid],force)
    for actuator,F in enumerate(forces):
        if F<1e-10:continue
        tid=int(m.actuator_trnid[actuator,0]);p=int(d.ten_wrapadr[tid]);end=p+int(d.ten_wrapnum[tid]);previous=None
        for j in range(int(m.tendon_adr[tid]),int(m.tendon_adr[tid]+m.tendon_num[tid])):
            kind=int(m.wrap_type[j]);obj=int(m.wrap_objid[j])
            if kind==int(mujoco.mjtWrap.mjWRAP_SITE):count=1;bid=int(m.site_bodyid[obj])
            elif kind in [int(mujoco.mjtWrap.mjWRAP_SPHERE),int(mujoco.mjtWrap.mjWRAP_CYLINDER)]:
                count=2 if p<end and objects[p]==obj else 0;bid=int(m.geom_bodyid[obj])
            else:raise ValueError('Unsupported tendon path type')
            for _ in range(count):
                point=points[p]
                if previous is not None and previous[0]!=bid:
                    other,prior=previous;delta=point-prior;length=np.linalg.norm(delta)
                    if length>1e-12:
                        pull=F*m.actuator_gear[actuator,0]*delta/length;add(other,prior,pull);add(bid,point,-pull)
                previous=(bid,point);p+=1
        if p!=end:raise RuntimeError('Path wrap indexing mismatch')
    return wrenches


def subtree(model,root):
    found=[]
    for bid in range(1,model.nbody):
        current=bid
        while current:
            if current==root:found.append(bid);break
            current=int(model.body_parentid[current])
    return found
