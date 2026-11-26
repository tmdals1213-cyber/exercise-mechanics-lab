"""Conservative anatomical capsule/box screen, separate from force solving."""
import numpy as np


def segments(model,data):
    points={}
    for side in ['r','l']:
        for name,joint in [('hip','hip_flexion'),('knee','knee_angle'),('ankle','ankle_angle'),('shoulder','shoulder_elv'),('elbow','elbow_flexion'),('wrist','flexion')]:
            points[name+'_'+side]=data.xanchor[model.joint(joint+'_'+side).id].copy()
    rows=[];radii=[]
    for side in ['r','l']:
        for a,b,r in [('hip','knee',.07),('knee','ankle',.05),('shoulder','elbow',.045),('elbow','wrist',.035)]:rows.append([points[a+'_'+side],points[b+'_'+side]]);radii.append(r)
    hips=(points['hip_r']+points['hip_l'])/2;shoulders=(points['shoulder_r']+points['shoulder_l'])/2
    rows.append([hips,shoulders]);radii.append(.12)
    head=data.xipos[model.body('head').id].copy();rows.append([head,head]);radii.append(.10)
    return np.array(rows),np.array(radii)


def segment_box_distance(segment,lower,upper,return_t=False):
    p=segment[0];v=segment[1]-p;cuts=[0.,1.];best_t=0.
    for axis in range(3):
        if abs(v[axis])>1e-12:
            for plane in [lower[axis],upper[axis]]:
                t=(plane-p[axis])/v[axis]
                if 0<t<1:cuts.append(float(t))
    cuts=sorted(set(cuts));best=np.inf
    for t in cuts:
        x=p+t*v;delta=np.maximum(lower-x,0)+np.minimum(upper-x,0);value=float(delta@delta)
        if value<best:best=value;best_t=t
    for a,b in zip(cuts[:-1],cuts[1:]):
        x=p+((a+b)/2)*v;active=(x<lower)|(x>upper);edge=np.where(x<lower,lower,upper)
        if not active.any():return (0.,(a+b)/2) if return_t else 0.
        vv=v[active];base=p[active]-edge[active];den=vv@vv
        if den>1e-16:
            t=np.clip(-float(vv@base)/den,a,b);delta=base+t*vv;value=float(delta@delta)
            if value<best:best=value;best_t=t
    distance=float(np.sqrt(max(0.,best)));return (distance,float(best_t)) if return_t else distance


def minimum_clearance(capsules,radii,front,height,depth=.8):
    lower=np.array([-.525,front-depth,0.]);upper=np.array([.475,front,height]);best=np.inf
    for segment,radius in zip(capsules,radii):
        # AABB separation is a safe early-out for nonintersecting capsules.
        gap=np.maximum(lower-segment.max(0),0)+np.maximum(segment.min(0)-upper,0)
        if np.linalg.norm(gap)>radius+.15:continue
        best=min(best,segment_box_distance(segment,lower,upper)-radius)
    return best
