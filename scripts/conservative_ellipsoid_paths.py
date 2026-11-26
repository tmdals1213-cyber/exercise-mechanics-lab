"""Conservative derivative of the registered OpenSim ellipsoid path length.

OpenSim midpoint wrapping is approximate. Its endpoint tangent-force operator
failed the adapter's finite-difference check. Here a single scalar length
function defines ALL body forces and moments, so length and virtual work use
the same approximation. This is not a claim of exact ellipsoid geodesics.
"""
import warnings
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from shoulder_path_bridge import ShoulderPaths,force_basis


class ConservativeEllipsoidPaths(ShoulderPaths):
    def relative_transforms(self):
        d=self.rig.d;chest=self.rig.m.body('chest_r').id;C=d.xmat[chest].reshape(3,3);p=d.xpos[chest]
        return {bid:(C.T@(d.xpos[bid]-p),C.T@d.xmat[bid].reshape(3,3)) for bid in sorted(set(self.parts.values()))},C,chest

    def update(self):
        base,_,_=self.relative_transforms();return self.set_transforms(base)

    def set_transforms(self,transforms):
        for alias,bid in self.parts.items():
            p,R=transforms[bid]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');v=np.r_[Rotation.from_matrix(R).as_euler('XYZ'),p]
            for suffix,x in zip(['rot_x','rot_y','rot_z','tx','ty','tz'],v):self.coords.get(alias+'_free_'+suffix).setValue(self.state,float(x),False)
        self.model.realizePosition(self.state)
        return np.array([mus.getLength(self.state) for mus in self.muscles])

    def operators(self):
        r=self.rig;m,d=r.m,r.d;base,C,chest=self.relative_transforms();ids=[bid for bid in base if bid!=chest];nm=len(self.names)
        L=self.set_transforms(base);W=np.zeros((m.nbody,6,nm));R=np.zeros((m.nv,nm));eps=1e-5
        for bid in ids:
            grad=np.zeros((6,nm))
            for axis in range(6):
                lengths=[]
                for sign in [-1,1]:
                    transforms=base.copy();p,rot=base[bid];p=p.copy();rot=rot.copy()
                    if axis<3:p[axis]+=sign*eps
                    else:
                        v=np.zeros(3);v[axis-3]=sign*eps;rot=Rotation.from_rotvec(v).as_matrix()@rot
                    transforms[bid]=(p,rot);lengths.append(self.set_transforms(transforms))
                grad[axis]=-(lengths[1]-lengths[0])/(2*eps)
            F=C@grad[:3];T=C@grad[3:];W[bid,:3]=F;W[bid,3:]=T+np.cross(d.xpos[bid],F.T).T
        # Relative-coordinate differentiation enforces exact internal force and
        # moment closure, without subtracting an external residual/root force.
        W[chest]=-W.sum(0)
        for bid in [chest,*ids]:
            F=W[bid,:3];T=W[bid,3:]-np.cross(d.xpos[bid],F.T).T
            J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,d.xpos[bid],bid);R+=J.T@F+Jr.T@T
        self.set_transforms(base);self.unit_wrenches=W
        return R,W[:,:3],L,[]


if __name__=='__main__':
    from rigid_baseline_profile import RigidBaselineRig
    r=RigidBaselineRig();p=ConservativeEllipsoidPaths(r);R,A,L,_=p.operators();print('CONSERVATIVE_ELLIPSOID',abs(A.sum(0)).max(),abs(p.unit_wrenches[:,3:].sum(0)).max())
