"""Optional source-spline patella kinematics, replacing quartic extrapolation.

The exact OpenSim SimmSpline functions and their derivatives define position,
velocity reduction and curvature together. No force values are fitted/capped.
Old result files keep their original profile; callers explicitly opt in.
"""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np,opensim as osim
ROOT=Path(__file__).resolve().parents[1]


def apply(r):
    source=ROOT/'datasets/cmj_calisti2025/source/catelli_3dof_model.osim';doc=ET.parse(source);g=r.b.report['global_normalization_scale'];functions=[]
    for side in ['r','l']:
        joint=doc.find('.//CustomJoint[@name="patellofemoral_'+side+'"]')
        for axis,stem,scale in [('rotation1','rotation1',1.),('translation1','translation1',g),('translation2','translation2',g)]:
            node=joint.find('./SpatialTransform/TransformAxis[@name="'+axis+'"]/SimmSpline');x=np.fromstring(node.findtext('x'),sep=' ');y=np.fromstring(node.findtext('y'),sep=' ')*scale;f=osim.SimmSpline()
            for xx,yy in zip(x,y):f.addPoint(float(xx),float(yy))
            dep=r.b.joints['knee_angle_beta_'+stem+'_'+side];ind=r.b.joints['knee_angle_'+side];functions.append((int(r.m.jnt_qposadr[dep]),int(r.m.jnt_dofadr[dep]),int(r.m.jnt_qposadr[ind]),int(r.m.jnt_dofadr[ind]),f))
    removed={row[0] for row in functions};r.eqs=[row for row in r.eqs if row[0] not in removed];old_complete=r.original_complete;old_coordinates=r.coordinates
    def correct(q):
        q=old_complete(q)
        for qa,_,ia,_,f in functions:q[qa]=f.calcValue(osim.Vector(1,float(q[ia])))
        return q
    def coordinates(q,u):
        B,c=old_coordinates(q,u)
        for qa,dof,ia,idof,f in functions:
            v=osim.Vector(1,float(q[ia]));first=f.calcDerivative([0],v);second=f.calcDerivative([0,0],v);B[dof]=first*B[idof];speed=float(B[idof]@u);c[dof]=first*c[idof]+second*speed*speed
        return B,c
    r.original_complete=correct;r.coordinates=coordinates
    report=dict(kind='published_patella_spline_kinematics',source_file=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),replaces='Native quartic patellar translations/rotation, including extrapolation above the fitted range.',source_function='OpenSim SimmSpline; exact value, first and second derivatives including its native end extrapolation.',source_model_knee_range_deg=145.,changed_dependent_coordinates=sorted(r.m.joint(r.m.dof_jntid[row[1]]).name for row in functions),changes_strength_or_force_scale=False,requires_inverse_dynamics_recomputation=True,not_individual_anatomy_validation=True)
    r.profile['published_patella']=report;r.b.report['published_patella']=report;r.b.report['implementation_sha256']=hashlib.sha256((r.b.report['implementation_sha256']+json.dumps(report,sort_keys=True)).encode()).hexdigest();return report
