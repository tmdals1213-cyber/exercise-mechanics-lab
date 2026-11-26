"""OpenSim wrapped thoracoscapular paths on the full-body bone transforms.

This is a geometry/virtual-work adapter, not a second dynamics model. It uses
the source ellipsoid wrapping algorithm and exports unit-tension body forces
for the single full-body Newton-Euler system. Bone-local geometry is registered
by similarity and mirrored in the completed neutral anatomy.
"""
import copy, tempfile, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, mujoco, opensim as osim
from scipy.spatial.transform import Rotation
from jump_rig import ROOT

PREFIX=('Trapezius','Serratus','Rhomboideus','Levator','PectoralisMinor')


class ShoulderPaths:
    def __init__(self,rig):
        self.rig=rig;b=rig.b;m,d=rig.m,rig.d;b.pose(b.q0);scale=b.report['global_normalization_scale'];self.scale=scale
        source=ET.parse(ROOT/'models/rmr/TSM_subject_noWeight.osim').getroot();original=source.find('Model');new=ET.Element('OpenSimDocument',Version='40000');model=ET.SubElement(new,'Model',name='wrapped_path_adapter')
        ET.SubElement(model,'gravity').text='0 0 0';bodies=ET.SubElement(ET.SubElement(model,'BodySet',name='bodyset'),'objects');joints=ET.SubElement(ET.SubElement(model,'JointSet',name='jointset'),'objects');forces=ET.SubElement(ET.SubElement(model,'ForceSet',name='forceset'),'objects')
        joint_template=original.find('./JointSet/objects/CustomJoint[@name="ground_thorax"]');body_source={x.attrib['name']:x for x in original.findall('./BodySet/objects/Body')}
        self.parts={};self.maps={};self.names=[];self.refs=[];self.pathdefs=[]
        mirror=np.diag([-1.,1.,1.]);centre=d.xpos[m.body('chest_r').id].copy()
        for side in ['r','l']:
            mapping={}
            for native,src in [('chest_r','thorax'),('clavicle_'+side,'clavicle'),('scapula_'+side,'scapula')]:
                alias=src+'_'+side;bid=m.body(native).id;self.parts[alias]=bid
                if side=='r':T=np.eye(3)
                else:
                    right=m.body('chest_r' if src=='thorax' else src+'_r').id
                    T=d.xmat[bid].reshape(3,3).T@mirror@d.xmat[right].reshape(3,3)
                self.maps[alias]=T;body=copy.deepcopy(body_source[src]);body.attrib['name']=alias
                for tag in ['attached_geometry','components']:
                    for e in body.findall(tag):body.remove(e)
                for wrap in body.findall('./WrapObjectSet/objects/*'):wrap.attrib['name']+='_'+side
                for wrap in body.findall('./WrapObjectSet/objects/*'):
                    pos=np.fromstring(wrap.findtext('translation'),sep=' ')*scale;wrap.find('translation').text=' '.join(map(str,T@pos))
                    dims=wrap.find('dimensions')
                    if dims is not None:dims.text=' '.join(map(str,np.fromstring(dims.text,sep=' ')*scale))
                    rot=wrap.find('xyz_body_rotation')
                    if rot is not None:
                        R=Rotation.from_euler('XYZ',np.fromstring(rot.text,sep=' ')).as_matrix();R=T@R
                        if np.linalg.det(R)<0:R=R@np.diag([1,1,-1])
                        rot.text=' '.join(map(str,Rotation.from_matrix(R).as_euler('XYZ')))
                bodies.append(body);joint=copy.deepcopy(joint_template);text=ET.tostring(joint,encoding='unicode').replace('ground_thorax',alias+'_free').replace('/bodyset/thorax','/bodyset/'+alias).replace('thorax_offset',alias+'_offset')
                joint=ET.fromstring(text);joint.attrib['name']=alias+'_free'
                for c in joint.findall('./coordinates/Coordinate'):c.find('default_value').text='0'
                # Remove source default body/ground offset poses.
                for e in joint.findall('./frames/PhysicalOffsetFrame'):
                    for tag in ['translation','orientation']:
                        if e.find(tag) is not None:e.find(tag).text='0 0 0'
                joints.append(joint);mapping[src]=alias
            for mus in original.findall('./ForceSet/objects/Millard2012EquilibriumMuscle'):
                if not mus.attrib['name'].startswith(PREFIX):continue
                clone=copy.deepcopy(mus);name=mus.attrib['name']+'_'+side;clone.attrib['name']=name
                self.names.append(name);self.refs.append(float(mus.findtext('max_isometric_force')))
                pp=[]
                for point in clone.findall('./GeometryPath/PathPointSet/objects/*'):
                    frame=point.find('socket_parent_frame');src=frame.text.split('/')[-1];alias=mapping[src];frame.text='/bodyset/'+alias;pos=point.find('location');loc=self.maps[alias]@(np.fromstring(pos.text,sep=' ')*scale);pos.text=' '.join(map(str,loc));pp.append((self.parts[alias],loc))
                # Source wrap names are resolved on the points' shared thorax.
                for wrap in clone.findall('./GeometryPath/PathWrapSet/objects/PathWrap'):wrap.find('wrap_object').text='Thorax_'+side
                forces.append(clone);self.pathdefs.append(pp)
        osim.Logger.setLevelString('error');self.temp=Path('D:/Exercise_mechanics/results/model_foundation/shoulder_candidate/wrapped_path_adapter.osim');ET.ElementTree(new).write(self.temp,encoding='utf-8',xml_declaration=True)
        self.model=osim.Model(str(self.temp));self.state=self.model.initSystem();self.muscles=list(self.model.getMuscles());self.coords=self.model.getCoordinateSet();self.refs=np.asarray(self.refs)
        self.max_transform_error=0.

    def update(self):
        d=self.rig.d
        for alias,bid in self.parts.items():
            R=d.xmat[bid].reshape(3,3);v=np.r_[Rotation.from_matrix(R).as_euler('XYZ'),d.xpos[bid]]
            for suffix,x in zip(['rot_x','rot_y','rot_z','tx','ty','tz'],v):self.coords.get(alias+'_free_'+suffix).setValue(self.state,float(x),False)
        self.model.realizePosition(self.state)
        for alias,bid in self.parts.items():
            T=self.model.getBodySet().get(alias).getTransformInGround(self.state);R=T.R();p=T.p();error=max(np.max(abs(np.array([[R.get(i,j) for j in range(3)] for i in range(3)])-d.xmat[bid].reshape(3,3))),np.max(abs(np.array([p.get(i) for i in range(3)])-d.xpos[bid])))
            self.max_transform_error=max(self.max_transform_error,float(error))
        if self.max_transform_error>1e-7:raise RuntimeError(('Geometry registration transform',self.max_transform_error))

    def operators(self):
        self.update();r=self.rig;m,d=r.m,r.d;nm=len(self.muscles);R=np.zeros((m.nv,nm));A=np.zeros((m.nbody,3,nm));length=[];paths=[]
        for j,mus in enumerate(self.muscles):
            gp=mus.getGeometryPath();length.append(gp.getLength(self.state));dirs=osim.ArrayPointForceDirection();gp.getPointForceDirections(self.state,dirs);pts=[]
            for k in range(dirs.getSize()):
                pf=dirs.get(k);frame=pf.frame();alias=frame.getName();bid=self.parts[alias];p=pf.point();v=pf.direction();local=np.array([p.get(i) for i in range(3)]);direction=np.array([v.get(i) for i in range(3)])
                # OpenSim PointForceDirection vectors are expressed in ground;
                # point is in its body frame. Checked by length virtual work.
                J,world=r.b.jacobian(bid,local);R[:,j]+=J.T@direction;A[bid,:,j]+=direction;pts.append(world)
            paths.append(np.asarray(pts))
        return R,A,np.array(length),paths


def force_basis(rig,q,u):
    """Release five anatomical SC/AC rotations per side for force balance.

    The inherited scapulohumeral rhythm prescribes the trajectory only. Ideal
    coupling torques are not allowed to replace thoracoscapular muscle torques.
    Coordinate-helper counter-rotations remain a change of coordinates.
    """
    B,_=rig.coordinates(q,u);B=B.copy();extra=[];names=rig.b.coordinate_names.copy()
    for side in ['r','l']:
        shoulder=6+rig.b.ind_names.index('shoulder_elv_'+side)
        defs=[('sternoclavicular_r2','unrotscap_r2'),('sternoclavicular_r3','unrotscap_r3'),('acromioclavicular_r1',None),('acromioclavicular_r2',None),('acromioclavicular_r3',None)]
        for name,counter in defs:
            col=np.zeros(rig.m.nv);jid=rig.b.joints[name+'_'+side];adr=rig.m.jnt_dofadr[jid];B[adr,shoulder]=0;col[adr]=1
            if counter:
                cad=rig.m.jnt_dofadr[rig.b.joints[counter+'_'+side]];B[cad,shoulder]=0;col[cad]=-1
            extra.append(col);names.append(name+'_'+side)
    return np.column_stack([B,*extra]),names


if __name__=='__main__':
    from rigid_baseline_profile import RigidBaselineRig
    rig=RigidBaselineRig();paths=ShoulderPaths(rig);rig.b.pose(rig.b.q0);R,A,L,p=paths.operators();print('SHOULDER_PATHS',len(paths.names),'transform error',paths.max_transform_error,'lengths',L,flush=True)
