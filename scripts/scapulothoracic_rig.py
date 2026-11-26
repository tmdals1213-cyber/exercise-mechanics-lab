"""Generic two-point, frictionless scapulothoracic apposition constraint.

An explicit approximation, not the original Seth joint or measured anatomy.
Two source medial scapular landmarks lie on a thorax-fixed ellipsoid. Its
center is fitted to neutral bone geometry ONLY, never to forces. Five SC/AC
rotations therefore have three admissible freedoms. The animation rhythm
specifies three; the other two are solved from surface closure.
"""
import hashlib,json,xml.etree.ElementTree as ET
from itertools import combinations
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from rigid_baseline_profile import RigidBaselineRig
from jump_rig import ROOT


class ScapulothoracicRig(RigidBaselineRig):
    def __init__(self,subject=None):
        super().__init__(subject=subject);b,m,d=self.b,self.m,self.d;b.pose(b.q0);self.contacts=[];self.surface_fit=[]
        doc=ET.parse(ROOT/'models/rmr/TSM_subject_noWeight.osim');joint=doc.find('.//ScapulothoracicJoint');g=b.report['global_normalization_scale'];radii=np.fromstring(joint.findtext('thoracic_ellipsoid_radii_x_y_z'),sep=' ')*g
        parent=joint.find('./frames/PhysicalOffsetFrame[@name="thorax_offset"]');wanted=np.fromstring(parent.findtext('translation'),sep=' ')*g;rot=Rotation.from_euler('XYZ',np.fromstring(parent.findtext('orientation'),sep=' ')).as_matrix()
        points=[]
        for muscle in ['Rhomboideus_S','SerratusAnterior_I']:
            mus=doc.find('.//Millard2012EquilibriumMuscle[@name="'+muscle+'"]')
            point=next(p for p in mus.findall('./GeometryPath/PathPointSet/objects/*') if p.findtext('socket_parent_frame').endswith('/scapula'));points.append(np.fromstring(point.findtext('location'),sep=' ')*g)
        chest=m.body('chest_r').id;right=m.body('scapula_r').id;Rc=d.xmat[chest].reshape(3,3);Rs=d.xmat[right].reshape(3,3);ps=np.array([Rc.T@(d.xpos[right]+Rs@p-d.xpos[chest]) for p in points])
        def residual(c):return np.sum(((ps-c)@rot/radii)**2,axis=1)-1
        fit=minimize(lambda c:float(np.sum((c-wanted)**2)),wanted,jac=lambda c:2*(c-wanted),constraints=[dict(type='eq',fun=residual)],method='SLSQP',options=dict(ftol=1e-14,maxiter=1000))
        if not fit.success or abs(residual(fit.x)).max()>1e-8:raise RuntimeError(('Neutral ellipsoid registration',fit.message))
        for side in ['r','l']:
            bid=m.body('scapula_'+side).id;T=np.eye(3) if side=='r' else d.xmat[bid].reshape(3,3).T@np.diag([-1,1,1.])@Rs
            Tc=np.eye(3) if side=='r' else Rc.T@np.diag([-1,1,1.])@Rc;Rsurface=Tc@rot
            for point in points:self.contacts.append(dict(side=side,body=bid,chest=chest,local=T@point,centre=Tc@fit.x,rotation=Rsurface,radii=radii))
            self.surface_fit.append(dict(side=side,centre_local_m=(Tc@fit.x).tolist(),radii_m=radii.tolist(),centre_registration_m=float(np.linalg.norm(fit.x-wanted)),source_landmarks=['Rhomboideus_S scapular attachment','SerratusAnterior_I scapular attachment']))
        columns=[];labels=[]
        for side in ['r','l']:
            local=[]
            for name,counter in [('sternoclavicular_r2','unrotscap_r2'),('sternoclavicular_r3','unrotscap_r3'),('acromioclavicular_r1',None),('acromioclavicular_r2',None),('acromioclavicular_r3',None)]:
                col=np.zeros(m.nv);col[m.jnt_dofadr[b.joints[name+'_'+side]]]=1
                if counter:col[m.jnt_dofadr[b.joints[counter+'_'+side]]]=-1
                local.append(col)
            columns.extend(local)
        self.Pdep=np.column_stack(columns);self.dep_dofs=np.where(abs(self.Pdep).sum(1)>0)[0];self.dependent=[];self.cache={};self.last=np.zeros(10)
        self.profile['scapulothoracic_contact']=dict(dependent=labels,geometry=self.surface_fit)
        self.original_complete=b.complete;self.original_pose=b.pose;b.complete=self.complete;b.pose=self.pose
        self.qneutral=b.q0.copy();self.mapping=None;self.make_mapping();b.q0=self.complete(b.q0)
        b.report['implementation_sha256']=hashlib.sha256((b.report['implementation_sha256']+hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()
        self.contact_profile=dict(type='two_point_frictionless_ellipsoid',neutral_fit=self.surface_fit,anatomy_empirically_validated=False,rhythm_specifies='SC protraction/elevation and AC upward rotation; other AC rotations solve surface closure')

    def geometry(self,q):
        m,d=self.m,self.d;d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);gaps=[];G=[];data=[]
        for p in self.contacts:
            bid,cb=p['body'],p['chest'];Rc=d.xmat[cb].reshape(3,3);Rs=d.xmat[bid].reshape(3,3);world=d.xpos[bid]+Rs@p['local'];center=d.xpos[cb]+Rc@p['centre'];R=Rc@p['rotation'];x=R.T@(world-center);grad=R@(2*x/p['radii']**2);norm=np.linalg.norm(grad);normal=grad/norm
            J=np.zeros((3,m.nv));Jr=J.copy();K=J.copy();Kr=J.copy();mujoco.mj_jac(m,d,J,Jr,world,bid);mujoco.mj_jac(m,d,K,Kr,world,cb)
            G.append(grad@(J-K));gaps.append(np.sum((x/p['radii'])**2)-1);data.append((bid,cb,world,normal,(J-K).T@normal))
        return np.array(gaps),np.array(G),data

    def complete(self,q):
        q=self.original_complete(q).copy();key=tuple(round(float(q[self.m.jnt_qposadr[self.b.joints['shoulder_elv_'+s]]]),10) for s in ['r','l'])
        def configuration(delta):
            result=q.copy();mujoco.mj_integratePos(self.m,result,self.Pdep@delta,1.);return result
        if getattr(self,'mapping',None) is not None:
            if min(key)<-.025 or max(key)>1.875:raise RuntimeError(('Scapular map range',key))
            delta=np.r_[self.mapping(key[0])[:5],self.mapping(key[1])[5:]];result=configuration(delta)
            for _ in range(5):
                gap,G,_=self.geometry(result)
                if abs(gap).max()<1e-11:return result
                J=G@self.Pdep;step=-J.T@np.linalg.solve(J@J.T,gap);mujoco.mj_integratePos(self.m,result,self.Pdep@step,1.)
            raise RuntimeError(('Map closure refinement',gap))
        if key in self.cache:return configuration(self.cache[key])
        def fun(delta):return self.geometry(configuration(delta))[0]
        def jac(delta):return self.geometry(configuration(delta))[1]@self.Pdep
        fit=minimize(lambda delta:.5*float(delta@delta),np.zeros(10),jac=lambda delta:delta,constraints=[dict(type='eq',fun=fun,jac=jac)],method='SLSQP',options=dict(ftol=1e-12,maxiter=200))
        if abs(fun(fit.x)).max()>1e-8:raise RuntimeError(('Scapular closure',key,fit.message,fun(fit.x)))
        self.cache[key]=fit.x.copy();self.last=fit.x.copy();return configuration(fit.x)

    def make_mapping(self):
        from pathlib import Path
        signature=hashlib.sha256(Path(__file__).read_bytes()+json.dumps(self.b.report['subject'],sort_keys=True).encode()).hexdigest();dest=ROOT/'results/jump_v2'/('scapular_geometry_map_'+signature[:16]+'.npz')
        if dest.exists():
            data=np.load(dest)
            if str(data['implementation_sha256'])==signature:
                self.mapping=CubicSpline(data['angles'],data['offsets'],axis=0);return
        angles=np.arange(-.025,1.875+.0001,.025);offsets=[]
        for i,angle in enumerate(angles):
            q=self.qneutral.copy();self.rig_arm(q,float(angle));self.complete(q);key=(round(float(angle),10),)*2;offsets.append(self.cache[key])
            if i%20==0:print('SCAPULAR_GEOMETRY_MAP',i,len(angles),flush=True)
        offsets=np.array(offsets);np.savez_compressed(dest,angles=angles,offsets=offsets,implementation_sha256=signature);self.mapping=CubicSpline(angles,offsets,axis=0)

    def rig_arm(self,q,angle):
        for side in ['r','l']:q[self.m.jnt_qposadr[self.b.joints['shoulder_elv_'+side]]]=angle

    def pose(self,q,velocity=None,acceleration=None):
        q=self.complete(q);self.d.qpos[:]=q;self.d.qvel[:]=0 if velocity is None else velocity;mujoco.mj_forward(self.m,self.d);self.d.qacc[:]=0 if acceleration is None else acceleration

    def coordinates(self,q,u):
        B,c=super().coordinates(q,u);B[self.dep_dofs]=0;c[self.dep_dofs]=0;eps=5e-4
        for side in ['r','l']:
            jid=self.b.joints['shoulder_elv_'+side];qa=self.m.jnt_qposadr[jid];col=6+self.b.ind_names.index('shoulder_elv_'+side);qq=[]
            for sign in [-1,1]:
                qp=q.copy();qp[qa]+=sign*eps;qq.append(self.complete(qp))
            v=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,v,2*eps,qq[0],qq[1]);B[self.dep_dofs,col]=v[self.dep_dofs]
            for dof in self.dep_dofs:
                qa2=int(self.m.jnt_qposadr[self.m.dof_jntid[dof]]);c[dof]+=(qq[1][qa2]-2*q[qa2]+qq[0][qa2])/eps**2*u[col]**2
        self.geometry(q)
        return B,c

    def contact_operators(self,q):
        gaps,G,points=self.geometry(q);C=np.column_stack([p[4] for p in points]);return C,points,gaps


if __name__=='__main__':
    r=ScapulothoracicRig();out=[]
    for angle in np.linspace(.1,1.5,15):
        q=r.b.q0.copy();r.arm(q,1.4,angle,.25);q=r.b.complete(q);u=np.linspace(-.3,.3,r.n);B,c=r.coordinates(q,u);C,p,g=r.contact_operators(q)
        out.append(dict(elevation=float(angle),gap=float(abs(g).max()),contact_work_coefficient_m_s=float(abs(C.T@B@u).max()),dependent_angles=q[r.dependent].tolist(),acceleration_constraint=float(abs(r.geometry(q)[1]@c).max())))
    dest=ROOT/'results/jump_v2/scapular_contact_checks.json';dest.write_text(json.dumps(dict(profile=r.contact_profile,samples=out),indent=2));print('SCAPULAR_CONTACT',max(v['gap'] for v in out),max(v['contact_work_coefficient_m_s'] for v in out))
