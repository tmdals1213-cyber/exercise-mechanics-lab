"""Movement-independent, physical-SI full-body morphology builder.

Segment similarity scaling changes actual joints/sites/wrap geoms, not a skin.
This first version scales each selected segment uniformly in its own frame;
it is NOT length-only anisotropic scaling or an individualized anatomy model.
"""
from dataclasses import dataclass,field,asdict
import hashlib,json,warnings,importlib.metadata
from pathlib import Path
import numpy as np
import mujoco,myo_sim

REFERENCE_HEIGHT_M=1.788053
GROUPS={
 'upper_arm':('humerus_r','humerus_l'),
 'forearm':('ulna_r','ulna_l','radius_r','radius_l'),
 'thigh':('femur_r','femur_l','patella_r','patella_l'),
 'shank':('tibia_r','tibia_l'),
 'trunk':('sacrum','lumbar5','lumbar4','lumbar3','lumbar2','lumbar1','torso','Abdomen','chest_r','cervical_spine','head_attach','Arm_attachment','myotorso_root'),
 'pelvis':('pelvis',),
}
FINGER_PREFIXES=('cmc_','mp_flexion','ip_flexion','mcp','pm2','md2','pm3','md3','pm4','md4','pm5','md5')


@dataclass
class Subject:
    height_m:float=1.8
    mass_kg:float=90.
    segment_size:dict=field(default_factory=dict)
    mass_distribution:str='preserve_reference_fractions'
    muscle_force_parameter_scale:float=1.
    muscle_architecture:str='preserve_reference_normalized_length'

    def validate(self):
        if not np.isfinite(self.height_m) or self.height_m<=0 or not np.isfinite(self.mass_kg) or self.mass_kg<=0: raise ValueError('Height and mass must be finite and positive')
        if set(self.segment_size)-set(GROUPS): raise ValueError('Unknown segment groups: '+str(set(self.segment_size)-set(GROUPS)))
        if any(not np.isfinite(x) or x<=0 for x in self.segment_size.values()): raise ValueError('Segment scales must be finite and positive')
        if self.mass_distribution not in ['preserve_reference_fractions','constant_density_then_normalize']: raise ValueError('Unknown mass distribution policy')
        if self.muscle_architecture!='preserve_reference_normalized_length': raise ValueError('Unsupported muscle architecture policy')
        if not np.isfinite(self.muscle_force_parameter_scale) or self.muscle_force_parameter_scale<=0: raise ValueError('Force parameter scale must be finite and positive')


def load_subject_profiles(path):
    data=json.loads(Path(path).read_text(encoding='utf-8')); profiles={name:Subject(**settings) for name,settings in data['profiles'].items()}
    for subject in profiles.values():subject.validate()
    return profiles


def complete_coordinates(m,q):
    q=q.copy()
    for _ in range(m.neq+1):
        prior=q.copy()
        for k in range(m.neq):
            if m.eq_type[k]!=mujoco.mjtEq.mjEQ_JOINT: raise ValueError('Unsupported non-joint equality in this model adapter')
            dep=int(m.eq_obj1id[k]); ind=int(m.eq_obj2id[k]); x=q[m.jnt_qposadr[ind]] if ind>=0 else 0.
            q[m.jnt_qposadr[dep]]=np.polynomial.polynomial.polyval(x,m.eq_data[k,:5])
        if np.max(abs(q-prior))<1e-13:return q
    raise RuntimeError('Coordinate couplers did not converge')


def reference_data(m):
    d=mujoco.MjData(m); d.qpos[:]=complete_coordinates(m,m.qpos0); mujoco.mj_forward(m,d); return d


def _owner(x):
    p=x.parent
    while not isinstance(p,mujoco.MjsBody): p=p.parent
    return p.name


def _scale_spec(spec,base,subject,global_scale):
    factors={b.name:global_scale for b in spec.bodies}
    for group,value in subject.segment_size.items():
        for name in GROUPS[group]:
            if name in factors:factors[name]*=value
    # All currently shared mesh assets must have one geometry scale. This
    # adapter rejects conflicting ownership instead of silently editing both.
    mesh_factors={}
    for g in spec.geoms:
        if g.meshname:
            f=factors[_owner(g)]
            if g.meshname in mesh_factors and abs(mesh_factors[g.meshname]-f)>1e-12:
                raise ValueError('Shared mesh requires per-owner cloning: '+g.meshname)
            mesh_factors[g.meshname]=f
    native_mass=np.array([base.body_mass[base.body(b.name).id] for b in spec.bodies])
    body_factors=np.array([factors[b.name] for b in spec.bodies])
    weights=native_mass*(body_factors**3 if subject.mass_distribution=='constant_density_then_normalize' else 1)
    weights*=subject.mass_kg/weights.sum()
    for b,new_mass in zip(spec.bodies,weights):
        bid=base.body(b.name).id; f=factors[b.name]
        if b.parent is not None:b.pos=np.asarray(b.pos)*factors[b.parent.name]
        if bid==0:continue
        # Freeze compiled native inertials before scaling, including masses
        # that were inferred from source geometry. Cosmetic geometry/density
        # can therefore never unexpectedly reset the intended body masses.
        b.explicitinertial=True; b.mass=float(new_mass); b.ipos=base.body_ipos[bid]*f; b.iquat=base.body_iquat[bid]
        b.fullinertia=[np.nan,0,0,0,0,0]
        ratio=new_mass/base.body_mass[bid] if base.body_mass[bid]>0 else 0.
        b.inertia=base.body_inertia[bid]*(ratio*f*f)
    for frame in spec.frames: frame.pos=np.asarray(frame.pos)*factors[_owner(frame)]
    for j in spec.joints:
        f=factors[_owner(j)]; j.pos=np.asarray(j.pos)*f
        if j.type==mujoco.mjtJoint.mjJNT_SLIDE:
            j.range=np.asarray(j.range)*f; j.ref*=f; j.springref*=f
    for site in spec.sites:
        f=factors[_owner(site)]; site.pos=np.asarray(site.pos)*f; site.size=np.asarray(site.size)*f
        if np.isfinite(site.fromto).all():site.fromto=np.asarray(site.fromto)*f
    for g in spec.geoms:
        f=factors[_owner(g)]; g.pos=np.asarray(g.pos)*f
        if not g.meshname:g.size=np.asarray(g.size)*f
        if np.isfinite(g.fromto).all():g.fromto=np.asarray(g.fromto)*f
        g.margin*=f; g.gap*=f
    for mesh in spec.meshes:
        if mesh.name in mesh_factors:
            f=mesh_factors[mesh.name]; mesh.scale=np.asarray(mesh.scale)*f
    for eq in spec.equalities:
        dep=spec.joint(eq.name1); ind=spec.joint(eq.name2) if eq.name2 else None
        a=factors[_owner(dep)] if dep.type==mujoco.mjtJoint.mjJNT_SLIDE else 1.
        b=factors[_owner(ind)] if ind is not None and ind.type==mujoco.mjtJoint.mjJNT_SLIDE else 1.
        data=np.array(eq.data); data[:5]*=a/(b**np.arange(5)); eq.data=data
    return factors


class ParametricBody:
    """Shared kinematics/dynamics primitives, without an exercise or renderer."""
    def __init__(self,m,subject,report):
        self.m=m;self.d=mujoco.MjData(m);self.subject=subject;self.report=report
        self.q0=complete_coordinates(m,m.qpos0); self.names=[m.actuator(i).name for i in range(m.nu)]
        if m.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('This backend requires a free-root full-body model')
        self.joints={m.joint(i).name:i for i in range(m.njnt)}
        dependent={int(m.eq_obj1id[i]) for i in range(m.neq)}
        self.independent=[i for i in range(1,m.njnt) if i not in dependent and not m.joint(i).name.startswith(FINGER_PREFIXES)]
        self.ind_names=[m.joint(i).name for i in self.independent]
        self.coordinate_names=['root_x','root_y','root_z','root_rx','root_ry','root_rz']+self.ind_names
        self.pose(self.q0)

    def complete(self,q):return complete_coordinates(self.m,q)

    def pose(self,q,velocity=None,acceleration=None):
        self.d.qpos[:]=self.complete(q);self.d.qvel[:]=0 if velocity is None else velocity
        mujoco.mj_forward(self.m,self.d)
        self.d.qacc[:]=0 if acceleration is None else acceleration

    def reduction(self,q):
        B=np.zeros((self.m.nv,6+len(self.independent)));B[:6,:6]=np.eye(6);eps=1e-6
        for k,jid in enumerate(self.independent):
            plus=q.copy();minus=q.copy();a=self.m.jnt_qposadr[jid];plus[a]+=eps;minus[a]-=eps
            v=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,v,2*eps,self.complete(minus),self.complete(plus));B[:,6+k]=v
        return B

    def moments(self):
        d=self.d;m=self.m;R=np.zeros((m.nv,m.nu))
        for a in range(m.nu):
            start=d.moment_rowadr[a];count=d.moment_rownnz[a]
            R[d.moment_colind[start:start+count],a]=-d.actuator_moment[start:start+count]
        return R

    def jacobian(self,bid,local):
        p=self.d.xpos[bid]+self.d.xmat[bid].reshape(3,3)@local;J=np.zeros((3,self.m.nv));Jr=np.zeros_like(J)
        mujoco.mj_jac(self.m,self.d,J,Jr,p,bid);return J,p

    def save(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        binary=np.empty(mujoco.mj_sizeModel(self.m),dtype=np.uint8);mujoco.mj_saveModel(self.m,buffer=binary)
        (folder/'model.mjb').write_bytes(binary.tobytes())
        self.report['model_mjb_sha256']=hashlib.sha256(binary).hexdigest()
        (folder/'subject_model.json').write_text(json.dumps(self.report,indent=2),encoding='utf-8')

    def graph(self):
        m=self.m;dependent={int(m.eq_obj1id[k]):k for k in range(m.neq)};joints=[]
        for jid in range(m.njnt):
            name=m.joint(jid).name
            role='root_balance' if jid==0 else 'coupled' if jid in dependent else 'prescribed_fingers_in_force_adapter' if name.startswith(FINGER_PREFIXES) else 'independent'
            item={'name':name,'body':m.body(int(m.jnt_bodyid[jid])).name,'type':int(m.jnt_type[jid]),'range':m.jnt_range[jid].tolist(),'role':role}
            if jid in dependent:
                eq=dependent[jid];ind=int(m.eq_obj2id[eq]);item['coupled_to']=m.joint(ind).name if ind>=0 else 'constant';item['polynomial']=m.eq_data[eq,:5].tolist()
            joints.append(item)
        muscles=[]
        for aid in range(m.nu):
            tid=int(m.actuator_trnid[aid,0]);path=[]
            for k in range(m.tendon_adr[tid],m.tendon_adr[tid]+m.tendon_num[tid]):
                typ=int(m.wrap_type[k]);oid=int(m.wrap_objid[k])
                if typ==int(mujoco.mjtWrap.mjWRAP_SITE):
                    path.append({'kind':'attachment_or_via_site','name':m.site(oid).name,'body':m.body(int(m.site_bodyid[oid])).name,'local_position_m':m.site_pos[oid].tolist()})
                elif typ in [int(mujoco.mjtWrap.mjWRAP_SPHERE),int(mujoco.mjtWrap.mjWRAP_CYLINDER)]:
                    path.append({'kind':'wrapping_geometry','name':m.geom(oid).name,'body':m.body(int(m.geom_bodyid[oid])).name})
                else:raise ValueError('Unsupported muscle path in graph')
            assert path and path[0]['kind']=='attachment_or_via_site' and path[-1]['kind']=='attachment_or_via_site'
            muscles.append({'name':m.actuator(aid).name,'origin_body':path[0]['body'],'insertion_body':path[-1]['body'],'path':path})
        expected={'deltoid':('DELT',),'pectoralis_major':('PECM',),'latissimus':('LAT',),'rotator_cuff':('SUPSP','INFSP','SUBSC','TMIN'),'triceps':('TRI',),'biceps':('BIC',),'rectus_abdominis':('rect_abd',),'obliques':('EO','IO'),'spinal_extensors':('IL_','LTp'),'gluteals':('glmax','glmed','glmin'),'quadriceps':('vas','recfem'),'hamstrings':('bflh','bfsh','semimem','semiten'),'adductors':('add',),'hip_flexors':('psoas','iliacus'),'calf':('gas','soleus'),'trapezius':('Trapezius','TRAP'),'serratus_anterior':('Serratus',),'rhomboids':('Rhomboid',),'levator_scapulae':('Levator',)}
        coverage={group:[name for name in self.names if name.startswith(prefixes)] for group,prefixes in expected.items()}
        return {'schema':'exercise-mechanics.anatomical-graph/1.0','bodies':[{'name':m.body(i).name,'parent':m.body(int(m.body_parentid[i])).name if i else None,'mass_kg':float(m.body_mass[i]),'COM_local_m':m.body_ipos[i].tolist(),'inertia_principal_kg_m2':m.body_inertia[i].tolist()} for i in range(m.nbody)],
                'joints':joints,'muscles':muscles,'major_group_actuator_coverage':coverage,'missing_required_major_groups':[name for name,items in coverage.items() if not items],
                'major_group_anatomical_release_ready':all(coverage.values()),'note':'A path graph establishes modeled connectivity, not anatomical or physiological validation. Missing major groups are explicit release gates, never zero-force muscles.'}


def build_subject(subject):
    subject.validate()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); source=myo_sim.load_spec('myofullbody');source.memory=32*2**20;base=source.compile()
    # Disable solver-generated forces: task adapters solve and audit their
    # explicit external contacts and reduced equalities. Preserve all DOFs.
    base.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    native=reference_data(base)
    # Design landmarks define a consistent stature reference, not a measured
    # skin envelope. In the source standing pose they span REFERENCE_HEIGHT_M.
    hid=base.body('head').id; apex=native.xpos[hid].copy();apex[2]=REFERENCE_HEIGHT_M
    source.body('head').add_site(name='subject_head_reference',pos=native.xmat[hid].reshape(3,3).T@(apex-native.xpos[hid]),size=[.002,0,0])
    for side in ['r','l']:
        bid=base.body('calcn_'+side).id;p=native.xpos[bid].copy();p[2]=0.
        source.body('calcn_'+side).add_site(name='subject_sole_reference_'+side,pos=native.xmat[bid].reshape(3,3).T@(p-native.xpos[bid]),size=[.002,0,0])
    old_force=base.actuator_gainprm[:,2].copy();auto=old_force<=0;old_force[auto]=base.actuator_gainprm[auto,3]/base.actuator_acc0[auto]
    assert np.all(old_force>0)
    for a in source.actuators:
        aid=base.actuator(a.name).id;a.lengthrange=base.actuator_lengthrange[aid]
        for key in ['gainprm','biasprm']:
            p=np.array(getattr(a,key));p[2]=old_force[aid]*subject.muscle_force_parameter_scale;setattr(a,key,p)
        a.forcelimited=False;a.ctrllimited=False
    def construct(global_factor):
        spec=source.copy(); factors=_scale_spec(spec,base,subject,global_factor);model=spec.compile()
        model.opt.gravity[:]=[0,0,-9.80665];model.opt.disableflags|=base.opt.disableflags
        data=reference_data(model);height=data.site_xpos[model.site('subject_head_reference').id,2]-np.mean([data.site_xpos[model.site('subject_sole_reference_'+s).id,2] for s in ['r','l']])
        return spec,model,data,factors,float(height)
    _,shape,shape_data,_,shape_height=construct(1.)
    if shape_height<=0:raise ValueError('Invalid stature after shape transformation')
    scale=subject.height_m/shape_height
    spec,model,data,factors,height=construct(scale)
    ratio=data.actuator_length/native.actuator_length
    if not np.isfinite(ratio).all() or ratio.min()<=0:raise ValueError('Invalid muscle path after morphology change')
    # Explicit architecture assumption: preserve normalized neutral fiber
    # length by scaling both inferred L0 and rigid tendon length together.
    # No inference of personal strength or PCSA from total weight/appearance.
    for a in spec.actuators:
        aid=base.actuator(a.name).id;a.lengthrange=base.actuator_lengthrange[aid]*ratio[aid]
    model=spec.compile();model.opt.gravity[:]=[0,0,-9.80665];model.opt.disableflags|=base.opt.disableflags
    data=reference_data(model)
    physical_mass=float(model.body_mass.sum());assert abs(physical_mass-subject.mass_kg)<1e-7
    height=float(data.site_xpos[model.site('subject_head_reference').id,2]-np.mean([data.site_xpos[model.site('subject_sole_reference_'+s).id,2] for s in ['r','l']]))
    assert abs(height-subject.height_m)<1e-7
    # Inertia must be positive and obey the rigid-body triangle inequalities.
    positive=model.body_mass>0;I=model.body_inertia[positive]
    assert np.all(I>0) and np.max(2*I.max(1)-I.sum(1))<1e-8
    report={'schema':'exercise-mechanics.parametric-subject/1.0','status':'experimental_geometry_and_dynamics_adapter','subject':asdict(subject),'world_units':'physical SI, Z up; no external virtual length scale required',
      'stature_reference_m':height,'stature_definition':'Source standing head/sole design landmarks; not a measured skin envelope.','mass_kg':physical_mass,
      'global_normalization_scale':scale,'body_local_similarity_scales':factors,'body_names':[model.body(i).name for i in range(model.nbody)],'native_generalized_speeds':model.nv,'muscle_elements':model.nu,'coordinate_equalities':model.neq,
      'muscle_neutral_path_length_ratios':dict(zip([model.actuator(i).name for i in range(model.nu)],ratio.tolist())),
      'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runtime':{p:importlib.metadata.version(p) for p in ['mujoco','myo-sim','numpy']},
      'native_contacts_and_equalities_disabled_for_explicit_inverse_dynamics':True,'mjb_is_standalone_forward_human_controller':False,
      'limitations':['Segment scaling is locally uniform: it changes segment width/depth as well as length. Length-only and independent joint breadths are not implemented.','Existing native scapular couplers and missing thoracoscapular actuators remain; this is not completed shoulder-girdle integration.','Neutral muscle architecture scaling preserves a reference dimensionless length, not measured fiber or tendon properties. Full-ROM path continuity and operating range require per-task checks.','Native knee conversion limitations and joint-surface congruence under differential segment scaling require further validation.','The builder preserves all native joint freedoms. The force adapter deliberately reduces coupled coordinates and excludes prescribed fingers; there are no activation-dynamics or hard strength ceilings.','New bodies require new motion/contact fitting; cached baseline trajectories, ground anchors and visual meshes cannot be blindly reused.']}
    result=ParametricBody(model,subject,report)
    # Retain the model description for task-specific forward contact models.
    # Attaching this handle changes no compiled geometry or physical values.
    result.spec=spec
    return result
