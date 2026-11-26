"""Conservative native tendon port of published static high-flexion leg routes.

Only hamstring/gastrocnemius attachments and cylinders change. Skeleton,
external loads and physical parameters are preserved. No force fitting/capping.
"""
from pathlib import Path
import xml.etree.ElementTree as ET,json,hashlib,copy
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
STEMS=['bfsh','bflh','semimem','semiten','gasmed','gaslat']

def apply(r):
 source=ROOT/'datasets/cmj_calisti2025/source/catelli_3dof_model.osim';doc=ET.parse(source);old=r.m;b=r.b;g=b.report['global_normalization_scale'];spec=b.spec.copy();wraps={};changes=[]
 def fmt(x):return ' '.join(format(float(v),'.17g') for v in x)
 for body in doc.findall('.//BodySet/objects/Body'):
  for item in body.findall('./WrapObjectSet/objects/*'):wraps[item.attrib['name']]=(body.attrib['name'],item)
 def depth(name):
  i=old.body(name).id;n=0
  while i:n+=1;i=int(old.body_parentid[i])
  return n
 for side in ['r','l']:
  for stem in STEMS:
   name=stem+'_'+side;mus=doc.find('.//Millard2012EquilibriumMuscle[@name="'+name+'"]');pts=mus.findall('./GeometryPath/PathPointSet/objects/*');pws=mus.findall('./GeometryPath/PathWrapSet/objects/PathWrap');assert all(p.tag=='PathPoint' for p in pts)
   j=b.names.index(name);tid=int(old.actuator_trnid[j,0]);oldroute=spec.tendon(old.tendon(tid).name);rn=oldroute.name;defaults=oldroute.default;spec.delete(oldroute);route=spec.add_tendon(name=rn);desired=[]
   sites=[]
   for k,p in enumerate(pts):
    body=p.findtext('socket_parent_frame').split('/')[-1];parent=spec.body(body);sn='catelli_'+name+'_p'+str(k);parent.add_site(name=sn,pos=np.fromstring(p.findtext('location'),sep=' ')*g,size=[.0001]*3,group=3);sites.append(sn)
   desired.append(('site',sites[0],''));wlist=sorted([pw.findtext('wrap_object') for pw in pws],key=lambda n:depth(wraps[n][0]))
   # Native MuJoCo permits one cylinder between adjacent attachment sites.
   # Keep the knee obstacle for two-obstacle routes; the omitted proximal/
   # shank obstacle is disclosed and moment-arm agreement is audited.
   omitted=[]
   if len(wlist)>1:
    keep=[wn for wn in wlist if ('condyles' in wn)]
    omitted=[wn for wn in wlist if wn not in keep];wlist=keep
   for k,wn in enumerate(wlist):
    body,item=wraps[wn];assert item.tag=='WrapCylinder';parent=spec.body(body);gn='catelli_'+name+'_w'+str(k);pos=np.fromstring(item.findtext('translation'),sep=' ')*g;rot=Rotation.from_euler('XYZ',np.fromstring(item.findtext('xyz_body_rotation'),sep=' '));radius=float(item.findtext('radius'))*g
    parent.add_geom(name=gn,type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=pos,quat=np.roll(rot.as_quat(),1),size=[radius,float(item.findtext('length'))*g/2,0],contype=0,conaffinity=0,mass=0,group=3,rgba=[0,0,0,0])
    sn='';quad=item.findtext('quadrant','all')
    if quad!='all':
     direction=np.zeros(3);axis=quad[-1];assert axis in 'xy';direction['xyz'.index(axis)]=(-1 if quad.startswith('-') else 1)*1.5*radius;sn=gn+'_side';parent.add_site(name=sn,pos=pos+rot.apply(direction),size=[.0001]*3,group=3)
    desired.append(('geom',gn,sn))
   for sn in sites[1:]:desired.append(('site',sn,''))
   for kind,target,ss in desired:
    if kind=='site':route.wrap_site(target)
    else:route.wrap_geom(target,ss)
   changes.append(dict(element=name,wraps=wlist,omitted_source_wraps=omitted,site_count=len(sites),source_optimal_fiber_length_m=float(mus.findtext('optimal_fiber_length'))*g,source_tendon_slack_length_m=float(mus.findtext('tendon_slack_length'))*g))
 # MjSpec predates the task's rigid baseline/inertia/high-flex profiles. Reapply
 # every runtime physical/joint/actuator parameter after compiling only paths.
 new=spec.compile()
 for kind,count in [('body',old.nbody),('joint',old.njnt),('actuator',old.nu)]:
  assert count==getattr(new,{'body':'nbody','joint':'njnt','actuator':'nu','tendon':'ntendon'}[kind]);assert [getattr(old,kind)(i).name for i in range(count)]==[getattr(new,kind)(i).name for i in range(count)]
 assert (old.nq,old.nv,old.neq)==(new.nq,new.nv,new.neq)
 fields=['body_mass','body_inertia','body_iquat','body_ipos','body_pos','body_quat','body_gravcomp','dof_armature','dof_damping','dof_frictionloss','jnt_range','jnt_stiffness','jnt_margin','jnt_solref','jnt_solimp','eq_data','eq_active0','eq_solref','eq_solimp','qpos0','qpos_spring','actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_forcerange','actuator_actrange','actuator_lengthrange','actuator_gear',]
 for field in fields:getattr(new,field)[:]=getattr(old,field)
 for field in ['tendon_stiffness','tendon_damping','tendon_frictionloss','tendon_range']:
  for i in range(old.ntendon):getattr(new,field)[new.tendon(old.tendon(i).name).id]=getattr(old,field)[i]
 new.opt=copy.copy(old.opt)
 for item in changes:
  j=b.names.index(item['element']);new.actuator_lengthrange[j]=item['source_tendon_slack_length_m']+new.actuator_gainprm[j,:2]*item['source_optimal_fiber_length_m']
 r.m=b.m=new;r.d=b.d=mujoco.MjData(new);r.d.eq_active[:]=0;mujoco.mj_setConst(new,r.d);b.pose(b.q0)
 assert np.array_equal(new.body_mass,old.body_mass) and np.array_equal(new.body_inertia,old.body_inertia)
 report=dict(kind='published_single_obstacle_high_flexion_leg_approximation',source_file=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),changed=changes,force_scale_changed=False,force_ceiling_added=False,kinematics_changed=False,wrapping='Native conservative MuJoCo cylinders, source static bone-frame attachments and wrapping obstacles. Cylinders are infinite-axis wrapping approximations.',not_individual_anatomy=True,physical_arrays_preserved=fields)
 r.profile['published_knee_paths']=report;b.report['published_knee_paths']=report;return report

