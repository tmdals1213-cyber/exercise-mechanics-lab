"""One solved tension record drives anatomical and simplified display rigs."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import linprog
from jump_rig import JumpRig
from build_tension_reference import spatial_force_matrix
from muscle_mechanics import subtree
from tension_reference import reaction_cone,scale_parameters,solve_causal_history
from audit_jump_muscles import GROUPS
from tension_groups import REGISTRY as ALL_REGISTRY,matches,MASK_STEP,OVERVIEW
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/tension_visualization'
REGISTRY=[
 ('deltoid','Deltoid',('DELT',),'DELT2'),('pectoralis','Pectoralis',('PECM',),'PECM2'),
 ('triceps','Triceps',('TRI',),'TRIlong'),('biceps','Biceps',('BIC',),'BIClong'),
 ('rotator_cuff','Rotator cuff',('SUPSP','INFSP','SUBSC','TMIN'),'INFSP'),
 ('latissimus','Latissimus',('LAT',),'LAT2'),
 ('abdominals','Abdominals',('rect_abd','EO','IO'),'rect_abd'),
 ('spinal_extensors','Spinal extensors',('IL_','LTp','MF_'),'IL_L3'),
 ('gluteals','Gluteals',('glmax','glmed','glmin'),'glmax2'),
 ('quadriceps','Quadriceps',('vas','recfem'),'vaslat'),
 ('hamstrings','Hamstrings',('bflh','bfsh','semimem','semiten'),'semimem'),
 ('calf','Calf',('gas','soleus'),'soleus'),
 ('hip_flexors','Hip flexors',('psoas','iliacus','Ps_'),'iliacus'),
 ('adductors','Adductors',('addlong','addbrev','addmag'),'addlong'),
 ('forearm','Forearm',('FCR','FCU','ECRL','ECRB','ECU','PT','PQ','FDS','FDP','EDC'),'FCR')]

def side_of(name):
    if name.endswith('_l'):return 'l'
    return 'r'

def resample_path(points,n=8):
    points=np.asarray(points);keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-10];points=points[keep]
    if len(points)<2:return np.repeat(points[:1],n,axis=0)
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];u=np.linspace(0,distance[-1],n)
    return np.array([np.interp(u,distance,points[:,j]) for j in range(3)]).T

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fps',type=int,default=30);args=ap.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    folder=ROOT/'results/jump_validation/vertical/acceleration_2400hz';z=dict(np.load(folder/'motion.npz'));loads=dict(np.load(folder/'loads.npz'));setup=json.loads((folder/'setup.json').read_text(encoding='utf-8'));cfg=json.loads((ROOT/'scenarios/tension_reference.json').read_text(encoding='utf-8'))
    target_times=np.r_[np.arange(0,z['times'][-1],1/args.fps),z['times'][-1]];indices=np.unique([int(np.argmin(abs(z['times']-t))) for t in target_times]);times=z['times'][indices]
    rig=JumpRig();b=rig.b;m,d=b.m,b.d;groups=[];representatives=[]
    for key,label,prefixes,preferred in ALL_REGISTRY:
        for side in ['r','l']:
            ids=[j for j,name in enumerate(b.names) if matches(name,prefixes) and side_of(name)==side]
            if not ids:continue
            wanted=preferred+('_l' if side=='l' and preferred[0].isupper() and not preferred.startswith(('IL_','LTp','MF_','Ps_')) else '_'+side if not preferred[0].isupper() or preferred.startswith(('IL_','LTp','MF_','Ps_')) else '')
            representative=b.names.index(wanted) if wanted in b.names and b.names.index(wanted) in ids else ids[0]
            review=key=='latissimus';gid=len(groups)+1
            groups.append({'id':gid,'key':key,'label':label,'side':side,'indices':ids,'representative':b.names[representative],'status':'parameter_review' if review else 'computed','display_id':0 if review else gid});representatives.append(representative)
    assigned=[i for g in groups for i in g['indices']]
    assert sorted(assigned)==list(range(m.nu)),'Every actuator must appear exactly once'
    assert len(groups)*MASK_STEP<255
    node_names=['pelvis','lumbar','chest','neck','head']+[name+'_'+side for side in ['r','l'] for name in ['shoulder','elbow','wrist','hand','hip','knee','ankle','heel','toe']]
    def ni(name):return node_names.index(name)
    links=[('pelvis','lumbar'),('lumbar','chest'),('chest','neck'),('neck','head')]
    for s in ['r','l']:links += [('chest','shoulder_'+s),('shoulder_'+s,'elbow_'+s),('elbow_'+s,'wrist_'+s),('wrist_'+s,'hand_'+s),('pelvis','hip_'+s),('hip_'+s,'knee_'+s),('knee_'+s,'ankle_'+s),('ankle_'+s,'heel_'+s),('heel_'+s,'toe_'+s)]
    links=[(ni(a),ni(c)) for a,c in links];definition=json.loads((folder/'contact_definition.json').read_text(encoding='utf-8'));geometry=json.loads((ROOT/'results/grip_comparison/grip_65/cycle/physiology_operators.json').read_text(encoding='utf-8'))['geometry'];parts=[subtree(m,m.body('humerus_'+s).id) for s in ['r','l']]
    arrays={key:[] for key in ['R','target','gain','passive','S','offset','axes','nodes','paths']}
    for count,index in enumerate(indices):
        q=z['qpos'][index];b.pose(q,z['qvel'][index],z['qacc'][index]);B,_=rig.coordinates(q,z['u'][index]);R=(B.T@b.moments())[6:];gain=[];passive=[]
        for j in range(m.nu):
            lr=np.ascontiguousarray(m.actuator_lengthrange[j]);L=float(d.actuator_length[j]);V=float(d.actuator_velocity[j]);acc=float(m.actuator_acc0[j]);gain.append(max(0.,-mujoco.mju_muscleGain(L,V,lr,acc,np.ascontiguousarray(m.actuator_gainprm[j,:9]))));passive.append(max(0.,-mujoco.mju_muscleBias(L,lr,acc,np.ascontiguousarray(m.actuator_biasprm[j,:9]))))
        A=spatial_force_matrix(b);S=np.array([A[part].sum(0) for part in parts]);offset=[];axes=[]
        for part,g in zip(parts,geometry):
            required=np.zeros(3)
            for bid in part:
                if m.body_mass[bid]<=0:continue
                J=np.zeros((3,m.nv));Jr=np.zeros_like(J);Jdot=np.zeros_like(J);Jrdot=np.zeros_like(J);mujoco.mj_jacBodyCom(m,d,J,Jr,bid);mujoco.mj_jacDot(m,d,Jdot,Jrdot,d.xipos[bid],bid);required+=m.body_mass[bid]*(J@d.qacc+Jdot@d.qvel-m.opt.gravity)
            offset.append(required);sb=m.body('scapula_'+g['side']).id;hb=m.body('humerus_'+g['side']).id;centre=d.xpos[sb]+d.xmat[sb].reshape(3,3)@(np.array(g['glenoid_centre_local_native_m'])*b.report['global_normalization_scale']);axis=d.xpos[hb]-centre;axes.append(axis/np.linalg.norm(axis))
        nodes={};hip=[]
        for side in ['r','l']:
            for label,joint in [('shoulder','elv_angle'),('elbow','elbow_flexion'),('wrist','flexion'),('hip','hip_flexion'),('knee','knee_angle'),('ankle','ankle_angle')]:nodes[label+'_'+side]=d.xanchor[b.joints[joint+'_'+side]].copy()
            nodes['hand_'+side]=nodes['wrist_'+side]+d.xmat[m.body('lunate_'+side).id].reshape(3,3)@np.array([0,-.075,0])*b.report['global_normalization_scale'];hip.append(nodes['hip_'+side]);points=[]
            for p in definition['points']:
                if p['body']=='calcn_'+side:points.append((p['forefoot'],b.jacobian(p['body_id'],np.array(p['local_m']))[1]))
            nodes['heel_'+side]=np.mean([p for fore,p in points if not fore],axis=0);nodes['toe_'+side]=np.mean([p for fore,p in points if fore],axis=0)
        nodes['pelvis']=np.mean(hip,axis=0);nodes['lumbar']=d.xpos[m.body('lumbar3').id].copy();nodes['chest']=np.mean([nodes['shoulder_r'],nodes['shoulder_l']],axis=0);nodes['neck']=d.xpos[m.body('cervical_spine').id].copy();apex=d.site_xpos[m.site('subject_head_reference').id];nodes['head']=(nodes['neck']+apex)/2
        paths=[]
        for j in representatives:
            tid=int(m.actuator_trnid[j,0]);start=int(d.ten_wrapadr[tid]);num=int(d.ten_wrapnum[tid]);paths.append(resample_path(d.wrap_xpos.reshape(-1,3)[start:start+num]))
        values={'R':R,'target':loads['net_joint_moment_Nm'][index],'gain':gain,'passive':passive,'S':S,'offset':offset,'axes':axes,'nodes':[nodes[name] for name in node_names],'paths':paths}
        for key,value in values.items():arrays[key].append(value)
        if count%30==0:print('VIZ_OPERATORS',count+1,'/',len(indices),flush=True)
    arrays={key:np.asarray(value) for key,value in arrays.items()};rule=cfg['solution_selection'];stability=[reaction_cone(S,o,a,rule['GH_cone_half_angle_deg'],rule['GH_cone_faces']) for S,o,a in zip(arrays['S'],arrays['offset'],arrays['axes'])];profiles={};checks={};raw={}
    reference=m.actuator_gainprm[:,2].copy()
    b.pose(b.q0);np.savez_compressed(OUT/'geometry.npz',times=times,source_indices=indices,body_xpos=z['body_xpos'][indices],body_xmat=z['body_xmat'][indices],rest_body_xpos=d.xpos.copy(),rest_body_xmat=d.xmat.copy(),nodes=arrays['nodes'],paths=arrays['paths'])
    (OUT/'scene_metadata.json').write_text(json.dumps({'times':times.tolist(),'groups':groups,'scale':b.report['global_normalization_scale'],'body_names':setup['body_names'],'node_names':node_names,'links':links}),encoding='utf-8')
    np.savez_compressed(OUT/'operators.npz',times=times,reference=reference,**{k:v for k,v in arrays.items() if k not in ['nodes','paths']})
    for name,profile in cfg['profiles'].items():
        ref,gain,passive,scale=scale_parameters(b.names,reference,arrays['gain'],arrays['passive'],profile,GROUPS)
        for k,(R,target,P,G,(C,rhs)) in enumerate(zip(arrays['R'],arrays['target'],passive,gain,stability)):
            A=R*ref;eqscale=np.maximum(np.linalg.norm(A,axis=1),1.);U=C*ref;uscale=np.maximum(np.linalg.norm(U,axis=1),1.);lo=P/ref;hi=np.full(m.nu,np.inf);hi[G<1e-8]=lo[G<1e-8];lp=linprog(np.zeros(m.nu),A_eq=A/eqscale[:,None],b_eq=target/eqscale,A_ub=U/uscale[:,None],b_ub=rhs/uscale,bounds=list(zip(lo,hi)),method='highs')
            if not lp.success:
                (OUT/'dense_feasibility_failure.json').write_text(json.dumps({'profile':name,'frame':k,'time_s':float(times[k]),'status':int(lp.status),'message':lp.message}),encoding='utf-8');raise RuntimeError(('Dense frame infeasible',name,k,times[k],lp.message))
        print('VIZ_FEASIBLE',name,len(times),flush=True)
        solution=solve_causal_history(times,arrays['R'],arrays['target'],gain,passive,ref,stability,rule['temporal_weight'],rule['level_weight']);checks[name]=solution.pop('checks');raw[name]=solution
        total=[];active=[];pforce=[]
        for g in groups:
            ids=g['indices'];total.append(solution['force_N'][:,ids].sum(1).round(4).tolist() if g['status']=='computed' else None);active.append(solution['active_force_N'][:,ids].sum(1).round(4).tolist() if g['status']=='computed' else None);pforce.append(solution['passive_force_N'][:,ids].sum(1).round(4).tolist() if g['status']=='computed' else None)
        profiles[name]={'total':total,'active':active,'passive':pforce};np.savez_compressed(OUT/(name+'.npz'),times=times,source_indices=indices,**solution,reference_force_N=ref);print('VIZ_TENSIONS',name,checks[name]['moment_balance_max_error'],flush=True)
    b.pose(b.q0);np.savez_compressed(OUT/'geometry.npz',times=times,source_indices=indices,body_xpos=z['body_xpos'][indices],body_xmat=z['body_xmat'][indices],rest_body_xpos=d.xpos.copy(),rest_body_xmat=d.xmat.copy(),nodes=arrays['nodes'],paths=arrays['paths'])
    metadata={'schema':'exercise-mechanics.dual-view/1.0','fps':args.fps,'times':times.round(8).tolist(),'duration':float(times[-1]),'groups':groups,'node_names':node_names,'links':links,'profiles':profiles,'nodes':arrays['nodes'].round(5).tolist(),'paths':arrays['paths'].round(5).tolist(),'body_names':setup['body_names'],'height_m':1.8,'mass_kg':90.,'scale':b.report['global_normalization_scale'],'takeoff_s':setup['takeoff_s'],'landing_s':setup['landing_s'],'checks':checks,'color_scale':{'quantity':'sum of modeled muscle-tendon element tensions per group/side','units':'N','type':'log1p','ceiling_N':100000,'ceiling_is_color_only':True},'status':'development_reference','unmodeled_groups':['Trapezius','Serratus anterior','Rhomboids','Levator scapulae'],'render_is_analysis_model':False,'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'motion.npz',folder/'loads.npz',ROOT/'scenarios/tension_reference.json']},'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'solver_sha256':hashlib.sha256((ROOT/'scripts/tension_reference.py').read_bytes()).hexdigest()}
    metadata['mask_step']=MASK_STEP;metadata['overview_rows']=[{'label':label,'keys':keys} for label,keys in OVERVIEW]
    metadata['coverage']={'native_elements':m.nu,'assigned_elements':len(assigned),'nonoverlapping':True,'foot_intrinsic_actuators':0,'ankle_and_extrinsic_toe_muscles_included':True}
    metadata['unmodeled_groups'].append('Foot intrinsics')
    metadata['ground_reaction']={};contact_forces=loads['contact_force_world_N'][indices];contact_positions=loads['contact_position_world_m'][indices]
    for side in ['r','l']:
        ids=[j for j,p in enumerate(definition['points']) if p['body']=='calcn_'+side]
        metadata['ground_reaction'][side]=contact_forces[:,ids].sum(axis=1).round(4).tolist()
    metadata['ground_reaction']['is_external_contact_force_not_plantar_muscle_tension']=True
    with np.load(OUT/'geometry.npz') as old:geo={k:old[k] for k in old.files}
    np.savez_compressed(OUT/'geometry.npz',**geo,contact_forces_N=contact_forces,contact_positions_m=contact_positions)
    (OUT/'data.json').write_text(json.dumps(metadata,separators=(',',':')),encoding='utf-8');(OUT/'data.js').write_text('window.TENSION_DATA='+json.dumps(metadata,separators=(',',':'))+';',encoding='utf-8');print('DUAL_VIEW_DATA_COMPLETE',len(times),'frames',len(groups),'groups',flush=True)

if __name__=='__main__':main()
