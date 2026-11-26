"""Measured-marker replay and independent segment balance, no residual correction.

Reference-model inverse dynamics is cross-checked by explicit body wrenches.
An equivalent single-plantarflexor force is a diagnostic proxy, NOT a muscle
sharing solution or a measured Achilles tendon force.
"""
import argparse,json,xml.etree.ElementTree as ET,numpy as np
from pathlib import Path
from scipy.signal import butter,sosfiltfilt
from scipy.interpolate import CubicSpline
import opensim as osim
from newton_euler_reference import body_wrenches

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/measured_cmj';G=9.80665
def vec(v):return np.array([v.get(i) for i in range(3)])
def mat(M):return np.array([[M.get(i,j) for j in range(3)] for i in range(3)])
def table(path):
    z=osim.TimeSeriesTable(str(path));return np.array(z.getIndependentColumn()),list(z.getColumnLabels()),z.getMatrix().to_numpy()

def mechanics_model():
    p=OUT/'reference_model/scaled_model.osim';tree=ET.parse(p);node=tree.find('.//ForceSet/objects')
    for child in list(node):node.remove(child)
    target=OUT/'reference_model/mechanics_only_model.osim';tree.write(target,encoding='utf-8',xml_declaration=True);m=osim.Model(str(target));m.setGravity(osim.Vec3(0,-G,0));s=m.initSystem();return m,s

def mapping(m,s):
    cs=list(m.getCoordinateSet());qi=[];ui=[]
    for c in cs:
        if c.getDefaultLocked():qi.append(-1);ui.append(-1);continue
        original=c.getValue(s);before=s.getQ().to_numpy().copy();c.setValue(s,original+1e-5,False);delta=s.getQ().to_numpy()-before;j=int(abs(delta).argmax());assert abs(delta[j]-1e-5)<1e-10 and np.sum(abs(delta)>1e-10)==1;c.setValue(s,original,False);qi.append(j)
        before=s.getU().to_numpy().copy();original=c.getSpeedValue(s);c.setSpeedValue(s,original+1e-5);delta=s.getU().to_numpy()-before;j=int(abs(delta).argmax());assert abs(delta[j]-1e-5)<1e-10 and np.sum(abs(delta)>1e-10)==1;c.setSpeedValue(s,original);ui.append(j)
    assert len(set(i for i in qi if i>=0))==sum(i>=0 for i in qi) and len(set(i for i in ui if i>=0))==sum(i>=0 for i in ui);names=[c.getName() for c in cs];dependent={};tree=ET.parse(OUT/'reference_model/scaled_model.osim')
    for node in tree.findall('.//CoordinateCouplerConstraint'):
        slope,offset=map(float,node.findtext('coupled_coordinates_function/LinearFunction/coefficients').split());scale=float(node.findtext('scale_factor'));dependent[names.index(node.findtext('dependent_coordinate_name'))]=(names.index(node.findtext('independent_coordinate_names')),slope*scale,offset*scale)
    independent=[i for i,c in enumerate(cs) if not c.getDefaultLocked() and i not in dependent];B=np.zeros((s.getNU(),len(independent)))
    for j,i in enumerate(independent):B[ui[i],j]=1
    for dep,(ind,slope,_) in dependent.items():B[ui[dep]]=slope*B[ui[ind]]
    return cs,names,np.array(qi),np.array(ui),dependent,independent,B

def analyze(trial,cutoff=15.):
    m,s=mechanics_model();cs,names,qi,ui,dep,ind,B=mapping(m,s);t,labels,values=table(OUT/trial/'ik.mot');measured=np.load(OUT/f'{trial}_measured.npz');assert np.max(abs(t-measured['times']))<1e-8
    q=np.array([values[:,labels.index(n)] for n in names]).T
    for j,c in enumerate(cs):
        if c.getMotionType()==1:q[:,j]=np.deg2rad(q[:,j])
    q=sosfiltfilt(butter(3,cutoff,fs=1/(t[1]-t[0]),output='sos'),q,axis=0);spl=CubicSpline(t,q);v=spl(t,1);a=spl(t,2)
    for j,c in enumerate(cs):
        if c.getDefaultLocked():q[:,j]=c.getDefaultValue();v[:,j]=0;a[:,j]=0
    for j,(independent,slope,offset) in dep.items():q[:,j]=slope*q[:,independent]+offset;v[:,j]=slope*v[:,independent];a[:,j]=slope*a[:,independent]
    bodies=list(m.getBodySet());nb=m.getNumBodies()+1;body_names=['ground']+['']*(nb-1);mass=np.zeros(nb);inertia=np.zeros((nb,3,3));local=np.zeros((nb,3));ids=[]
    for body in bodies:
        i=int(body.getMobilizedBodyIndex());ids.append(i);body_names[i]=body.getName();mass[i]=body.getMass();local[i]=vec(body.getMassCenter());I=body.getInertia();xx,yy,zz=vec(I.getMoments());xy,xz,yz=vec(I.getProducts());inertia[i]=[[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]
    J=osim.Matrix();A=osim.VectorOfSpatialVec();matter=m.getMatterSubsystem();solver=osim.InverseDynamicsSolver(m);full=osim.Model(str(OUT/'reference_model/scaled_model.osim'));fs=full.initSystem();muscles=[full.getMuscles().get(stem+'_'+side) for side in ['r','l'] for stem in ['soleus','gasmed','gaslat']];calf_names=[x.getName() for x in muscles];ankles=[full.getCoordinateSet().get('ankle_angle_'+side) for side in ['r','l']];ms=list(m.getMarkerSet());mlabels=list(measured['marker_names']);marker_ids=[mlabels.index(marker.getName()) for marker in ms];lower=[i for i,marker in enumerate(ms) if marker.getName()[1:] in ['ASI','PSI','GT','CI','KNE','KNEM','ANK','ANKM','HEE','TOE'] or marker.getName()[1:].startswith(('THI','TIB'))]
    arrays={key:[] for key in ['COM_m','COM_velocity_m_s','COM_acceleration_m_s2','GRF_from_kinematics_N','root_force_residual_N','root_moment_about_COM_residual_Nm','net_generalized_load','body_origins_m','body_rotations','calf_moment_arm_m','calf_path_length_m','marker_RMS_m','lower_body_marker_RMS_m','marker_max_m','generalized_crosscheck_error','velocity_adapter_error']}
    for k,tt in enumerate(t):
        Q=s.getQ().to_numpy().copy();U=np.zeros(s.getNU());UD=np.zeros(s.getNU());valid=qi>=0;Q[qi[valid]]=q[k,valid];valid=ui>=0;U[ui[valid]]=v[k,valid];UD[ui[valid]]=a[k,valid];s.setTime(float(tt));s.setQ(osim.Vector(Q));s.setU(osim.Vector(U));m.realizeVelocity(s);matter.calcSystemJacobian(s,J);jac=J.to_numpy();vel=(jac@U).reshape(nb,6);matter.calcBodyAccelerationFromUDot(s,osim.Vector(UD),A);orig=np.zeros((nb,3));R=np.repeat(np.eye(3)[None],nb,0);com=orig.copy();vc=orig.copy();acc=orig.copy();omega=vel[:,:3];alpha=orig.copy();Iw=np.zeros_like(inertia)
        for body,i in zip(bodies,ids):
            X=body.getTransformInGround(s);orig[i]=vec(X.p());R[i]=mat(X.R().asMat33());r=R[i]@local[i];com[i]=orig[i]+r;vc[i]=vel[i,3:]+np.cross(omega[i],r);alpha[i]=vec(A.get(i).get(0));acc[i]=vec(A.get(i).get(1))+np.cross(alpha[i],r)+np.cross(omega[i],np.cross(omega[i],r));Iw[i]=R[i]@inertia[i]@R[i].T
        own=body_wrenches(mass,com,Iw,acc,omega,alpha,[0,-G,0]);spatial=np.c_[own[:,3:]-np.cross(orig,own[:,:3]),own[:,:3]];gen=jac.T@spatial.ravel();engine=solver.solve(s,osim.Vector(UD)).to_numpy();error=float(abs(gen-engine).max())
        if error>1e-5:raise RuntimeError(('Independent body wrench mismatch',trial,k,error))
        ext=np.zeros((nb,6));total_F=np.zeros(3);total_M=np.zeros(3)
        for j,side in enumerate(measured['plate_sides']):
            i=body_names.index('calcn_'+str(side));F=measured['applied_ground_force_N'][k,j];p=np.nan_to_num(measured['COP_m'][k,j]);T=measured['applied_free_moment_Nm'][k,j];ext[i,:3]+=np.cross(p-orig[i],F)+T;ext[i,3:]+=F;total_F+=F;total_M+=np.cross(p,F)+T
        demand=B.T@(gen-jac.T@ext.ravel());C=mass@com/mass.sum();CV=mass@vc/mass.sum();CA=mass@acc/mass.sum();force=own[:,:3].sum(0);moment=own[:,3:].sum(0)-total_M-np.cross(C,force-total_F)
        fs.setTime(float(tt));fs.setQ(osim.Vector(Q));full.realizePosition(fs);arms=[muscle.computeMomentArm(fs,ankles[j//3]) for j,muscle in enumerate(muscles)];lengths=[muscle.getLength(fs) for muscle in muscles]
        pred=np.array([vec(marker.getLocationInGround(s)) for marker in ms]);errors=np.linalg.norm(pred-measured['markers_m'][k,marker_ids],axis=1);adapter_error=0.
        if k%100==0:
            adapter_error=max(np.max(abs(np.r_[vec(body.getVelocityInGround(s).get(0)),vec(body.getVelocityInGround(s).get(1))]-vel[i])) for body,i in zip(bodies,ids))
        vals=dict(COM_m=C,COM_velocity_m_s=CV,COM_acceleration_m_s2=CA,GRF_from_kinematics_N=force,root_force_residual_N=force-total_F,root_moment_about_COM_residual_Nm=moment,net_generalized_load=demand,body_origins_m=orig,body_rotations=R,calf_moment_arm_m=arms,calf_path_length_m=lengths,marker_RMS_m=np.sqrt(np.mean(errors**2)),lower_body_marker_RMS_m=np.sqrt(np.mean(errors[lower]**2)),marker_max_m=errors.max(),generalized_crosscheck_error=error,velocity_adapter_error=adapter_error)
        for key,value in vals.items():arrays[key].append(value)
        if k%200==0:print('MEASURED_DYNAMICS',trial,k,len(t),'error',error,flush=True)
    arrays={key:np.array(value) for key,value in arrays.items()};report_input=json.loads((OUT/'input_checks.json').read_text());input_trial=next(r for r in report_input['reports'] if r['trial']==trial);takeoff,landing=input_trial['flight_interval_s'];stance=(t>.2)&(t<takeoff-.03);W=mass.sum()*G;err=arrays['root_force_residual_N'][stance];rms=np.sqrt(np.mean(err**2,axis=0));jmoment={}
    for side in ['r','l']:
        for joint in ['ankle_angle','knee_angle','hip_flexion']:
            name=joint+'_'+side;j=ind.index(names.index(name));jmoment[name]=float(np.max(abs(arrays['net_generalized_load'][stance,j])))
    report=dict(trial=trial,status='reference_model_measured_replay',mass_kg=float(mass.sum()),kinematic_derivative_filter_Hz=cutoff,external_loads='Measured filtered force/COP, 20 N per-plate application threshold; no amplitude rescaling.',no_residual_force_correction=True,mean_marker_RMS_m=float(arrays['marker_RMS_m'].mean()),mean_lower_body_marker_RMS_m=float(arrays['lower_body_marker_RMS_m'].mean()),max_marker_error_m=float(arrays['marker_max_m'].max()),independent_NE_vs_OpenSim_max_SI=float(arrays['generalized_crosscheck_error'].max()),velocity_adapter_max_error=float(arrays['velocity_adapter_error'].max()),stance_root_force_RMS_N=rms.tolist(),stance_vertical_force_RMS_bodyweight=float(rms[1]/W),stance_root_moment_RMS_Nm=np.sqrt(np.mean(arrays['root_moment_about_COM_residual_Nm'][stance]**2,axis=0)).tolist(),net_joint_moment_peaks_Nm=jmoment,calf_moment_arm_range_cm={name:[float(np.min(abs(arrays['calf_moment_arm_m'][stance,j]))*100),float(np.max(abs(arrays['calf_moment_arm_m'][stance,j]))*100)] for j,name in enumerate(calf_names)},takeoff_s=takeoff,landing_s=landing,stance_evaluation='0.2 s to 30 ms before first flight; includes countermovement.',muscle_tensions_solved=False,human_muscle_forces_validated=False,independent_anatomical_validation=False)
    arrays.update(times=t,coordinates=np.array([names[i] for i in ind]),coordinate_values=q[:,ind],coordinate_velocities=v[:,ind],body_names=np.array(body_names),body_mass_kg=mass,calf_names=np.array(calf_names));dest=OUT/trial;stem='mechanics_replay' if cutoff==15 else f'mechanics_replay_{cutoff:g}hz';np.savez_compressed(dest/(stem+'.npz'),**arrays);(dest/(stem+'.json')).write_text(json.dumps(report,indent=2));print('MEASURED_ANALYSIS_COMPLETE',json.dumps(report),flush=True);return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trial',default='s10_CMJ_t1');ap.add_argument('--cutoff',type=float,default=15.);args=ap.parse_args();osim.Logger.setLevelString('error');analyze(args.trial,args.cutoff)

if __name__=='__main__':main()
