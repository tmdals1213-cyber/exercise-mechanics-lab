"""Convert synchronized measured C3D markers/plate wrenches to SI/OpenSim.

Preserves source force magnitudes. Filters forces and moments at a fixed
origin, then reconstructs COP; never filters an unloaded COP singularity.
"""
import json,hashlib,csv,numpy as np,ezc3d
from pathlib import Path
from scipy.signal import butter,sosfiltfilt

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'datasets/cmj_calisti2025';OUT=ROOT/'results/measured_cmj';G=9.80665

def filtered(x,hz):return sosfiltfilt(butter(3,15,fs=hz,output='sos'),x,axis=0)

def write_trc(path,t,P,labels,hz):
    with path.open('w',newline='') as f:
        f.write('PathFileType\t4\t(X/Y/Z)\t'+path.name+'\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write(f'{hz}\t{hz}\t{len(t)}\t{len(labels)}\tm\t{hz}\t1\t{len(t)}\nFrame#\tTime\t'+'\t\t\t'.join(labels)+'\t\t\n\t\t'+'\t'.join(f'{a}{i+1}' for i in range(len(labels)) for a in 'XYZ')+'\n\n')
        for i,(tt,p) in enumerate(zip(t,P)):f.write(f'{i+1}\t{tt:.9f}\t'+'\t'.join(f'{x:.9f}' for x in p.ravel())+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True);subject=json.loads((DATA/'subject.json').read_text());static=ezc3d.c3d(str(DATA/'source/static.c3d'),extract_forceplat_data=True);labels=static['parameters']['POINT']['LABELS']['value'];P=static['data']['points'][:3].transpose(2,1,0)*.001
    mean=lambda name:np.median(P[:,labels.index(name)],axis=0)
    forward=(mean('RASI')+mean('LASI')-mean('RPSI')-mean('LPSI'))/2;forward[2]=0;forward/=np.linalg.norm(forward);up=np.array([0.,0.,1.]);right=np.cross(forward,up);rotation=np.array([forward,up,right]);origin=np.mean([p['corners'].mean(1)*.001 for p in static['data']['platform']],axis=0)
    assert abs(np.linalg.det(rotation)-1)<1e-10 and (rotation@(mean('RASI')-mean('LASI')))[2]>0
    plates=[]
    for plate in static['data']['platform']:
        cop=np.median(plate['center_of_pressure'][:,:800],axis=1)*.001
        distances={side:np.linalg.norm((mean(side+'ANK')+mean(side+'ANKM'))/2-cop) for side in ['R','L']};plates.append(min(distances,key=distances.get).lower())
    assert sorted(plates)==['l','r'];reports=[]
    for trial in ['static']+subject['trials']:
        source=DATA/'source'/f'{trial}.c3d';c=ezc3d.c3d(str(source),extract_forceplat_data=True);phz=float(c['header']['points']['frame_rate']);ahz=float(c['header']['analogs']['frame_rate']);stride=int(ahz/phz);assert abs(stride*phz-ahz)<1e-9
        assert c['parameters']['POINT']['UNITS']['value']==['mm'];L=c['parameters']['POINT']['LABELS']['value'];points=c['data']['points'][:3].transpose(2,1,0)*.001;res=c['data']['meta_points']['residuals'][0].T;missing=(res<0)|~np.isfinite(points).all(2);t=np.arange(len(points))/phz;gap_report={}
        for j,name in enumerate(L):
            bad=missing[:,j];gap_report[name]=int(bad.sum())
            if bad.any():
                starts=np.flatnonzero(np.diff(np.r_[False,bad,False].astype(int))==1);ends=np.flatnonzero(np.diff(np.r_[False,bad,False].astype(int))==-1)
                if bad[0] or bad[-1] or max(ends-starts)/phz>.1:raise RuntimeError(('Marker gap needs review',trial,name,int(bad.sum())))
                for axis in range(3):points[bad,j,axis]=np.interp(t[bad],t[~bad],points[~bad,j,axis])
        points=np.einsum('ij,tkj->tki',rotation,points-origin);points=filtered(points,phz);write_trc(OUT/f'{trial}.trc',t,points,L,phz)
        forces=[];cops=[];free=[];originmom=[];wrench_errors=[]
        for plate in c['data']['platform']:
            assert plate['unit_force']=='N' and plate['unit_moment']=='Nmm' and plate['unit_position']=='mm'
            rawF=plate['force'].T;rawM=plate['moment'].T*.001;center=plate['corners'].mean(1)*.001;valid=rawF[:,2]>20
            check=rawM[valid]-np.cross(plate['center_of_pressure'][:,valid].T*.001-center,rawF[valid])-plate['Tz'][:,valid].T*.001;wrench_errors.append(float(abs(check).max()))
            if np.median(rawF[:400,2])<=0:raise RuntimeError('Unexpected force direction: inspect before conversion')
            F=filtered(rawF@rotation.T,ahz)[::stride];Mo=filtered(rawM@rotation.T+np.cross(rotation@(center-origin),rawF@rotation.T),ahz)[::stride];assert len(F)==len(t)
            valid=F[:,1]>20;cop=np.full_like(F,np.nan);cop[valid,0]=Mo[valid,2]/F[valid,1];cop[valid,1]=0;cop[valid,2]=-Mo[valid,0]/F[valid,1];T=Mo-np.cross(np.nan_to_num(cop),F)
            forces.append(F);cops.append(cop);free.append(T);originmom.append(Mo)
        forces=np.stack(forces,1);cops=np.stack(cops,1);free=np.stack(free,1);Mo=np.stack(originmom,1);contact=forces[:,:,1]>20;applied=forces.copy();applied[~contact]=0;free_applied=free.copy();free_applied[~contact]=0;COPexport=np.nan_to_num(cops)
        columns=['time']
        for side in plates:columns += [f'ground_force_{side}_v{a}' for a in 'xyz']+[f'ground_force_{side}_p{a}' for a in 'xyz']+[f'ground_torque_{side}_{a}' for a in 'xyz']
        matrix=np.column_stack([t]+[np.column_stack([applied[:,i],COPexport[:,i],free_applied[:,i]]) for i in range(2)])
        with (OUT/f'{trial}_grf.mot').open('w') as f:
            f.write(f'{trial} measured GRF\nversion=1\nnRows={len(t)}\nnColumns={len(columns)}\ninDegrees=no\nendheader\n'+'\t'.join(columns)+'\n');np.savetxt(f,matrix,delimiter='\t',fmt='%.10g')
        total=forces.sum(1);ankle_moments=[];effective=[]
        for i,side in enumerate(plates):
            ankle=(points[:,L.index(side.upper()+'ANK')]+points[:,L.index(side.upper()+'ANKM')])/2;w=free[:,i]+np.cross(cops[:,i]-ankle,forces[:,i]);w[~contact[:,i]]=np.nan;ankle_moments.append(w);effective.append(np.linalg.norm(w,axis=1)/np.maximum(np.linalg.norm(forces[:,i],axis=1),1))
        quiet=t<.4;meanF=float(total[quiet,1].mean());force_equiv=meanF/G;flight=total[:,1]<40;edges=np.diff(np.r_[False,flight,False].astype(int));intervals=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if (b-a)/phz>.1];interval=next(((a,b) for a,b in intervals if a/phz>.3),None)
        report=dict(trial=trial,frames=len(t),marker_hz=phz,force_hz=ahz,plate_sides=plates,quiet_mean_vertical_force_N=meanF,metadata_mass_kg=subject['mass_kg'],force_equivalent_mass_kg=force_equiv,quiet_force_mass_difference_percent=100*(force_equiv/subject['mass_kg']-1),force_moment_reconstruction_error_Nm=max(wrench_errors),marker_gap_frames={k:v for k,v in gap_report.items() if v},flight_interval_s=[t[interval[0]],interval[1]/phz] if interval else None,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if interval:
            a,b=interval;report.update(peak_propulsion_vertical_GRF_N=float(total[:a,1].max()),flight_time_height_m=G*((b-a)/phz)**2/8,peak_GRF_marker_ankle_moment_Nm={side:float(np.nanmax(abs(ankle_moments[i][:a,2]))) for i,side in enumerate(plates)})
        np.savez_compressed(OUT/f'{trial}_measured.npz',times=t,markers_m=points,marker_names=np.array(L),ground_force_filtered_N=forces,COP_m=cops,free_moment_Nm=free,ground_moment_about_origin_Nm=Mo,applied_ground_force_N=applied,applied_free_moment_Nm=free_applied,contact=contact,plate_sides=np.array(plates),GRF_ankle_marker_moment_Nm=np.stack(ankle_moments,1),external_effective_lever_m=np.stack(effective,1))
        reports.append(report);print('MEASURED_PREPARED',json.dumps(report),flush=True)
    setup=dict(subject=subject,world='X anterior from static pelvis, Y up, Z right',rotation_raw_to_opensim=rotation.tolist(),origin_raw_m=origin.tolist(),filter='3rd-order zero-phase Butterworth 15 Hz, markers and plate force/moment. COP reconstructed after filtering at a fixed origin.',contact_threshold_N=20,application_rule='Filtered forces/free moments below 20 N vertical per plate are set to zero only in the applied external-load trace. Original C3D and filtered measured forces retained.',time='Shared C3D clock; relative zero is first point frame. No trial warping or amplitude scaling.',measured_force_magnitudes_rescaled=False,reports=reports,human_muscle_forces_validated=False)
    (OUT/'input_checks.json').write_text(json.dumps(setup,indent=2))

if __name__=='__main__':main()
