"""Resolve saved parent-on-child reactions along rotating bone long axes.

Positive axial force points distally and is termed axial compression; negative
is axial distraction. These are ideal-joint resultant components, NOT cartilage
contact normals, contact pressure, ligament forces, or a new force solution.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/power_comparison'
AXES={
    'shoulder':('humerus','shoulder','elbow'),
    'elbow':('ulna','elbow','wrist'),
    'wrist':('radius','elbow','wrist'),
    'hip':('femur','hip','knee'),
    'knee':('tibia','knee','ankle'),
    'ankle':('tibia','knee','ankle'),
}


def split(force, axis):
    axial=np.sum(force*axis,axis=-1)
    shear_vector=force-axial[...,None]*axis
    shear=np.linalg.norm(shear_vector,axis=-1)
    return axial, np.maximum(axial,0), np.maximum(-axial,0), shear, shear_vector


def verify_convention():
    axis=np.array([[0.,0.,-1.]]*3)
    force=np.array([[0.,0.,-100.],[0.,0.,100.],[30.,40.,-100.]])
    axial,c,t,s,_=split(force,axis)
    np.testing.assert_allclose(c,[100,0,100])
    np.testing.assert_allclose(t,[0,100,0])
    np.testing.assert_allclose(s,[0,0,50])
    from scipy.spatial.transform import Rotation
    R=Rotation.from_rotvec([.3,-.5,.8]).as_matrix()
    rotated=split(force@R.T,axis@R.T)
    for a,b in zip((axial,c,t,s),rotated[:4]):np.testing.assert_allclose(a,b,atol=1e-12)


def analyze(key):
    folder=OUT/('cmj_analysis' if key=='cmj' else key)
    source=folder if key=='cmj' else folder/'analysis'
    visual=source/'visualization'
    z=np.load(visual/'joint_reactions.npz')
    geo=np.load(visual/'geometry.npz')
    data=json.loads((visual/'data.json').read_text())
    names=data['body_names']; ts=z['times']; F=z['joint_wrenches'][...,:3]
    np.testing.assert_array_equal(ts,geo['times'])
    rest={n:geo['rest_nodes'][i] for i,n in enumerate(data['node_names'])}
    axes=[]; definitions=[]
    for joint in data['joints']:
        stem,side=joint['key'].rsplit('_',1)
        bone,start,end=AXES[stem]; bone+='_'+side
        bid=names.index(bone)
        direction=rest[end+'_'+side]-rest[start+'_'+side]
        assert np.linalg.norm(direction)>.1
        direction/=np.linalg.norm(direction)
        local=geo['rest_body_xmat'][bid].reshape(3,3).T@direction
        world=np.einsum('tij,j->ti',geo['body_xmat'][:,bid].reshape(-1,3,3),local)
        axes.append(world)
        definitions.append(dict(joint=joint['key'],axis_frame=bone,axis_local=local.tolist(),
            positive_direction=f'{start}_{side} toward {end}_{side}',
            use='Long-axis axial reaction; not an articular surface normal'))
    axes=np.stack(axes,axis=1)
    np.testing.assert_allclose(np.linalg.norm(axes,axis=-1),1,atol=1e-12)
    axial,compression,distraction,shear,shear_vector=split(F,axes)
    reconstruction=axial[...,None]*axes+shear_vector
    error=float(abs(F-reconstruction).max())
    identity=float(abs(np.sum(F*F,axis=-1)-axial**2-shear**2).max())
    np.testing.assert_allclose(np.sum(F*F,axis=-1),axial**2+shear**2,rtol=1e-12,atol=1e-7)
    assert error<1e-8
    np.savez_compressed(visual/'joint_components.npz',times=ts,force_on_child_world_N=F,
        distal_axis_world=axes,signed_axial_N=axial,axial_compression_N=compression,
        axial_distraction_N=distraction,shear_magnitude_N=shear,shear_world_N=shear_vector)
    rows=[]
    for j,joint in enumerate(data['joints']):
        rows.append(dict(joint=joint['key'],peak_resultant_N=float(z['joint_force_N'][:,j].max()),
            peak_axial_compression_N=float(compression[:,j].max()),
            peak_axial_distraction_N=float(distraction[:,j].max()),peak_shear_N=float(shear[:,j].max())))
    report=dict(status='decomposition_checks_passed',case=key,force_solution_changed=False,
        force_applied_on='child subtree, by proximal body; parent action is equal and opposite',
        force_frame='world; axes rotate with specified bone',
        sign='Positive axial = force toward distal end = axial compression; negative = distraction.',
        definitions=definitions,checks=dict(known_compression_distraction_shear_cases=True,
            rotation_invariance=True,reconstruction_max_error_N=error,
            squared_magnitude_identity_max_absolute_error_N2=identity),
        limitations=['These long axes are explicit conventions, not individually identified contact normals.',
            'Shoulder axial distraction is not the glenohumeral contact-normal component.',
            'Ideal joint reactions do not identify individual ligament tension or cartilage pressure.',
            'Muscle-sharing and source-motion uncertainty remain in the reaction components.'],
        source_joint_reactions_sha256=hashlib.sha256((visual/'joint_reactions.npz').read_bytes()).hexdigest(),
        source_geometry_sha256=hashlib.sha256((visual/'geometry.npz').read_bytes()).hexdigest(),
        peak_components=rows)
    (folder/'joint_components_audit.json').write_text(json.dumps(report,indent=2))
    with (folder/'joint_components.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','joint','signed_axial_N','axial_compression_N','axial_distraction_N','shear_N'])
        for k,t in enumerate(ts):
            for j,joint in enumerate(data['joints']):w.writerow([t,joint['key'],axial[k,j],compression[k,j],distraction[k,j],shear[k,j]])
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,stem in zip(axs.flat,AXES):
        ids=[j for j,item in enumerate(data['joints']) if item['key'].startswith(stem+'_')]
        for value,label,color in [(compression,'Axial compression','#c64b3d'),(distraction,'Axial distraction','#387ab9'),(shear,'Shear magnitude','#27877c')]:
            ax.plot(ts,value[:,ids].mean(1)/1000,label=label,color=color)
        ax.set(title=stem.title(),xlabel='Physical time [s]',ylabel='Mean per side [kN]')
        ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle(key.replace('_',' ').upper()+' / conditional reaction components\n'
        'Rotating bone long axes; ideal joints, not tissue contact pressure',fontsize=15)
    fig.savefig(OUT/(key+'_joint_components.png'),dpi=150);plt.close(fig)
    print(key,json.dumps(rows[8:10]),flush=True)


if __name__=='__main__':
    verify_convention()
    for key in ['cmj','squat_60','squat_100','squat_140']:analyze(key)
