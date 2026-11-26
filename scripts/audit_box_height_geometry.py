"""Configuration-only box-height audit with a fixed CMJ takeoff reference.

This tests an explicit upright tuck family, not a continuous box-jump motion.
It does not infer takeoff power, landing force or athletic capability from a box.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retarget_measured_jump import MeasuredJumpRig

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'


def main():
    rig=MeasuredJumpRig();b,m,d=rig.b,rig.m,rig.d
    src=ROOT/'results/measured_jump_analysis/s10_CMJ_t1';motion=np.load(src/'motion.npz');setup=json.loads((src/'setup.json').read_text())
    k=int(np.argmin(abs(motion['times']-setup['takeoff_s'])));C=motion['com'][k].copy();v=motion['velocity'][k,2];apex=C.copy();apex[2]+=v*v/(2*9.80665)
    b.pose(b.q0);feet=[]
    for side in ['r','l']:
        bid=m.body('calcn_'+side).id;sid=m.site('subject_sole_reference_'+side).id;R=d.xmat[bid].reshape(3,3).copy();origin=d.site_xpos[sid].copy()
        points=np.array([R.T@(origin+np.array([x,y,0])-d.xpos[bid]) for x,y in [(-.04,-.245),(.04,-.245),(-.04,.01),(.04,.01)]])
        feet.append((side,bid,R,points))
    rows=[];poses=[];rejected=[];seeds={side:np.zeros(3) for side in ['r','l']}
    for angle in np.linspace(0,120,61):
        q=motion['qpos'][0].copy();q[3:7]=[1,0,0,0]
        failed=None
        for side,bid,R0,points in feet:
            for name,value in [('hip_flexion',np.deg2rad(angle)),('knee_angle',np.deg2rad(angle)),('hip_adduction',-.10),('mtp_angle',0.)]:q[m.jnt_qposadr[b.joints[name+'_'+side]]]=value
            js=[b.joints[n+'_'+side] for n in ['hip_rotation','ankle_angle','subtalar_angle']];adrs=[m.jnt_qposadr[j] for j in js]
            def fun(x):
                q[adrs]=x;d.qpos[:]=b.complete(q);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
                return Rotation.from_matrix(R0.T@d.xmat[bid].reshape(3,3)).as_rotvec()
            lo=[m.jnt_range[j,0] for j in js];hi=[m.jnt_range[j,1] for j in js]
            fit=least_squares(fun,seeds[side],bounds=(lo,hi),max_nfev=80,ftol=1e-12,xtol=1e-12,gtol=1e-10)
            seeds[side]=fit.x;q[adrs]=fit.x
            error=float(max(abs(fun(fit.x))))
            if error>1e-5:failed=dict(equal_hip_knee_flexion_deg=float(angle),side=side,orientation_residual_rad=error);break
        if failed is not None:rejected.append(failed);continue
        b.pose(q);q[:3]+=apex-d.subtree_com[1];b.pose(q)
        heights=[]
        for side,bid,R0,points in feet:heights.extend((points@d.xmat[bid].reshape(3,3).T+d.xpos[bid])[:,2])
        rows.append(dict(equal_hip_knee_flexion_deg=float(angle),lowest_reference_sole_height_m=float(min(heights)),COM_height_m=float(d.subtree_com[1,2])))
        poses.append(d.qpos.copy())
    height=np.array([r['lowest_reference_sole_height_m'] for r in rows]);angles=np.array([r['equal_hip_knee_flexion_deg'] for r in rows]);best=int(height.argmax());conditions=[]
    for cm in [40,60,80]:
        required=cm/100+.03;clear=required<=height[best]
        theta=float(np.interp(required,height[:best+1],angles[:best+1])) if clear else None
        conditions.append(dict(box_height_m=cm/100,assumed_clearance_m=.03,configuration_demonstrated_in_this_family=bool(clear),required_equal_hip_knee_flexion_deg=theta,
                               power_estimated_from_box_height_W=None))
    report=dict(status='configuration_only_not_a_continuous_jump',source='s10_CMJ_t1',mass_kg=88,height_m=1.8,
                reference_takeoff_COM_height_m=float(C[2]),reference_takeoff_vertical_velocity_m_s=float(v),ballistic_apex_COM_height_m=float(apex[2]),
                marker_reference_apex_COM_height_m=float(motion['com'][:,2].max()),maximum_sole_height_in_tested_family_m=float(height[best]),
                family='Fixed upright pelvis orientation, source quiet upper-body posture, equal hip/knee flexion 0–120 degrees, flat reference sole planes. Original ROM retained.',
                conditions=conditions,angle_sweep=rows,rejected_configuration_fits=rejected,continuous_motion_solved=False,angular_momentum_of_transition_verified=False,landing_contact_solved=False,
                interpretation='At one fixed COM apex height, different sole heights can be constructed by changing tuck. Transition dynamics are not solved. Failure in this restricted configuration family is not a personal maximum or proof of general impossibility.')
    (OUT/'box_height_geometry.json').write_text(json.dumps(report,indent=2),encoding='utf-8');np.savez_compressed(OUT/'box_height_configuration_poses.npz',angles_deg=angles,qpos=poses,reference_apex_COM_m=apex)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');ax.plot(angles,height*100,color='#169f8c',lw=2.5,label='Lowest sole plane / same reference COM apex')
    for cm,color in [(40,'#d8a24d'),(60,'#8e80d6'),(80,'#4c8ab7')]:ax.axhline(cm+3,color=color,ls='--',label=f'{cm} cm box + 3 cm clearance')
    ax.set(xlabel='Equal hip / knee flexion [degrees]',ylabel='Reference sole height above floor [cm]',title='Box-height configuration test · same takeoff reference')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.savefig(OUT/'box_height_geometry.png',dpi=160);plt.close(fig)
    print(json.dumps({k:report[k] for k in ['status','maximum_sole_height_in_tested_family_m','conditions']},indent=2))


if __name__=='__main__':main()
