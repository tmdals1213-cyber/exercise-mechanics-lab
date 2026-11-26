"""Adapt checked loaded squats to the full-body muscle-sharing solver.

The reaction of the translating/rotating bar is an explicit external wrench
on the upper back. No root reserve or invented hand contact is introduced.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,mujoco
from retarget_measured_jump import MeasuredJumpRig
from mechanics_engine_adapter import sample

ROOT=Path(__file__).resolve().parents[1]

def main(kg,folder=None):
    src=ROOT/folder if folder else ROOT/f'results/power_comparison/squat_{kg:g}';out=src/'analysis';out.mkdir(exist_ok=True)
    meta=json.loads((src/'summary.json').read_text());assert meta['all_mechanics_gates_passed']
    z=dict(np.load(src/'motion.npz'));rig=MeasuredJumpRig(mass_kg=meta['body_mass_kg'],height_m=meta['height_m']);b,m,d=rig.b,rig.m,rig.d;chest=m.body('chest_r').id
    foot_ids=[m.body('calcn_'+s).id for s in ['r','l'] for _ in range(4)]
    u=np.c_[z['qvel'][:,:6],z['qvel'][:,rig.dofs]];moments=[];force_err=[];moment_err=[];engine=[];velocity_err=[]
    audit=set(np.unique(np.r_[np.linspace(0,len(z['times'])-1,25).round().astype(int),len(z['times'])-1]))
    for k,(q,v,a) in enumerate(zip(z['qpos'],z['qvel'],z['qacc'])):
        b.pose(q,v,a);J=np.zeros((3,m.nv));Jr=J.copy();Jd=J.copy();Jrd=J.copy()
        mujoco.mj_jac(m,d,J,Jr,d.xpos[chest],chest);mujoco.mj_jacDot(m,d,Jd,Jrd,d.xpos[chest],chest)
        omega=Jr@v;alpha=Jr@a+Jrd@v;R=z['bar_rotation'][k]
        I=R@np.diag([kg*.014**2/2,kg*2.2**2/12,kg*2.2**2/12])@R.T
        T=I@alpha+np.cross(omega,I@omega);moments.append(-T)
        if k in audit:
            contacts=[(bid,z['contact_positions_m'][k,j],z['contact_forces_N'][k,j],np.zeros(3)) for j,bid in enumerate(foot_ids)]
            contacts.append((chest,z['bar_position_m'][k],-z['bar_support_force_N'][k],-T))
            check=sample(rig,q,v,a,contacts);w=check['net_wrenches'].sum(0);C=m.body_mass@d.xipos/m.body_mass.sum()
            force_err.append(abs(w[:3]).max());moment_err.append(abs(w[3:]-np.cross(C,w[:3])).max());engine.append(check['crosscheck_max'])
            B,_=rig.coordinates(q,u[k]);velocity_err.append(abs(B@u[k]-v).max())
    data=dict(times=z['times'],qpos=z['qpos'],qvel=z['qvel'],qacc=z['qacc'],u=u,body_xpos=z['body_xpos'],body_xmat=z['body_xmat'],com=z['com'],velocity=z['velocity'],force_world_N=z['GRF_N'],
              contact_positions_m=np.concatenate([z['contact_positions_m'],z['bar_position_m'][:,None]],axis=1),
              contact_forces_N=np.concatenate([z['contact_forces_N'],-z['bar_support_force_N'][:,None]],axis=1),
              contact_free_moments_Nm=np.concatenate([np.zeros_like(z['contact_forces_N']),np.array(moments)[:,None]],axis=1))
    np.savez_compressed(out/'motion.npz',**data);digest=hashlib.sha256((out/'motion.npz').read_bytes()).hexdigest()
    checks=dict(root_force_max_N=float(max(force_err)),root_moment_max_Nm=float(max(moment_err)),independent_engine_max_SI=float(max(engine)),velocity_reduction_error=float(max(velocity_err)))
    gates=dict(source_squat_checks=bool(meta['all_mechanics_gates_passed']),external_bar_wrench=checks['root_force_max_N']<1e-4 and checks['root_moment_max_Nm']<1e-4,engine=checks['independent_engine_max_SI']<1e-5,reduction=checks['velocity_reduction_error']<1e-5)
    verification=dict(status='checked_prescribed_squat_external_load_adapter',motion_sha256=digest,gates=gates,checks=checks,root_reserve_applied=False)
    (out/'verification.json').write_text(json.dumps(verification,indent=2));assert all(gates.values()),verification
    (out/'motion_quality.json').write_text(json.dumps(dict(status='passed',motion_sha256=digest,task='prescribed loaded squat',source_human_motion=False,source_motion_sha256=hashlib.sha256((src/'motion.npz').read_bytes()).hexdigest()),indent=2))
    setup=dict(rig_type='measured_cmj_standard',movement_kind='squat',body_names=meta['body_names'],mass_kg=meta['body_mass_kg'],height_m=meta['height_m'],bar_mass_kg=kg,
               takeoff_s=None,countermovement_end_s=meta['phase_times_s']['bottom'],heel_rise_start_s=None,ground_reaction_and_COP_are_inputs=False,
               scope='Prescribed loaded squat; full-body internal moments with upper-back bar reaction; no measured recruitment claim.')
    (out/'setup.json').write_text(json.dumps(setup,indent=2))
    points=[dict(body_id=bid,body=m.body(bid).name,role='ground') for bid in foot_ids]+[dict(body_id=chest,body='chest_r',role='bar_reaction')]
    (out/'contact_definition.json').write_text(json.dumps(dict(points=points,bar_force_and_rotational_moment_included=True),indent=2))
    print(json.dumps(verification),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kg',type=float,default=100);p.add_argument('--folder');a=p.parse_args();main(a.kg,a.folder)
