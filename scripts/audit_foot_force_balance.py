"""Reconstruct the foot free-body diagram against saved joint reactions.

The saved reactions were generated separately by the visualization pipeline.
This checks consistency of the ideal-joint model, not anatomical validity.
"""
import json,numpy as np
from jump_rig import ROOT
from jump_model import create_rig
from mechanics_engine_adapter import sample
from muscle_mechanics import tendon_body_wrenches,subtree

def main():
    p=ROOT/'results/contact_jump_scapula/600hz';out=ROOT/'results/jump_contact_visualization'
    motion=np.load(p/'motion.npz');t=np.load(p/'tensions.npz');s=json.loads((p/'tension_summary.json').read_text());setup=json.loads((p/'setup.json').read_text());points=json.loads((p/'contact_definition.json').read_text())['points'];data=json.loads((out/'data.json').read_text());saved=np.load(out/'joint_reactions.npz');rig=create_rig(setup['rig_type']);b,m,d=rig.b,rig.m,rig.d;rows=[]
    assert np.max(abs(saved['times']-t['times']))<1e-9
    for side in ['r','l']:
        calf=[i for i,n in enumerate(s['names'][:m.nu]) if n.startswith(('soleus','gasmed','gaslat')) and n.endswith('_'+side)];k=int(np.argmax(t['force_N'][:,calf].sum(1)));index=int(t['source_indices'][k]);contacts=[(point['body_id'],motion['contact_positions_m'][index,j],motion['contact_forces_N'][index,j],np.zeros(3)) for j,point in enumerate(points)];mech=sample(rig,motion['qpos'][index],motion['qvel'][index],motion['qacc'][index],contacts)
        muscle=tendon_body_wrenches(b,t['force_N'][k,:m.nu]);only=np.zeros(m.nu);only[calf]=t['force_N'][k,calf];calfw=tendon_body_wrenches(b,only)
        jid=b.joints['ankle_angle_'+side];bodies=subtree(m,int(m.jnt_bodyid[jid]));ids=[j for j,point in enumerate(points) if point['body_id'] in bodies];jindex=next(i for i,j in enumerate(data['joints']) if j['key']=='ankle_'+side);R=saved['joint_wrenches'][k,jindex,:3];G=m.body_mass[bodies,None]*m.opt.gravity;inertial=(m.body_mass[bodies,None]*mech['acceleration'][bodies]).sum(0);grf=motion['contact_forces_N'][index,ids].sum(0);MF=muscle[bodies,:3].sum(0);CF=calfw[bodies,:3].sum(0);weight=G.sum(0);residual=R+MF+grf+weight-inertial
        rows.append(dict(side=side,time_s=float(t['times'][k]),foot_bodies=[m.body(i).name for i in bodies],frame='world XYZ; +Z up; all vectors act ON the complete foot',ground_force_N=grf.tolist(),weight_N=weight.tolist(),calf_force_vector_N=CF.tolist(),other_muscle_force_vector_N=(MF-CF).tolist(),total_muscle_force_vector_N=MF.tolist(),proximal_ankle_reaction_on_foot_N=R.tolist(),proximal_reaction_magnitude_N=float(np.linalg.norm(R)),mass_times_COM_acceleration_N=inertial.tolist(),force_balance_residual_N=residual.tolist(),force_balance_error_N=float(np.linalg.norm(residual)),joint_reaction_free_hinge_moment_Nm=float(saved['joint_wrenches'][k,jindex,3:]@d.xaxis[jid]),calf_scalar_tension_sum_N=float(t['force_N'][k,calf].sum()),native_all_body_internal_force_sum_N=muscle[:,:3].sum(0).tolist()))
    assert max(r['force_balance_error_N'] for r in rows)<1e-5
    assert max(abs(r['joint_reaction_free_hinge_moment_Nm']) for r in rows)<1e-5
    result=dict(status='passed',equation='ankle reaction + muscle forces + GRF + gravity = sum(m_i * a_i)',reaction_source='Previously saved cut reactions, not fitted anew in this audit',limitations='An unrestricted ideal joint can supply large reactions. Closure alone is not a contact-pressure, tissue-strength or recruitment validation.',audits=rows)
    (out/'foot_force_balance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
