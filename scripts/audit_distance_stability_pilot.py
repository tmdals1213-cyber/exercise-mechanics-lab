"""Audit exact pre-contact replay, defaults, reward use and actual weight changes."""
import copy,json
import numpy as np
from jump_monitor_data import ROOT,read,atomic

def main():
    from controlled_broad_jump import run
    out=ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1';summary=read(out/'pilot_summary.json')
    rows=[json.loads(line)for folder in sorted(out.glob('worker_*'))for line in(folder/'episodes.jsonl').read_text().splitlines()]
    train=[r for r in rows if r['kind']=='train'];assert len(train)==32
    errors=[abs(r['reward']-(r['distance_m']+.8*r['stability']-.8*float(r['fell'])))for r in train if r['eligible']]
    assert max(errors)<1e-12
    assert all(r['reward']>0 for r in train if r['eligible']and r['fell'])
    before=np.load(out/'network_update_0000_before.npz');after=np.load(out/'network_update_0002_after.npz')
    delta=np.linalg.norm(after['action_net.weight']-before['action_net.weight'],axis=1)+abs(after['action_net.bias']-before['action_net.bias'])
    assert len(delta)==58 and np.all(delta>0)
    selected=read(ROOT/'results/controlled_broad_jump/effort_launch_seed_v2/initial_solution.json');source=ROOT/'results/controlled_broad_jump/effort_launch_seed_v2/refinement_trial_00190/dt_500us'
    original=np.load(source/'control_motion.npz');fall=np.load(out/'long_distance_fall/control_motion.npz');n=len(original['times'])
    # The original trajectory contains a forced terminal sample at touchdown,
    # so compare only the common regular sample grid.
    common=np.intersect1d(original['times'],fall['times']);a=np.searchsorted(original['times'],common);b=np.searchsorted(fall['times'],common)
    exact=float(abs(original['qpos'][a]-fall['qpos'][b]).max());assert exact<1e-12
    legacy=ROOT/'results/controlled_broad_jump/overnight_review_2026_11_25/best_500us';ref=read(legacy/'control_summary.json');p=copy.deepcopy(ref['parameters']);report,z=run(p,dt=ref['physics_dt_s']);prior=np.load(legacy/'control_motion.npz')
    legacy_q=float(abs(z['qpos']-prior['qpos']).max());legacy_F=float(abs(z['physics_force_world_N']-prior['physics_force_world_N']).max());assert legacy_q==0 and legacy_F==0
    result=dict(status='passed',training_episodes=len(train),fallen_training_episodes=sum(r['fell']for r in train),reward_formula_max_error=max(errors),all_58_mean_outputs_changed=True,mean_output_delta_min=float(delta.min()),actual_update_weight_change_L2=summary['real_weight_change_each_update'],common_precontact_qpos_max_error=exact,legacy_default_off_qpos_max_error=legacy_q,legacy_default_off_GRF_max_error_N=legacy_F,body_floor_contact_recorded=read(out/'long_distance_fall/control_summary.json')['first_body_floor_contact_s'],failed_distance_retained=True,no_convergence_claim=True)
    atomic(out/'audit.json',result);print('DISTANCE_STABILITY_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
