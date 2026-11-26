"""Initialize an untrained Gaussian policy and verify zero-action native replay."""
from pathlib import Path
import json,subprocess,sys
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from recover_jump_ppo_batch import SetupOnlyEnv
from jump_monitor_data import ROOT,read,atomic
from effort_launch_objective import INITIAL_CONTACT_ENVELOPE

def main():
    folder=ROOT/'results/controlled_broad_jump/effort_launch_seed_v2';file=folder/'policy_design.json';design=read(file);assert design is not None
    refined=read(folder/'initial_solution.json')
    design.update(initial_contact_envelope=INITIAL_CONTACT_ENVELOPE,major_variable_scope='Local normalized finite perturbations around one initial solution; not global biomechanical importance.',reference_distances_by_dt_s={format(r['dt_s'],'g'):r['heel_projection_at_first_foot_contact_m']for r in refined['cases']},excluded_postcontact_variables=['landing_hip','landing_knee','landing_transition_s','landing_ankle','landing_trunk','landing_shoulder'])
    design['training_proposal'].update(native_worker_dt_s=[.0005,.00025,.00025],reason_for_mixed_dt='Avoid optimizing only one numerical/contact-event sampling phase; evaluate125us separately.')
    atomic(file,design)
    out=ROOT/'deliverables/standing_long_jump_effort_seed';out.mkdir(parents=True,exist_ok=True);obs=np.array(design['observation'],np.float32);n=len(design['actions']);assert n==52
    torch.set_num_threads(1);torch.manual_seed(1125);env=DummyVecEnv([lambda:SetupOnlyEnv(obs,n)])
    model=PPO('MlpPolicy',env,learning_rate=.00001,n_steps=32,batch_size=32,n_epochs=4,gamma=1.,gae_lambda=1.,clip_range=.1,ent_coef=.0001,target_kl=.01,max_grad_norm=.5,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,log_std_init=float(np.log(.03))),device='cpu',seed=1125,verbose=0)
    with torch.no_grad():
        model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_();model.policy.log_std[:]=torch.as_tensor(np.log(design['initial_normalized_std']),dtype=torch.float32)
    action,_=model.predict(obs,deterministic=True);assert np.array_equal(action,np.zeros(n,np.float32));assert all(p.requires_grad for p in model.policy.parameters());model.save(out/'effort_launch_policy_initial')
    native=folder/'policy_zero_action_replay';native.mkdir(exist_ok=True)
    with(native/'worker.log').open('w',encoding='utf-8')as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'scripts/effort_launch_policy_worker.py'),'--definition',str(file),'--out',str(native)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        ready=process.stdout.readline();assert ready.startswith('EFFORT_WORKER_READY '),ready
        process.stdin.write(json.dumps(dict(cmd='rollout',kind='policy_initialization_check',action=action.tolist()))+'\n');process.stdin.flush();line=process.stdout.readline();assert line.startswith('EFFORT_RESULT '),line;result=json.loads(line.split(' ',1)[1]);process.stdin.write('{"cmd":"close"}\n');process.stdin.flush();process.wait(timeout=30)
    reference=read(folder/'initial_solution.json')['cases'][0]['heel_projection_at_first_foot_contact_m'];error=abs(result['distance_m']-reference);assert result['eligible']and error<1e-8
    report=dict(status='initialized_untrained',actor=[7,64,64,n],critic=[7,64,64,1],training_episodes=0,initialization_test_rollouts=1,all_outputs_and_standard_deviations_trainable=True,zero_action_native_distance_m=result['distance_m'],zero_action_native_replay_error_m=error,major_posture_keys=[v['key']for v in design['major_posture_variables']],equivalent_effort_outputs=15,posture_timing_outputs=37,native_worker='scripts/effort_launch_policy_worker.py',policy_definition=str(file),checkpoint=str(out/'effort_launch_policy_initial.zip'),recovery_reward_separate=True,individual_muscle_tension=False)
    atomic(out/'policy_initialization.json',report);env.close();print('EFFORT_POLICY_INITIALIZED',result['distance_m'],'exact replay',error,'training episodes 0',flush=True)

if __name__=='__main__':main()
