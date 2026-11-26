"""Checkpointed recovery initialization then genuine conditional PPO.

Lightweight attempt histories remain; raw failures have bounded retention. Context mixes a working stable
anchor with the long falling anchor; torque capacities/contact physics are fixed.
"""
from pathlib import Path
import argparse,contextlib,copy,json,os,shutil,subprocess,sys,threading,time,traceback,hashlib
import numpy as np
from jump_monitor_data import ROOT,read,atomic
from landing_learning_objective import VERSION,TARGET_FIRST,TARGET_RETAINED

OUT=ROOT/'results/controlled_broad_jump/landing_stability_v2'
WEB=ROOT/'deliverables/jump_landing_live'
RECOVERY_KEYS={'flight_hip','flight_knee','flight_ankle','flight_shoulder','flight_trunk','flight_transition_s','prepare_landing_s','prepare_hip','prepare_knee','prepare_shoulder','prepare_trunk','prepare_transition_s','landing_hip','landing_knee','landing_ankle','landing_trunk','landing_shoulder','landing_transition_s'}

def qualified_completion(cases,completion,passed):
    """No rounded display value or incomplete timestep set can trigger stop."""
    if not passed or not completion:return False
    required=completion.get('required_timesteps_s',[.0005,.00025,.000125])
    return bool(len(cases)==len(required)and {r['dt_s']for r in cases}==set(required)and all(r['stable']and r['eligible']and r['retained_distance_m']>=float(completion['min_retained_m'])for r in cases))

def definition():
    old=read(ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1/definition.json');stable=read(ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1/stable_reference/control_summary.json')['parameters'];long=read(ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1/pilot_summary.json')['final_evaluation']['parameters']
    for p in [stable,long]:
        p.pop('stop_after_touchdown_s',None);p.update(fall_contact_model=True,fall_followthrough_s=.55,recovery_evaluation_s=2.5,seconds=7.,record_energy_ledger=False)
        p.setdefault('propulsion_effort_knots_fraction',{key:[0.,0.,0.]for key in ['hip_flexion','knee_angle','ankle_angle','flex_extension','shoulder_elv']})
    assert stable['joint_torque_limits_Nm']==long['joint_torque_limits_Nm']
    items=copy.deepcopy(old['actions'])
    # ROM bounds come from the actual native registration, already stored in
    # the baseline's model. The coordinator does not import MuJoCo with torch.
    for item in items:
        if item['kind']=='equivalent_effort':item.update(minimum=-.4,maximum=.4,half_span=.4);continue
        key=item['key'];span=max(item['maximum']-item['center'],item['center']-item['minimum'])
        if key in RECOVERY_KEYS:span=max(span*3.,.12 if item['unit']=='s'else .35)
        elif key=='push_s':span=.07
        else:span*=1.8
        item['half_span']=span
    return dict(version=VERSION,stable_anchor=stable,long_anchor=long,actions=items,physical_observation=old['observation'],observation_count=9,policy_outputs=58,goal_first_contact_m=TARGET_FIRST,goal_retained_contact_m=TARGET_RETAINED,dt_s=.0005,torque_profile_fixed=True,first_stage='18 recovery coordinates, launch held unchanged; population search for initialization, not PPO.',main_stage='All58 outputs and std learned jointly by actual PPO; observation includes anchor blend and target distance.',reward='D + 2*S + .75*stable - .75*fall - .75*target_shortfall - .75*retreat',recovery_score='S=.30*survival+.25*height+.20*quiet+.15*capture_alignment+.10*foot_support. Additive rather than vanishing product; falling trials evaluated before body-floor impact.',all_failures_saved=True,individual_muscle_force_solver=False)

def map_items(config,blend):
    stable=config['stable_anchor'];long=config['long_anchor'];base=copy.deepcopy(stable)
    for key,value in long.items():
        if isinstance(value,(int,float))and not isinstance(value,bool)and isinstance(stable.get(key),(int,float)):base[key]=float((1-blend)*stable[key]+blend*value)
    base['propulsion_effort_knots_fraction']={key:[float((1-blend)*a+blend*b)for a,b in zip(stable['propulsion_effort_knots_fraction'][key],value)]for key,value in long['propulsion_effort_knots_fraction'].items()}
    items=[]
    for template in config['actions']:
        item=template.copy();key=item['key'];span=item['half_span']
        if item['kind']=='equivalent_effort':item['center']=base['propulsion_effort_knots_fraction'][item['stem']][item['knot']]
        else:
            c=float(base[key]);item['center']=c;lo=c-span;hi=c+span
            if item['unit']=='rad'and not key.startswith('pitch_'):
                stem='hip_flexion'if'hip'in key else'knee_angle'if'knee'in key else'ankle_angle'if'ankle'in key else'shoulder_elv'if'shoulder'in key else'elbow_flexion'if key=='elbow'else'flex_extension'if'trunk'in key else None
                if stem:
                    a,b=config['joint_ranges_rad'][stem];lo=max(lo,a);hi=min(hi,b)
            elif item['unit']=='fraction':lo=max(0.,lo);hi=min(.8,hi)
            elif item['unit']=='duration_fraction':lo=max(.25,lo);hi=min(1.3,hi)
            elif item['unit']=='s'and key!='arm_delay_s':lo=max(0. if key=='prepush_lean_s'else .04,lo)
            item.update(minimum=lo,maximum=hi)
            assert lo<=c<=hi,(key,lo,c,hi)
        items.append(item)
    return base,items

def encode_parameters(config,parameters,blend=1.):
    base,items=map_items(config,blend);action=[]
    for item in items:
        value=parameters[item['key']]if item['kind']=='posture_timing'else parameters['propulsion_effort_knots_fraction'][item['stem']][item['knot']]
        centre=item['center'];delta=value-centre;span=item['maximum']-centre if delta>=0 else centre-item['minimum'];x=delta/span if abs(delta)>1e-12 else 0.
        assert -1.000001<=x<=1.000001,(item['key'],value,item);action.append(float(np.clip(x,-1,1)))
    return np.array(action,np.float32)

class Experiment:
    def __init__(self,config,workers):
        self.config=config;self.processes=[];self.logs=[];self.current={};self.lock=threading.RLock();self.closed=False;self.started=time.time();self.state=read(OUT/'live.json',{})
        self.state.update(status='starting',stage='native_workers',run_label=OUT.name,controller_profile=config.get('controller_profile','legacy_combined_loops'),inherited_training_episodes=config.get('inherited_training_episodes',0),started_at=self.started,heartbeat=self.started,objective_version=VERSION,training_episodes=self.state.get('training_episodes',0),policy_updates=self.state.get('policy_updates',0),native_evaluations=self.state.get('native_evaluations',0),target_first_m=TARGET_FIRST,target_retained_m=TARGET_RETAINED,model=dict(height_cm=178,mass_kg=87,outputs=58,observations=9,actor=[9,64,64,58],fixed_net_joint_torque_limits=True,body_floor_proxies=True,individual_muscle_forces=False),best_stable=self.state.get('best_stable'),best_distance=self.state.get('best_distance'),verified=self.state.get('verified'),recent=[],history=self.state.get('history',[]),pareto=self.state.get('pareto',[]),updates=self.state.get('updates',[]),all_failures_retained=True)
        self.state.setdefault('stable_candidates',[]);self.state.setdefault('tested_candidates',{})
        self.state.update(all_failures_retained=False,all_lightweight_outcomes_retained=True,raw_failure_retention='Newest24 per worker plus protected video/validation cases')
        self.state['training_precision_mixture_s']=[.0005,.00025,.00025]
        # Preserve qualification attempts, including rejected coarse winners.
        for file in sorted(OUT.glob('validation_*.json'),key=lambda f:f.stat().st_mtime):
            value=read(file)
            required=config.get('completion_rule',{}).get('required_timesteps_s',[.0005,.00025,.000125])
            if value and value.get('cases')and {r['dt_s']for r in value['cases']}==set(required):
                source=value['cases'][0];key=self.candidate_key(source['parameters']);self.state['tested_candidates'][key]=dict(passed=value['passed'],id=source['id'])
                if source['id']==(self.state.get('best_stable')or{}).get('id'):self.state['best_stable']['validation_passed']=value['passed']
        for small in self.state['history']:
            if small.get('stable')and small.get('eligible')and small['kind']in ['ppo_train','policy_mean_eval','stable_anchor']:
                worker,episode=small['id'].split('_episode_');worker='worker_'+worker.rsplit('worker_',1)[-1];source=read(OUT/worker/('episode_'+episode)/'score.json')
                if source:self.queue_candidate(source)
        self.thread=threading.Thread(target=self.publish_loop,daemon=True);self.thread.start()
        for i in range(workers):
            log=(OUT/f'worker_{i:02d}.log').open('a',encoding='utf-8');self.logs.append(log)
            p=subprocess.Popen([sys.executable,str(ROOT/'scripts/landing_learning_worker.py'),'--run',str(OUT),'--worker',str(i)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.processes.append(p)
        for i in range(workers):self.receive(i,'LANDING_READY')
        self.state['worker_pids']=[p.pid for p in self.processes];self.state['coordinator_pid']=os.getpid()
    def publish_loop(self):
        while not self.closed:
            try:
                with self.lock:
                    self.state['heartbeat']=time.time();self.state['workers']=[dict(worker=i,**v,elapsed_s=time.time()-v['started_at'])for i,v in self.current.items()];atomic(OUT/'live.json',self.state);atomic(WEB/'status.json',self.state)
            except Exception as exc:print('PUBLISH_ERROR',repr(exc),flush=True)
            time.sleep(2)
    @staticmethod
    def candidate_key(parameters):return hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()[:20]
    def queue_candidate(self,row):
        key=self.candidate_key(row['parameters'])
        if key in self.state['tested_candidates']:return
        candidates=self.state['stable_candidates']
        if not any(c['key']==key for c in candidates):candidates.append(dict(key=key,id=row['id'],folder=row['folder'],retained_distance_m=row['retained_distance_m'],dt_s=row['dt_s']))
        completion=self.config.get('completion_rule');target=float(completion['min_retained_m'])if completion else float('inf')
        candidates.sort(key=lambda c:(c['retained_distance_m']>=target,c['dt_s']<.0005,c['retained_distance_m']),reverse=True);self.state['stable_candidates']=candidates[:30]
    def send(self,i,command):
        self.current[i]=dict(kind=command.get('kind'),context=command.get('context'),started_at=time.time())
        self.processes[i].stdin.write(json.dumps(command)+'\n');self.processes[i].stdin.flush()
    def receive(self,i,prefix='LANDING_RESULT'):
        while True:
            line=self.processes[i].stdout.readline()
            if not line:raise RuntimeError(f'Native worker{i} exited {self.processes[i].poll()}; see worker log')
            if line.startswith('LANDING_ERROR '):raise RuntimeError(line)
            if line.startswith(prefix+' '):
                self.current.pop(i,None);return json.loads(line[len(prefix)+1:])
    def batch(self,commands):
        for i,c in enumerate(commands):self.send(i,c)
        rows=[self.receive(i)for i in range(len(commands))]
        for row in rows:self.accept(row)
        return rows
    def accept(self,row):
        fields=['id','kind','context','distance_m','retained_distance_m','stability','stable','fell','reward','eligible','goal_met','finished_at','body_floor_contact_s','survival_fraction','late_COM_speed_m_s','compute_seconds','dt_s','objective_version']
        small={key:row.get(key)for key in fields}
        with self.lock:
            self.state['native_evaluations']+=1;self.state['recent']=(self.state['recent']+[small])[-50:];self.state['history']=(self.state['history']+[small])[-1200:];self.state['latest']=small
            with(OUT/'history.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(small)+'\n')
            shutil.copyfile(Path(row['folder'])/'motion.json',WEB/'latest_motion.pending.json');os.replace(WEB/'latest_motion.pending.json',WEB/'latest_motion.json')
            if row['eligible']:
                if row['stable']and row['kind']in ['ppo_train','policy_mean_eval','stable_anchor','recovery_initialization']:self.queue_candidate(row)
                best=self.state.get('best_distance')
                if best is None or row['distance_m']>best['distance_m']:
                    self.state['best_distance']=small;atomic(OUT/'best_distance.json',row);shutil.copyfile(Path(row['folder'])/'motion.json',WEB/'best_distance_motion.pending.json');os.replace(WEB/'best_distance_motion.pending.json',WEB/'best_distance_motion.json')
                frontier=self.state['pareto']
                if not any(q['distance_m']>=row['distance_m']and q['stability']>=row['stability']for q in frontier):
                    frontier=[q for q in frontier if not(row['distance_m']>=q['distance_m']and row['stability']>=q['stability'])];frontier.append(small);self.state['pareto']=frontier
                best=self.state.get('best_stable')
                if row['stable']and(best is None or row['retained_distance_m']>best['retained_distance_m']):
                    self.state['best_stable']=small;atomic(OUT/'best_stable.json',row);shutil.copyfile(Path(row['folder'])/'motion.json',WEB/'best_stable_motion.pending.json');os.replace(WEB/'best_stable_motion.pending.json',WEB/'best_stable_motion.json');print('NEW_STABLE',row['distance_m'],row['retained_distance_m'],row['stability'],flush=True)
    def validate(self,row):
        oldstage=self.state['stage'];self.state['stage']='fine_timestep_validation'
        timesteps=self.config.get('completion_rule',{}).get('required_timesteps_s',[.0005,.00025,.000125])
        commands=[dict(cmd='rollout',base=row['parameters'],dt=dt,kind='fine_validation',context=row.get('context'))for dt in timesteps if abs(dt-row['dt_s'])>1e-10]
        refined=self.batch(commands);cases=[row]+refined;distances=[r['retained_distance_m']for r in cases]
        spread=(max(distances)-min(distances))/max(distances)if all(v is not None and np.isfinite(v)and v>0 for v in distances)else None
        passed=bool(all(r['stable']and r['eligible']for r in cases)and spread is not None and spread<.05)
        completion=self.config.get('completion_rule');goal_met=qualified_completion(cases,completion,passed)if completion else bool(passed and min(r['distance_m']for r in cases)>=TARGET_FIRST and min(distances)>=TARGET_RETAINED)
        report=dict(passed=passed,cases=cases,relative_distance_spread=spread,goal_met=goal_met,completion_rule=completion)
        atomic(OUT/f"validation_{row['id']}_{int(time.time())}.json",report)
        key=self.candidate_key(row['parameters']);self.state['tested_candidates'][key]=dict(passed=passed,id=row['id']);self.state['stable_candidates']=[c for c in self.state['stable_candidates']if c['key']!=key]
        if row['id']==(self.state.get('best_stable')or{}).get('id'):self.state['best_stable']['validation_passed']=passed
        if passed:
            value={key:row.get(key)for key in ['id','distance_m','retained_distance_m','stability','stable','fell']};value.update(motion_id=refined[-1]['id'],min_fine_first_m=min(r['distance_m']for r in cases),min_fine_retained_m=min(distances),goal_met=report['goal_met'],timesteps_s=timesteps)
            previous=self.state.get('verified')or{};same_protocol=set(previous.get('timesteps_s',[]))==set(timesteps)
            if not same_protocol or value['min_fine_retained_m']>previous.get('min_fine_retained_m',0.):
                self.state['verified']=value;atomic(OUT/'verified_best.json',report);shutil.copyfile(Path(refined[-1]['folder'])/'motion.json',WEB/'verified_motion.pending.json');os.replace(WEB/'verified_motion.pending.json',WEB/'verified_motion.json')
        self.state['stage']=oldstage
        print('FINE_VALIDATION',row['id'],passed,report['goal_met'],flush=True);return report
    def close(self):
        for p in self.processes:
            if p.poll()is None:
                with contextlib.suppress(BrokenPipeError,OSError):p.stdin.write('{"cmd":"close"}\n');p.stdin.flush()
        for p in self.processes:
            with contextlib.suppress(subprocess.TimeoutExpired):p.wait(timeout=30)
        self.closed=True;self.state['heartbeat']=time.time();atomic(OUT/'live.json',self.state);atomic(WEB/'status.json',self.state)
        for log in self.logs:log.close()

def recovery_initialization(exp,args):
    if exp.config.get('initialization_complete'):return
    exp.state.update(status='running',stage='recovery_initialization',stage_note='공중 자세·착지 18변수의 초기 탐색. CEM이며 PPO 학습 횟수와 구분합니다.')
    anchors=exp.batch([dict(cmd='rollout',base=exp.config['stable_anchor'],kind='stable_anchor'),dict(cmd='rollout',base=exp.config['long_anchor'],kind='long_anchor')]);exp.validate(anchors[0])
    base,items=map_items(exp.config,1.);active=[i for i,x in enumerate(items)if x['key']in RECOVERY_KEYS];assert len(active)==18
    mean=np.zeros(len(active));std=np.full(len(active),.35);rng=np.random.default_rng(112502);best=anchors[1]
    for gen in range(args.generations):
        population=np.clip(rng.normal(mean,std,(args.population,len(active))),-1,1);population[0]=mean
        if gen==0:population[0]=0.
        actions=np.zeros((len(population),58));actions[:,active]=population;rows=[]
        for start in range(0,len(actions),len(exp.processes)):
            commands=[dict(cmd='rollout',base=base,items=items,action=action.tolist(),kind='recovery_initialization',cache_launch=True,cache_id='fixed_long_launch',context=1.)for action in actions[start:start+len(exp.processes)]]
            batch=exp.batch(commands);rows.extend(batch)
            for row in batch:
                if row['eligible']and row['reward']>best['reward']:best=row
        ranked=sorted(range(len(rows)),key=lambda i:rows[i]['reward'],reverse=True);elite=population[ranked[:max(3,len(rows)//4)]];mean=.25*mean+.75*elite.mean(0);std=np.maximum(.08,.40*std+.60*elite.std(0))
        exp.state['initialization']=dict(generation=gen+1,generations=args.generations,active_variables=18,best_distance_m=best['distance_m'],best_stability=best['stability'],best_fell=best['fell'],best_reward=best['reward']);atomic(OUT/'recovery_initialization_best.json',best)
        print('RECOVERY_GENERATION',gen+1,best['distance_m'],best['stability'],best['fell'],flush=True)
    # Both endpoints remain available. A short stable initializer does not
    # overwrite the long-distance record or the fixed-distance target.
    exp.config['long_anchor']=best['parameters'];exp.config['initialization_complete']=True;atomic(OUT/'definition.json',exp.config)

def train(exp,args,deadline):
    import torch,gymnasium as gym
    from stable_baselines3.common.vec_env import VecEnv
    from train_jump_overnight import InstrumentedPPO,capture,save_policy
    config=exp.config;rng=np.random.default_rng(112503);focus=bool(args.reseed_parameters or config.get('focus_context_one',False))
    class Environment(VecEnv):
        def __init__(self):
            self.contexts=np.ones(len(exp.processes))if focus else np.zeros(len(exp.processes))
            if not focus:self.contexts[1:]=np.linspace(.45,1.,len(exp.processes)-1)
            self.pending=[]
            super().__init__(len(exp.processes),gym.spaces.Box(-5,5,shape=(9,),dtype=np.float32),gym.spaces.Box(-1,1,shape=(58,),dtype=np.float32))
        def observation(self,i):return np.array(config['physical_observation']+[float(self.contexts[i]),TARGET_FIRST/3.],np.float32)
        def reset(self):return np.array([self.observation(i)for i in range(self.num_envs)])
        def step_async(self,actions):
            self.pending=[]
            for i,action in enumerate(actions):
                base,items=map_items(config,float(self.contexts[i]));command=dict(cmd='rollout',base=base,items=items,action=action.tolist(),dt=[.0005,.00025,.00025][i],context=float(self.contexts[i]),kind='ppo_train');exp.send(i,command);self.pending.append(self.observation(i))
        def step_wait(self):
            rows=[exp.receive(i)for i in range(self.num_envs)];info=[]
            for i,row in enumerate(rows):
                exp.accept(row);item={key:row.get(key)for key in ['id','distance_m','retained_distance_m','stability','stable','fell','reward','eligible']};item['terminal_observation']=self.pending[i];info.append(item)
                # Stable, long and bridge branches all remain present.
                self.contexts[i]=1. if focus else float(rng.choice([0.,.25,.5,.75,1.],p=[.40,.10,.15,.10,.25]))
            exp.state['training_episodes']+=len(rows)
            return self.reset(),np.array([r['reward']for r in rows],np.float32),np.ones(self.num_envs,bool),info
        def close(self):pass
        def get_attr(self,attr_name,indices=None):return[getattr(self,attr_name,None)for _ in self._get_indices(indices)]
        def set_attr(self,attr_name,value,indices=None):setattr(self,attr_name,value)
        def env_method(self,method_name,*args,indices=None,**kwargs):return[None for _ in self._get_indices(indices)]
        def env_is_wrapped(self,wrapper_class,indices=None):return[False for _ in self._get_indices(indices)]
    torch.set_num_threads(1);env=Environment();latest=OUT/'policy_latest.zip'
    if args.resume and latest.exists():model=InstrumentedPPO.load(latest,env=env,device='cpu')
    elif args.warm_start:
        model=InstrumentedPPO.load(args.warm_start,env=env,device='cpu');exp.state['inherited_training_episodes']=int(model.num_timesteps);model.num_timesteps=0;save_policy(model,OUT/'policy_initial');save_policy(model,latest)
    else:
        model=InstrumentedPPO('MlpPolicy',env,learning_rate=1e-4,n_steps=16,batch_size=48,n_epochs=6,gamma=1.,gae_lambda=1.,clip_range=.15,ent_coef=.0003,target_kl=.02,max_grad_norm=.5,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,log_std_init=float(np.log(.10))),device='cpu',seed=112503,verbose=0)
        # Warm-start compatible layers from the previous actual58-output PPO.
        from stable_baselines3 import PPO
        prior=PPO.load(ROOT/'results/controlled_broad_jump/distance_stability_pilot_v1/policy_update_02.zip',device='cpu');old=prior.policy.state_dict();new=model.policy.state_dict()
        with torch.no_grad():
            for key,value in new.items():
                if key in old and value.shape==old[key].shape:value.copy_(old[key])
                elif key in old and value.ndim==2 and value.shape[1]==9 and old[key].shape[1]==7:value.zero_();value[:,:7].copy_(old[key])
            model.policy.load_state_dict(new);model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_();model.policy.log_std[:]=float(np.log(.10));model.policy.log_std[52:]=float(np.log(.18))
        save_policy(model,OUT/'policy_initial');save_policy(model,latest)
    if args.reseed_parameters:
        from effort_launch_parameters import decode
        seed=read(args.reseed_parameters);parameters=seed.get('parameters',seed);target_action=encode_parameters(config,parameters);base,items=map_items(config,1.);reconstructed=decode(base,items,target_action)
        assert reconstructed['joint_torque_limits_Nm']==parameters['joint_torque_limits_Nm'];save_policy(model,OUT/f'policy_before_recenter_{int(model.num_timesteps):06d}')
        recenter_std=config.get('recenter_std_groups',dict(posture_timing=.12,effort=.20,postcontact=.045))
        assert all(0.<float(recenter_std[k])<=.35 for k in ['posture_timing','effort','postcontact'])
        with torch.no_grad():
            model.policy.action_net.weight.zero_();model.policy.action_net.bias.copy_(torch.as_tensor(target_action));model.policy.log_std[:37]=float(np.log(recenter_std['posture_timing']));model.policy.log_std[37:52]=float(np.log(recenter_std['effort']));model.policy.log_std[52:]=float(np.log(recenter_std['postcontact']))
        for parameter in [model.policy.action_net.weight,model.policy.action_net.bias,model.policy.log_std]:model.policy.optimizer.state.pop(parameter,None)
        model.learning_rate=2e-4;model.lr_schedule=lambda _:2e-4
        config.update(focus_context_one=True,adaptive_exploration=True,optimizer_protocol='best_verified_recentered_ppo_v1');atomic(OUT/'definition.json',config)
        record=dict(at=time.time(),episode=int(model.num_timesteps),source=str(args.reseed_parameters),source_algorithm=args.reseed_source_algorithm,action=target_action.tolist(),same_58_actions=True,absolute_bounds_unchanged=True,strength_capacities_unchanged=True,physics_and_controller_unchanged=True,action_mean_reinitialized=True,hidden_and_value_networks_inherited=True,initial_std_groups=recenter_std,learning_rate=2e-4)
        atomic(OUT/f'recenter_{int(model.num_timesteps):06d}.json',record);exp.state['optimizer_recenter']=record;save_policy(model,OUT/f'policy_recentered_{int(model.num_timesteps):06d}');save_policy(model,latest)
        exp.state.update(status='running',stage='recenter_integration_check',stage_note='같은58개 변수·상한으로 찾은 최고해에 탐색 분포를 옮기고 실제 재생을 확인합니다.')
        row=exp.batch([dict(cmd='rollout',base=base,items=items,action=target_action.tolist(),dt=.00025,context=1.,kind='policy_recenter_eval')])[0];assert row['eligible']and row['stable'],'Recentered policy mean did not reproduce a stable candidate.'
        if row['retained_distance_m']>(exp.state.get('verified')or{}).get('min_fine_retained_m',0.):exp.validate(row)
    exp.state.update(status='running',stage='ppo_training',stage_note='최고 검증해 중심으로 같은58개 변수를 탐색합니다. 추진력 탐색과 착지 자세 보존의 폭을 분리했습니다.'if focus else'58개 전체 변수를 실제 PPO로 공동 학습. 실패 동작도 모두 보존합니다.',deadline=deadline,planned_training_episodes=args.episodes,policy_architecture=[9,64,64,58],continue_after_goal=bool(args.continue_after_goal),completion_rule=config.get('completion_rule'))
    exp.state['focus_context_one']=focus
    capture(model,np.array(config['physical_observation']+[1.,TARGET_FIRST/3.],np.float32),OUT,int(exp.state['policy_updates']),'resume'if args.resume else'before')
    update=int(exp.state['policy_updates'])
    while int(model.num_timesteps)+48<=args.episodes and time.time()<deadline and not(OUT/'stop.request').exists():
        if config.get('adaptive_exploration'):
            save_policy(model,OUT/f'policy_sampling_before_update_{update+1:04d}');capture(model,np.array(config['physical_observation']+[1.,TARGET_FIRST/3.],np.float32),OUT,update+1,'before')
        model.learn(total_timesteps=48,reset_num_timesteps=False);update+=1;save_policy(model,latest);save_policy(model,OUT/f'policy_update_{update:04d}')
        capture(model,np.array(config['physical_observation']+[1.,TARGET_FIRST/3.],np.float32),OUT,update,'after');buf=model.rollout_buffer
        np.savez_compressed(OUT/f'rollout_batch_update_{update:04d}.npz',**{key:getattr(buf,key)for key in ['observations','actions','rewards','episode_starts','values','log_probs','advantages','returns']})
        record=dict(update=update,training_episodes=int(model.num_timesteps),optimizer_epochs=int(model._n_updates),weight_change_L2=model.last_weight_delta,at=time.time());exp.state['policy_updates']=update;exp.state['updates']=(exp.state['updates']+[record])[-200:]
        with(OUT/'updates.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(record)+'\n')
        print('LANDING_PPO_UPDATE',update,model.num_timesteps,model.last_weight_delta,flush=True)
        if config.get('adaptive_exploration'):
            recent=[r for r in exp.state['history']if r['kind']=='ppo_train'][-48:];fraction=float(np.mean([r['stable']and r['eligible']for r in recent]));factor=.90 if fraction<.15 else 1.12 if fraction>.60 else 1.
            if factor!=1.:
                with torch.no_grad():
                    floors=config.get('exploration_min_std_groups',dict(posture_timing=.04,effort=.06));std=model.policy.log_std.exp();std[:37]=torch.clamp(std[:37]*factor,floors['posture_timing'],.30);std[37:52]=torch.clamp(std[37:52]*factor,floors['effort'],.35);model.policy.log_std.copy_(std.log())
                save_policy(model,latest)
            exp.state['exploration']=dict(stable_fraction=fraction,rescale_factor=factor,groups_std=[float(model.policy.log_std.exp()[a:b].mean().detach())for a,b in [(0,37),(37,52),(52,58)]],manual_schedule_not_counted_as_PPO_update=True)
            atomic(OUT/f'exploration_after_update_{update:04d}.json',exp.state['exploration'])
        if update%2==0:
            commands=[]
            for context in([1.]if focus else[0.,.5,1.]):
                obs=np.array(config['physical_observation']+[context,TARGET_FIRST/3.],np.float32);action=model.predict(obs,deterministic=True)[0];base,items=map_items(config,context);commands.append(dict(cmd='rollout',base=base,items=items,action=action.tolist(),dt=.00025,context=context,kind='policy_mean_eval'))
            exp.batch(commands)
        reference=(exp.state.get('verified')or{}).get('min_fine_retained_m',1.9)
        threshold=min(reference+.02,float(config['completion_rule']['min_retained_m']))if config.get('completion_rule')else reference+.02
        candidates=[c for c in exp.state['stable_candidates']if c['key']not in exp.state['tested_candidates']and c['retained_distance_m']>=threshold]
        if candidates and update%2==0:
            candidate=read(Path(candidates[0]['folder'])/'score.json');validation=exp.validate(candidate)
            if validation['goal_met']:
                exp.state['goal_achieved']=True
                if not args.continue_after_goal or config.get('completion_rule',{}).get('stop_on_success'):
                    exp.state.update(status='goal_found',stage='completed',stage_note='요청한 모든 시간 간격에서 3m 이상과 안정 유지 조건을 통과했습니다.');break
                exp.state.update(status='running',stage='ppo_training',stage_note='목표 거리를 통과했으며 고정된 토크 상한 안에서 더 긴 안정 착지 후보를 계속 탐색합니다.')
    if exp.state['status']!='goal_found':
        if(OUT/'stop.request').exists():exp.state.update(status='checkpointed',stage='reconfiguration',stage_note='완료된 정책 배치와 모든 시도를 저장했습니다. 재개할 수 있는 체크포인트가 있습니다.')
        else:exp.state.update(status='budget_finished',stage='completed',stage_note='설정된 계산 범위를 마쳤습니다. 목표 달성 여부와 최선 안정 후보를 구분해서 보존했습니다.')

def main():
    global OUT,TARGET_FIRST,TARGET_RETAINED,VERSION
    ap=argparse.ArgumentParser();ap.add_argument('--episodes',type=int,default=4096);ap.add_argument('--hours',type=float,default=6.);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--generations',type=int,default=6);ap.add_argument('--population',type=int,default=18);ap.add_argument('--resume',action='store_true');ap.add_argument('--label',default='landing_stability_v2');ap.add_argument('--definition-file');ap.add_argument('--warm-start');ap.add_argument('--reseed-parameters');ap.add_argument('--reseed-source-algorithm',default='Bounded same-domain CEM diagnostic; not a PPO update');ap.add_argument('--renew-budget',action='store_true');ap.add_argument('--continue-after-goal',action='store_true');args=ap.parse_args();assert args.workers==3 and args.label in ['landing_stability_v2','landing_stability_v3']
    OUT=ROOT/'results/controlled_broad_jump'/args.label
    OUT.mkdir(parents=True,exist_ok=True);WEB.mkdir(parents=True,exist_ok=True)
    if(OUT/'definition.json').exists()and not args.resume:raise RuntimeError('Existing experiment must be resumed, never overwritten. Use --resume.')
    config=read(OUT/'definition.json')if args.resume else read(args.definition_file)if args.definition_file else None
    if config is None:
        config=definition()
        # Read native ROM in a short isolated process, avoiding solver/torch DLL mixing.
        result=subprocess.run([sys.executable,'-c',"import mujoco,json; m=mujoco.MjModel.from_binary_path(r'D:/Exercise_mechanics/results/controlled_broad_jump/distance_stability_pilot_v1/stable_reference/control_model.mjb'); print(json.dumps({k:m.jnt_range[m.joint(k if k=='flex_extension'else k+'_r').id].tolist()for k in ['hip_flexion','knee_angle','ankle_angle','shoulder_elv','elbow_flexion','flex_extension']}))"],capture_output=True,text=True,check=True)
        config['joint_ranges_rad']=json.loads(result.stdout.strip());atomic(OUT/'definition.json',config)
    TARGET_FIRST=float(config.get('goal_first_contact_m',TARGET_FIRST));TARGET_RETAINED=float(config.get('goal_retained_contact_m',TARGET_RETAINED));VERSION=config.get('score_version',VERSION)
    previous_version=config.get('version');config['version']=VERSION;atomic(OUT/'definition.json',config)
    config['training_precision_mixture_s']=[.0005,.00025,.00025];config['mean_policy_evaluation_dt_s']=.00025;atomic(OUT/'definition.json',config)
    if previous_version!=VERSION:
        with(OUT/'objective_versions.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(dict(at=time.time(),previous=previous_version,current=VERSION,reason='Continuous native-contact-free interval reference for conservation gates; reward shaping weights unchanged.',historical_rows_unchanged=True))+'\n')
    exp=Experiment(config,args.workers);new_deadline=time.time()+args.hours*3600;deadline=new_deadline if args.renew_budget else min(new_deadline,exp.state.get('deadline',new_deadline))if args.resume else min(new_deadline,config.get('original_budget_deadline',new_deadline))
    if args.renew_budget:
        extension=dict(at=time.time(),previous_deadline=exp.state.get('deadline'),new_deadline=deadline,training_episodes_at_resume=exp.state['training_episodes'],planned_total_episodes=args.episodes,reason='User explicitly requested further search; body, torque limits, contact physics and action bounds unchanged.');exp.state['budget_extension']=extension
        with(OUT/'budget_extensions.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(extension)+'\n')
    try:
        recovery_initialization(exp,args)
        completion=config.get('completion_rule');verified=exp.state.get('verified')or{}
        if completion and verified.get('min_fine_retained_m',0.)>=float(completion['min_retained_m'])and set(verified.get('timesteps_s',[]))==set(completion.get('required_timesteps_s',[.0005,.00025,.000125])):
            exp.state.update(status='goal_found',stage='completed',goal_achieved=True,continue_after_goal=False,completion_rule=completion,stage_note='3개 시간 간격에서 안정 유지 3m를 통과했습니다. 요청한 종료 조건에 따라 학습을 마쳤습니다.');return
        if args.warm_start and exp.state['native_evaluations']==0:
            exp.state.update(status='running',stage='controller_profile_baseline',stage_note='수정된 착지 제어로 안정 기준과 장거리 후보를 재평가합니다. 이전 실패와 정책은 보존했습니다.')
            anchors=exp.batch([dict(cmd='rollout',base=exp.config['stable_anchor'],kind='stable_anchor'),dict(cmd='rollout',base=exp.config['long_anchor'],kind='long_anchor')]);exp.validate(anchors[0])
            if anchors[1]['goal_met']:
                validation=exp.validate(anchors[1])
                if validation['goal_met']:exp.state.update(status='goal_found',stage='completed',stage_note='수정된 착지 제어의 기준 동작에서 목표 거리와 안정 조건을 정밀 시간 스텝에서 통과했습니다.');return
        if args.resume and exp.state.get('verified')is None:
            best=read(OUT/'best_stable.json')
            if best is not None:exp.validate(best)
        train(exp,args,deadline)
    except Exception as exc:
        exp.state.update(status='failed',stage='error',error=repr(exc));traceback.print_exc();raise
    finally:exp.close()

if __name__=='__main__':main()


