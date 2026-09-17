"""V4.1 staged curriculum (protocol 3.3).

Main track: connectome + BC/DAgger + anchored PPO; only it decides promotion, with fixed
validation blocks. Science track: controls on fixed tasks with a wall-time budget; it never
blocks promotion. Every decision is traced; infrastructure incidents never reach learning.
"""
from pathlib import Path
import sys,os,json,time,random,traceback,argparse,uuid,faulthandler
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
faulthandler.enable() # native crashes (numba, torch, ctypes) leave a traceback in worker.stderr.log
parser=argparse.ArgumentParser()
parser.add_argument('--hours',type=float,default=24)
parser.add_argument('--reset-deadline',action='store_true',help='Start a new pilot window of --hours from now')
parser.add_argument('--config',default='config/learning.json',help='Protocol file relative to the lab root')
parser.add_argument('--max-iterations',type=int,default=0,help='Stop after this many learner iterations (smoke tests)')
args=parser.parse_args()
CONFIG_PATH=ROOT/args.config
runtime_threads=str(json.loads(CONFIG_PATH.read_text()).get('max_cpu_threads',2))
os.environ['OMP_NUM_THREADS']=runtime_threads;os.environ['MKL_NUM_THREADS']=runtime_threads
import numpy as np,torch
from learning_brain import LearningBrain,Dynamics
from learning_env import FlyOperatorEnv,ExerciseTeacher
from learning_adapter import native_progress,decision_required
from learning_policy import make_model,decision,imitate,actor_hash
from learning_curriculum import add_demonstration,run_checkpointed_evaluation,block_passed,promotion_ready,training_stage,wilson_lower
from learning_ppo import RolloutCollector,ppo_update,new_state,install_group_optimizer
from learning_checkpoint import save_checkpoint,load_checkpoint,load_teaching_examples
from learning_runtime import ResourceGuard,WorkerLock,StopTraining,PilotFinished,LiveState
from lab_store import Store,RUNTIME,atomic_json
from windows_session import keep_system_awake

REQUIRED={'move':'move','stance':'crouch','cancel':'cancel','orient':'turn','switch':'equip','shoot':'fire','reload':'reload','door':'door_open','breach':'door_breach','grenade':'throw','elimination':'fire','rescue':'follow','defuse':'defuse','loadout':'loadout'}
STAGE_ACTIONS={
 'move':{'wait','move','stop','cancel'},'stance':{'wait','crouch'},'cancel':{'wait','cancel','door_breach'},
 'orient':{'wait','turn','stop','cancel'},'switch':{'wait','equip','cancel'},
 'shoot':{'wait','turn','aim_target','fire','reload','stop','cancel','crouch'},'reload':{'wait','turn','fire','reload','stop','cancel'},
 'door':{'wait','move','stop','cancel','door_open'},'breach':{'wait','move','stop','cancel','equip','door_breach'},
 'grenade':{'wait','turn','throw','stop','cancel','crouch'},
 'elimination':{'wait','move','stop','cancel','crouch','turn','aim_target','fire','reload','equip','door_open','door_breach','throw'},
 'rescue':{'wait','move','stop','cancel','crouch','turn','follow','door_open'},
 'defuse':{'wait','move','stop','cancel','crouch','turn','defuse','door_open','fire','reload','aim_target'},
 'loadout':{'wait','loadout','move','stop','cancel','crouch','turn','aim_target','fire','reload','equip','door_open','door_breach','throw'}
}
SCHEDULE_VERSION=4

class CalibrationFailed(Exception):pass

def append_jsonl(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(value,allow_nan=False)+'\n')

def main():
    cfg=json.loads(CONFIG_PATH.read_text());run=cfg['experiment_id'];store=Store(run);guard=ResourceGuard(store,cfg,args.hours)
    if cfg.get('semantic_protocol_version')!='3.3' or 'ppo_v33' not in cfg:raise ValueError('This trainer runs protocol 3.3 only')
    campaign=RUNTIME/'campaigns'/run;checkpoint_root=campaign/'checkpoints';campaign.mkdir(parents=True,exist_ok=True)
    schedule_path=campaign/'schedule.json'
    schedule=json.loads(schedule_path.read_text()) if schedule_path.exists() else {
      'version':SCHEDULE_VERSION,'stage_index':0,'prepared':{},'bootstrap_evaluations':{},'blocks':{},'best':{},
      'main_index':0,'science_index':0,'round':0,'wall':{'main':0.,'science':0.},'science_initialized':[],'promotions':[],'rollbacks':[]}
    if schedule.get('version')!=SCHEDULE_VERSION:raise ValueError('Curriculum schedule version mismatch')
    # The pilot window belongs to the campaign, not to one worker process.
    if args.reset_deadline or 'pilot_deadline' not in schedule:schedule['pilot_deadline']=guard.deadline;atomic_json(schedule_path,schedule)
    guard.deadline=float(schedule['pilot_deadline'])
    cap=json.loads((ROOT/'outputs/native_capabilities_v2.json').read_text());stages=[s for s in cfg['stage_order'] if REQUIRED[s] in cap['validated_actions']]
    tracks=cfg['tracks'];main_conditions=list(tracks['main']);science_conditions=list(tracks.get('science',[]))
    science_stage=tracks['science_stages'][0] if tracks.get('science_stages') else None
    settings=cfg['condition_settings'];promotion=cfg['promotion'];ppo_cfg=cfg['ppo_v33'];seeds=list(cfg['seeds'])
    for condition in main_conditions+science_conditions:
        if condition not in settings:raise ValueError(f'Condition without settings: {condition}')
    store.update(state='preparing',reason='Cargando MaleCNS completo y currículo v4.1 (protocolo 3.3)',enabled_stages=stages,pending_stages=[s for s in cfg['stage_order'] if s not in stages],
                 pilot_hours=args.hours,campaign_root=str(campaign),schedule_path=str(schedule_path),curriculum_version=4,
                 tracks=tracks,protocol=cfg['semantic_protocol_version'])
    dynamics=Dynamics(learning_rate=cfg['motor_plasticity_learning_rate']) if 'motor_plasticity_learning_rate' in cfg else None
    guard();brain=LearningBrain(dynamics=dynamics);env=FlyOperatorEnv(brain,guard=guard,sensory_schema=cfg.get('sensory_schema'));env.config=cfg;live=LiveState(brain,run)
    model=None;counters={};current_name=None;latest_checkpoint=None;examples=[];checkpoint_ready=False;checkpointing_enabled=True
    episode_actor_hash=None;evaluation_records=None;collectors={};timing_ema={};iterations=0
    legacy_examples=campaign/'no_legacy_examples.npz'

    def point(name):return checkpoint_root/name/'current.json'
    def persist():
        nonlocal latest_checkpoint
        if checkpointing_enabled and model is not None and current_name and checkpoint_ready:
            store.update(state='saving',reason='Guardando red, política, currículo y ejemplos')
            latest_checkpoint=save_checkpoint(current_name,brain,model,env,counters,store,
              teaching_examples=examples if env.condition=='common_teaching' else None,checkpoint_root=checkpoint_root)
            atomic_json(schedule_path,schedule)
    guard.on_pause=persist

    def set_checkpointing(enabled):
        nonlocal checkpointing_enabled
        checkpointing_enabled=bool(enabled)

    def record_step(info):
        if evaluation_records is None:store.step(info)
        else:
            snapshot=json.loads(json.dumps(info,allow_nan=False));evaluation_records['steps'].append(snapshot)
            store.preview(snapshot,evaluation_records['transaction_id'])

    def record_episode(info):
        if evaluation_records is None:store.episode(info)
        else:evaluation_records['episodes'].append(json.loads(json.dumps(info,allow_nan=False)))

    def configure(condition,seed,name):
        """Select learner identity and feature space; the model is rebuilt for that space."""
        nonlocal model,current_name,checkpoint_ready
        env.seed_base=seed;env.condition=condition;env.train_plasticity=False
        env.set_brain_mode('connectome' if condition=='common_teaching' else settings[condition]['brain_mode'])
        current_name=name;checkpoint_ready=False;model=make_model(env,cfg,seed)

    def fresh_counters(stage,origin):
        return {'decisions':0,'episodes':0,'common_stage':stage,'ppo':new_state(),'last_block_decisions':0,'practice':{},
                'diagnostics':{'instructor_executed':0,'excluded_episodes':0,'ppo_updates':0,'actor_delta_norm_total':0.,'identity_failures':0},
                'rollbacks':0,'initialized_from':origin}

    def reset(stage,split='train',mission=None):
        env.stage=stage;env.split=split;env.curriculum_allowed=STAGE_ACTIONS[stage];live.begin()
        try:return env.reset(options={'mission':mission} if mission else {})[0]
        finally:live.end(getattr(env,'last_counts',np.zeros(brain.n,np.int32)),env.condition,env.seed_base)

    def advance(x,mode,instructor=False,collect=False,deterministic=False,teacher_beta=0.):
        guard();started=time.perf_counter();mask=env.action_masks()
        learner,value,logprob,detail=decision(model,x,mask,env.catalog,env.readout.body_ids,deterministic)
        # The instructor labels every visited state (anchor and audit) but controls only when told to.
        teacher=ExerciseTeacher(env.catalog).choose(env)
        use_teacher=bool(instructor or (teacher_beta and random.random()<teacher_beta));action=teacher if use_teacher else learner
        was_progress=native_progress(env.obs);cognitive_decision=decision_required(env.obs,mask)
        stored=add_demonstration(examples,x,teacher,mask,env.stage,env.episode_count,env.last_sensor_features) if collect else False
        detail.update(learner_selected=learner,requested_by_actor=learner,teacher_action=teacher,teacher_label_legal=bool(mask[teacher]),
                      executed=action,demonstration_stored=stored,controller='instructor' if use_teacher else 'neuronal_actor',
                      native_progress=was_progress,cognitive_decision=cognitive_decision,feature_space=env.brain_mode)
        decision_seconds=time.perf_counter()-started
        live.begin()
        try:
            next_x,reward,term,trunc,info=env.step(action)
            info.update(decision=detail,learning={'changed_edges':0},phase=mode,training_steps=counters.get('decisions',0),actor_sha256=episode_actor_hash)
            recorded=time.perf_counter();record_step(info);record_seconds=time.perf_counter()-recorded
            if guard.probe:
                guard.probe=False;persist();env.neural_record=True
            elif env.neural_record:
                events=brain.last_events
                if events is not None:
                    path=RUNTIME/'probes';path.mkdir(exist_ok=True)
                    np.savez_compressed(path/f'{env.condition}_{env.seed_base}_{info["episode"]}_{info["sequence"]}.npz',indices=events[0],ticks=events[1],counts=env.last_counts)
                env.neural_record=False
        finally:live.end(getattr(env,'last_counts',np.zeros(brain.n,np.int32)),env.condition,env.seed_base)
        timing=dict(info['timing'],decision=decision_seconds,record=record_seconds)
        for key,seconds in timing.items():timing_ema[key]=seconds if key not in timing_ema else .98*timing_ema[key]+.02*seconds
        guard.after_step();store.update(condition=env.condition,seed=env.seed_base,stage=env.stage,phase=mode,episode=info['episode'],
          simulated_ms=env.total_simulated_ms,steps=counters.get('decisions',0),neural_wall_seconds=info['activity']['wall_seconds'],last_reward=reward,
          graph_hash=brain.graph_hash,wait_probability=detail['wait_probability'],action_entropy=detail['entropy'],free_wait_streak=info['free_wait_streak'],
          brain_mode=env.brain_mode,timing_ema_seconds=timing_ema)
        return next_x,reward,term,trunc,info,action,value,logprob,mask,cognitive_decision,teacher,use_teacher

    def episode(stage,mode,instructor=False,collect=False,deterministic=False,mission=None,split='train',teacher_beta=0.):
        """One natural episode. Returns cognitive transitions (macro rewards discounted by gamma**ticks)."""
        nonlocal episode_actor_hash
        x=reset(stage,split,mission);transitions=[];actor_before=actor_hash(model);current=None;ticks=0;instructor_executed=0
        episode_actor_hash=actor_before;gamma=float(ppo_cfg['gamma'])
        while True:
            previous=x.copy()
            x,reward,term,trunc,info,action,value,logprob,mask,cognitive,teacher,use_teacher=advance(previous,mode,instructor,collect,deterministic,teacher_beta)
            ticks+=1
            if cognitive:
                if current is not None:transitions.append(current)
                instructor_executed+=int(use_teacher)
                current={'observation':previous,'action':int(action),'reward':float(reward),'value':float(value.reshape(-1)[0]),
                         'logprob':float(logprob.reshape(-1)[0]),'mask':mask.copy(),'macro_steps':1,'teacher':int(teacher)}
                if mode not in ('evaluation','teaching'):counters['decisions']=counters.get('decisions',0)+1
            elif current is not None:
                current['reward']+=(gamma**current['macro_steps'])*float(reward);current['macro_steps']+=1
            if term or trunc:
                if current is not None:transitions.append(current)
                break
        if actor_hash(model)!=actor_before:raise AssertionError('Actor changed inside an episode')
        bootstrap=None
        if trunc and not term:
            with torch.no_grad():bootstrap=float(model.policy.predict_values(torch.as_tensor(x[None,:],dtype=torch.float32))[0,0])
        metrics={'ticks':ticks,'cognitive_decisions':len(transitions),'instructor_executed':instructor_executed,
                 'incidents':len(env.incidents),'excluded_from_learning':bool(env.incidents),'episode_uid':env.episode_uid,
                 'success':bool(info['success']),'truncation_reason':info.get('truncation_reason')}
        return info,metrics,transitions,term,trunc,bootstrap

    def evaluate(stage,retention,count=None):
        env.train_plasticity=False;results={};seed_slot=seeds.index(env.seed_base)
        tested=(stages[:stages.index(stage)] if retention else [])+[stage]
        for test_stage in tested:
            if count is not None:episodes=int(count)
            else:episodes=int(promotion['block_episodes'] if test_stage==stage else promotion['retention_episodes'])
            candidates=[m for m in env.scenarios if m['stage']==test_stage and m['split']=='validation']
            outcomes=[];reasons=[];incidents=0
            for i in range(episodes):
                mission=candidates[seed_slot*30+i]['name']
                info,metrics,_,_,_,_=episode(test_stage,'evaluation',deterministic=True,mission=mission,split='validation')
                outcomes.append(bool(info['success']));reasons.append(info.get('truncation_reason'));incidents+=metrics['incidents']
            results[test_stage]={'successes':sum(outcomes),'episodes':len(outcomes),'rate':sum(outcomes)/len(outcomes),
                                 'wilson_lower':wilson_lower(sum(outcomes),len(outcomes)),'policy_collapses':sum(r=='policy_collapse' for r in reasons),
                                 'truncation_reasons':dict(Counter(r for r in reasons if r)),'incidents':incidents}
        return results

    def evaluate_clean(stage,retention,count=None):
        """Evaluate from a clean checkpoint and restore even on stop/error."""
        nonlocal counters,evaluation_records
        saved_counters=json.loads(json.dumps(counters));saved_condition=env.condition
        def save():
            persist()
            return point(current_name)
        def restore(saved):
            nonlocal counters
            counters=load_checkpoint(saved,brain,model,env);counters.update(saved_counters)
            env.condition=saved_condition;env.train_plasticity=False
        if evaluation_records is not None:raise RuntimeError('Nested evaluation transaction')
        transaction_id=f'{current_name}:{stage}:{uuid.uuid4().hex}'
        evaluation_records={'transaction_id':transaction_id,'steps':[],'episodes':[]}
        try:
            results=run_checkpointed_evaluation(save,restore,lambda:evaluate(stage,retention,count),set_checkpointing)
            buffered=evaluation_records;evaluation_records=None
            store.evaluation(transaction_id,buffered['steps'],buffered['episodes'])
            return results,transaction_id
        finally:
            evaluation_records=None

    def run_block(stage,track,reason):
        """A fixed validation block on a frozen checkpoint; best-checkpoint bookkeeping and rollback."""
        nonlocal counters
        started=time.time();store.update(state='evaluating',reason=f'Bloque de validación {current_name} · {stage}')
        results,transaction=evaluate_clean(stage,retention=track=='main')
        passed=block_passed(results,promotion,stage);rate=results[stage]['rate']
        record={'stage':stage,'learner':current_name,'track':track,'reason':reason,'results':results,'passed':passed,
                'decisions':counters.get('decisions',0),'ppo_updates':counters.get('ppo',{}).get('updates',0),
                'ppo_phase':counters.get('ppo',{}).get('phase'),'actor_sha256':actor_hash(model),'transaction_id':transaction,
                'time':time.time(),'wall_seconds':time.time()-started}
        counters['last_block_decisions']=counters.get('decisions',0)
        key=f'{current_name}:{stage}';best=schedule['best'].get(key)
        if best is None or rate>best['rate']:
            save_checkpoint(current_name+'_best',brain,model,env,counters,store,checkpoint_root=checkpoint_root)
            schedule['best'][key]={'rate':rate,'decisions':counters.get('decisions',0),'time':time.time()};record['new_best']=True
        elif rate<best['rate']-float(promotion.get('rollback_drop',.2)):
            lifetime={'decisions':counters.get('decisions',0),'episodes':counters.get('episodes',0),'rollbacks':counters.get('rollbacks',0)}
            restored=load_checkpoint(point(current_name+'_best'),brain,model,env)
            restored['rollbacks']=int(lifetime['rollbacks'])+1;restored['lifetime_before_rollback']=lifetime
            counters=restored;collectors.pop(current_name,None)
            record['rollback']={'to_rate':best['rate'],'from_rate':rate,'restored_decisions':restored.get('decisions',0)}
            schedule['rollbacks'].append({'learner':current_name,'stage':stage,**record['rollback'],'time':time.time()})
        schedule['blocks'].setdefault(stage,{}).setdefault(current_name,[]).append(record)
        append_jsonl(campaign/'blocks.jsonl',record);store.update(last_block=record)
        persist();return record

    def practice_ok(stage):
        window=int(promotion['prescreen_window']);history=counters.get('practice',{}).get(stage,[])
        return len(history)>=window and sum(history[-window:])/window>=float(promotion['prescreen_success'])

    def learner_iteration(condition,seed,stage,track):
        nonlocal counters,examples
        name=f'{condition}_{seed}';configure(condition,seed,name);examples=[]
        own=point(name);common=point(f'common_{seed}')
        if own.exists():counters=load_checkpoint(own,brain,model,env)
        else:
            if settings[condition]['brain_mode']!='connectome':raise RuntimeError(f'{name} must be initialized from its own imitation')
            load_checkpoint(common,brain,model,env)
            # Fresh optimizer between behaviour cloning and PPO.
            install_group_optimizer(model,0.,ppo_cfg['critic_learning_rate']);counters=fresh_counters(stage,f'common_{seed}')
        _set_ready()
        if counters.get('common_stage')!=stage:
            common_examples=load_teaching_examples(common,legacy_examples)
            transfer=imitate(model,common_examples,cfg['stage_transfer_imitation_epochs'],own_optimizer=True)
            ppo=counters['ppo'];ppo.update(phase='critic_warmup',warmup_mse=[],stage_decisions=0)
            counters['common_stage']=stage;counters['last_block_decisions']=counters.get('decisions',0);collectors.pop(name,None)
            store.update(stage_transfer=transfer)
        learner=settings[condition]['learner'];started=time.time()
        practice_stage=training_stage(stages,stage,counters.get('episodes',0),int(round(1/cfg['retention_fraction']))) if track=='main' else stage
        store.update(state='training',reason=f'{name} · {practice_stage} · {track}')
        if learner=='none' and schedule['blocks'].get(stage,{}).get(name):
            info,metrics,transitions,term,trunc,bootstrap=episode(practice_stage,'frozen')
        elif learner=='none':
            record=run_block(stage,track,'initial_block');return time.time()-started
        else:
            info,metrics,transitions,term,trunc,bootstrap=episode(practice_stage,'ppo')
        counters['episodes']=counters.get('episodes',0)+1
        history=counters.setdefault('practice',{}).setdefault(practice_stage,[]);history.append(bool(info['success']))
        del history[:-int(promotion['prescreen_window'])]
        diagnostics=counters.setdefault('diagnostics',{})
        diagnostics['instructor_executed']=diagnostics.get('instructor_executed',0)+metrics['instructor_executed']
        diagnostics['excluded_episodes']=diagnostics.get('excluded_episodes',0)+int(metrics['excluded_from_learning'])
        update=None
        if learner=='ppo':
            collector=collectors.setdefault(name,RolloutCollector())
            collector.add_episode(transitions,term,trunc,bootstrap,metrics['excluded_from_learning'],metrics['episode_uid'])
            if collector.decisions>=int(ppo_cfg['rollout_decisions']):
                store.update(state='updating',reason=f'Actualización PPO {name}')
                learner_settings={**ppo_cfg,**{k:v for k,v in settings[condition].items() if k in ('update_every_rollouts','advantage_normalization')}}
                ppo=counters['ppo']
                update=ppo_update(model,collector,learner_settings,ppo);collector.clear()
                diagnostics['ppo_updates']=ppo['updates'];diagnostics['identity_failures']=ppo['identity_failures']
                diagnostics['actor_delta_norm_total']=diagnostics.get('actor_delta_norm_total',0.)+float(update.get('actor_delta_norm') or 0.)
                update.update(learner=name,stage=stage,time=time.time());append_jsonl(campaign/'ppo_updates.jsonl',update)
                store.update(last_ppo_update=update)
        store.update(last_rollout=metrics,learner_diagnostics={name:diagnostics},gains_delta_norm=float(np.linalg.norm(brain.gains-1.)) if env.brain_mode=='connectome' else 0.)
        persist()
        if learner=='ppo' and update and update.get('updated'):
            since=counters.get('decisions',0)-counters.get('last_block_decisions',0)
            if since>=int(promotion['block_every_decisions']):run_block(stage,track,'decision_milestone')
            elif track=='main' and counters['ppo']['phase']=='actor' and practice_ok(stage) and since>=int(promotion['min_decisions_between_blocks']):
                run_block(stage,track,'practice_prescreen')
        return time.time()-started

    def _set_ready():
        nonlocal checkpoint_ready
        checkpoint_ready=True

    def teach(stage):
        """Common instruction for every seed; must pass its bootstrap check before learners branch."""
        nonlocal counters,examples
        prepared=schedule['prepared'].setdefault(stage,[])
        for seed in seeds:
            if seed in prepared:continue
            configure('common_teaching',seed,f'common_{seed}');examples=[]
            counters={'steps':0,'guided':{},'dagger':{},'extra_dagger':{},'common_stage':stage,'examples_fitted_count':0}
            common_point=point(current_name)
            if common_point.exists():
                counters=load_checkpoint(common_point,brain,model,env);examples=load_teaching_examples(common_point,legacy_examples)
                fitted=int(counters.setdefault('examples_fitted_count',len(examples)))
                if fitted<len(examples):
                    recovery=imitate(model,examples,cfg['imitation_epochs'],own_optimizer=True);counters['examples_fitted_count']=len(examples)
                    store.update(imitation_recovery=recovery,recovered_unfitted_examples=len(examples)-fitted)
            else:brain.gains.fill(1.);brain.reset(seed)
            _set_ready();store.update(resumed_from=str(common_point) if common_point.exists() else None,restored_actor_sha256=actor_hash(model),restored_examples_count=len(examples))
            def teaching_episode(instructor,beta):
                before=len(examples)
                info,metrics,_,_,_,_=episode(stage,'teaching',instructor=instructor,collect=True,teacher_beta=beta)
                # Failed guided demonstrations and episodes with infrastructure incidents teach nothing.
                if (instructor and not info['success']) or metrics['excluded_from_learning']:del examples[before:]
                return info
            for phase,target in [('guided',cfg['guided_episodes_per_stage']),('dagger',cfg['dagger_episodes_per_stage'])]:
                counts=counters.setdefault(phase,{})
                while counts.get(stage,0)<target:
                    teaching_episode(phase=='guided',cfg['dagger_teacher_beta'] if phase=='dagger' else 0.)
                    counts[stage]=counts.get(stage,0)+1
                    metrics=imitate(model,examples,cfg['imitation_epochs'],own_optimizer=True);counters['examples_fitted_count']=len(examples)
                    store.update(imitation=metrics,teaching_stage=stage);persist()
            passed=False;attempt=0
            while not passed:
                results,_=evaluate_clean(stage,retention=False,count=cfg['bootstrap_evaluation_episodes'])
                passed=all(r['rate']>=cfg['promotion_success'] for r in results.values())
                schedule['bootstrap_evaluations'][f'{stage}:{seed}:{attempt}']=results
                if passed:break
                if attempt>=cfg['max_extra_dagger_episodes']:raise CalibrationFailed(f'Bootstrap {stage}/{seed} did not reach {cfg["promotion_success"]:.0%}')
                teaching_episode(False,cfg['dagger_teacher_beta'])
                counters.setdefault('extra_dagger',{})[stage]=counters.setdefault('extra_dagger',{}).get(stage,0)+1
                metrics=imitate(model,examples,cfg['imitation_epochs'],own_optimizer=True);counters['examples_fitted_count']=len(examples)
                store.update(imitation=metrics,teaching_stage=stage,bootstrap_retry=attempt+1);attempt+=1
            prepared.append(seed);persist()
            if stage==science_stage:initialize_science(seed)

    def initialize_science(seed):
        """Controls branch from the common checkpoint right after the science stage was taught."""
        nonlocal counters,examples
        if seed in schedule['science_initialized']:return
        common=point(f'common_{seed}');common_examples=load_teaching_examples(common,legacy_examples)
        for condition in science_conditions:
            name=f'{condition}_{seed}'
            if point(name).exists():continue
            configure(condition,seed,name);examples=[];mode=settings[condition]['brain_mode']
            if mode=='connectome':
                load_checkpoint(common,brain,model,env);install_group_optimizer(model,0.,ppo_cfg['critic_learning_rate'])
                counters=fresh_counters(science_stage,f'common_{seed}')
            else:
                if not common_examples or len(common_examples[0])<6:raise ValueError('Controls need sensor features stored with the teaching examples')
                inputs=[(e[5] if mode=='sensor_only' else np.zeros_like(e[5]),e[1],e[2],e[3],e[4]) for e in common_examples]
                torch.manual_seed(seed);fit=imitate(model,inputs,cfg['control_imitation_epochs'],own_optimizer=True)
                counters=fresh_counters(science_stage,f'imitation_of_common_{seed}_examples');counters['control_imitation']=fit
            _set_ready();persist()
        schedule['science_initialized'].append(seed);atomic_json(schedule_path,schedule)

    def science_iteration():
        """Round-robin over science learners; controls without a learner need only their blocks."""
        pairs=[(c,s) for c in science_conditions for s in seeds]
        for offset in range(len(pairs)):
            condition,seed=pairs[(schedule['science_index']+offset)%len(pairs)]
            if seed not in schedule['science_initialized']:continue
            if settings[condition]['learner']=='none':
                done=len(schedule['blocks'].get(science_stage,{}).get(f'{condition}_{seed}',[]))
                if done>=int(tracks.get('control_blocks',1)):continue
                schedule['science_index']=(schedule['science_index']+offset+1)%len(pairs)
                name=f'{condition}_{seed}';configure(condition,seed,name)
                nonlocal_load(name);started=time.time();run_block(science_stage,'science','control_block');return time.time()-started
            schedule['science_index']=(schedule['science_index']+offset+1)%len(pairs)
            return learner_iteration(condition,seed,science_stage,'science')
        return None

    def nonlocal_load(name):
        nonlocal counters
        counters=load_checkpoint(point(name),brain,model,env);_set_ready()

    env.on_episode=record_episode
    try:
      while True:
        stage=stages[min(schedule['stage_index'],len(stages)-1)]
        teach(stage)
        for seed in seeds:
            if science_stage in schedule['prepared'] and seed in schedule['prepared'][science_stage]:initialize_science(seed)
        total=schedule['wall']['main']+schedule['wall']['science']
        took=None;track='main'
        if science_conditions and schedule['wall']['science']<float(tracks.get('science_fraction',0.))*max(total,1.):
            took=science_iteration();track='science'
        if took is None:
            track='main';pairs=[(c,s) for c in main_conditions for s in seeds]
            condition,seed=pairs[schedule['main_index']%len(pairs)]
            took=learner_iteration(condition,seed,stage,'main')
            schedule['main_index']+=1
            if schedule['main_index']%len(pairs)==0:schedule['round']+=1
        schedule['wall'][track]+=float(took)
        if promotion_ready(schedule['blocks'],stage,main_conditions,seeds):
            schedule['promotions'].append({'stage':stage,'time':time.time(),'round':schedule['round']})
            if schedule['stage_index']<len(stages)-1:schedule['stage_index']+=1
            else:
                atomic_json(schedule_path,schedule)
                store.update(state='curriculum_complete',reason='Todas las habilidades superaron la puerta de promoción; falta el examen final en test');raise StopTraining()
        atomic_json(schedule_path,schedule)
        iterations+=1
        if args.max_iterations and iterations>=args.max_iterations:raise StopTraining()
    except CalibrationFailed as exc:
        persist();store.update(state='needs_calibration',reason=str(exc),checkpoint=str(latest_checkpoint))
    except (StopTraining,PilotFinished) as exc:
        persist()
        # A finished curriculum already published its own final state; do not overwrite it.
        if store.status.get('state')=='curriculum_complete':store.update(checkpoint=str(latest_checkpoint))
        else:store.update(state='stopped',reason='Piloto terminado' if isinstance(exc,PilotFinished) else 'Detenido con checkpoint',checkpoint=str(latest_checkpoint))
    except Exception as exc:
        try:persist()
        except Exception:traceback.print_exc()
        store.update(state='error',reason=str(exc),traceback=traceback.format_exc());raise
    finally:keep_system_awake(False);env.close();store.close()

if __name__=='__main__':
    with WorkerLock():
        try:main()
        except (StopTraining,PilotFinished):pass
