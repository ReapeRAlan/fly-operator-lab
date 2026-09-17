"""V3 staged curriculum: natural episodes, corrective DAgger and held-out gates."""
from pathlib import Path
import sys,os,json,time,random,traceback,argparse,uuid,faulthandler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
faulthandler.enable() # native crashes (numba, torch, ctypes) leave a traceback in worker.stderr.log
runtime_threads=str(json.loads((ROOT/'config/learning.json').read_text()).get('max_cpu_threads',2))
os.environ['OMP_NUM_THREADS']=runtime_threads;os.environ['MKL_NUM_THREADS']=runtime_threads
import numpy as np,torch
from learning_brain import LearningBrain,Dynamics
from learning_env import FlyOperatorEnv,ExerciseTeacher
from learning_adapter import native_progress,decision_required
from learning_policy import make_model,decision,imitate,internal_update,actor_hash
from learning_curriculum import add_demonstration,train_adapter_episode,forced_wait,run_checkpointed_evaluation
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

class CalibrationFailed(Exception):pass

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=24);parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--reset-deadline',action='store_true',help='Start a new pilot window of --hours from now');args=parser.parse_args()
    cfg=json.loads((ROOT/'config/learning.json').read_text());run=cfg['experiment_id'];store=Store(run);guard=ResourceGuard(store,cfg,args.hours)
    campaign=RUNTIME/'campaigns'/run;checkpoint_root=campaign/'checkpoints';campaign.mkdir(parents=True,exist_ok=True)
    schedule_path=campaign/'schedule.json'
    schedule=json.loads(schedule_path.read_text()) if schedule_path.exists() else {
      'version':3,'seed_index':0,'condition_index':0,'round':0,'stage_index':0,
      'prepared':{},'bootstrap_evaluations':{},'evaluations':{}}
    if schedule.get('version')!=3:raise ValueError('Curriculum schedule version mismatch')
    # The pilot window belongs to the campaign, not to one worker process.
    if args.reset_deadline or 'pilot_deadline' not in schedule:schedule['pilot_deadline']=guard.deadline;atomic_json(schedule_path,schedule)
    guard.deadline=float(schedule['pilot_deadline'])
    cap=json.loads((ROOT/'outputs/native_capabilities_v2.json').read_text());stages=[s for s in cfg['stage_order'] if REQUIRED[s] in cap['validated_actions']]
    store.update(state='preparing',reason='Cargando MaleCNS completo y currículo v3',enabled_stages=stages,pending_stages=[s for s in cfg['stage_order'] if s not in stages],
                 pilot_hours=args.hours,campaign_root=str(campaign),schedule_path=str(schedule_path),curriculum_version=3)
    dynamics=Dynamics(learning_rate=cfg['motor_plasticity_learning_rate']) if 'motor_plasticity_learning_rate' in cfg else None
    guard();brain=LearningBrain(dynamics=dynamics);env=FlyOperatorEnv(brain,guard=guard,sensory_schema=cfg.get('sensory_schema'));env.config=cfg;live=LiveState(brain)
    # Causal motor rules only credit edges entering descending neurons; tracking the
    # rest of the graph produced identical updates at ~24x the integration cost.
    if cfg.get('plasticity_rule','').startswith('causal_motor_rstdp'):brain.set_plastic_posts(env.readout.indices)
    model=None;counters={};current_name=None;latest_checkpoint=None;examples=[];checkpoint_ready=False;checkpointing_enabled=True
    episode_actor_hash=None
    evaluation_records=None

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

    def reset(stage,split='train',mission=None):
        env.stage=stage;env.split=split;env.curriculum_allowed=STAGE_ACTIONS[stage];live.begin()
        try:return env.reset(options={'mission':mission} if mission else {})[0]
        finally:live.end(getattr(env,'last_counts',np.zeros(brain.n,np.int32)),env.condition,env.seed_base)

    def advance(x,mode,macro,instructor=False,collect=False,deterministic=False,teacher_beta=0.):
        guard();mask=env.action_masks();learner,value,logprob,detail=decision(model,x,mask,env.catalog,env.readout.body_ids,deterministic)
        teacher=ExerciseTeacher(env.catalog).choose(env) if collect or instructor or teacher_beta else None
        use_teacher=bool(instructor or (teacher_beta and random.random()<teacher_beta));action=teacher if use_teacher else learner
        was_progress=native_progress(env.obs);cognitive_decision=decision_required(env.obs,mask)
        stored=add_demonstration(examples,x,teacher,mask,env.stage,env.episode_count if cfg.get('imitation') else None) if collect and (not was_progress or teacher!=0) else False
        detail.update(learner_selected=learner,teacher_action=teacher,executed=action,demonstration_stored=stored,
                      controller='instructor' if use_teacher else 'neuronal_actor',native_progress=was_progress,
                      cognitive_decision=cognitive_decision)
        if mode=='internal' and cognitive_decision:
            prior_update=None
            if macro:
                prior_update=internal_update(model,brain,macro['previous'],x,macro['reward'],False,
                  cfg['ppo']['gamma']**macro['ticks'],cfg.get('plasticity_reward_gain',1.),
                  eligibility_snapshot=macro['snapshot'],action=macro['action'],mask=macro['mask'],descending_indices=env.readout.indices,
                  rule=cfg['plasticity_rule'])
                prior_update.update(macro_ticks=macro['ticks'],macro_reward=macro['reward'],
                  credit_start_sequence=macro['start_sequence'],macro_completed=True,interrupted_by_action=env.catalog.label(action));macro.clear()
            macro.update(previous=x.copy(),action=action,mask=mask.copy(),snapshot=brain.capture_motor_eligibility(env.readout.indices),
                          reward=0.,ticks=0,start_sequence=int(env.raw.get('sequence',0))+1)
        else:prior_update=None
        live.begin()
        try:
            next_x,reward,term,trunc,info=env.step(action)
            update={'changed_edges':0}
            if mode=='internal' and macro:
                macro['reward']+=(cfg['ppo']['gamma']**macro['ticks'])*reward;macro['ticks']+=1
                next_decision=not (term or trunc) and decision_required(env.obs,env.action_masks())
                if term or trunc or next_decision:
                    update=internal_update(model,brain,macro['previous'],next_x,macro['reward'],term,
                      cfg['ppo']['gamma']**macro['ticks'],cfg.get('plasticity_reward_gain',1.),
                      eligibility_snapshot=macro['snapshot'],action=macro['action'],mask=macro['mask'],descending_indices=env.readout.indices,
                      rule=cfg['plasticity_rule'])
                    update.update(macro_ticks=macro['ticks'],macro_reward=macro['reward'],credit_start_sequence=macro['start_sequence'],macro_completed=True);macro.clear()
                else:update={'changed_edges':0,'macro_completed':False,'macro_ticks':macro['ticks'],'credit_start_sequence':macro['start_sequence']}
            if prior_update:
                update['prior_macro']=prior_update;update['changed_edges']=int(update.get('changed_edges',0))+int(prior_update.get('changed_edges',0))
            # The actor only changes between episodes (imitation or PPO); hash it once per episode.
            info.update(decision=detail,learning=update,phase=mode,training_steps=counters.get('steps',0),actor_sha256=episode_actor_hash)
            record_step(info)
            if guard.probe:
                guard.probe=False;persist();env.neural_record=True
            elif env.neural_record:
                events=brain.last_events
                if events is not None:
                    path=RUNTIME/'probes';path.mkdir(exist_ok=True)
                    np.savez_compressed(path/f'{env.condition}_{env.seed_base}_{info["episode"]}_{info["sequence"]}.npz',indices=events[0],ticks=events[1],counts=env.last_counts)
                env.neural_record=False
        finally:live.end(getattr(env,'last_counts',np.zeros(brain.n,np.int32)),env.condition,env.seed_base)
        guard.after_step();store.update(condition=env.condition,seed=env.seed_base,stage=env.stage,phase=mode,episode=info['episode'],
          simulated_ms=env.total_simulated_ms,steps=counters.get('steps',0),neural_wall_seconds=info['activity']['wall_seconds'],last_reward=reward,
          graph_hash=brain.graph_hash,wait_probability=detail['wait_probability'],action_entropy=detail['entropy'],free_wait_streak=info['free_wait_streak'])
        return next_x,reward,term,trunc,info,action,value,logprob,mask,cognitive_decision

    def episode(stage,mode,instructor=False,collect=False,deterministic=False,mission=None,split='train',teacher_beta=0.,train_actor=False):
        nonlocal episode_actor_hash
        x=reset(stage,split,mission);transitions=[];episode_start=True;actor_before=actor_hash(model);current=None;macro={};environment_ticks=0
        episode_actor_hash=actor_before
        episode_counter_start=counters.get('steps',0)
        try:
            while True:
                previous=x.copy();x,reward,term,trunc,info,action,value,logprob,mask,cognitive_decision=advance(x,mode,macro,instructor,collect,deterministic,teacher_beta);environment_ticks+=1
                if cognitive_decision:
                    if current is not None:transitions.append(current)
                    current={'observation':previous,'action':action,'reward':reward,'episode_start':episode_start,'value':value,'logprob':logprob,'mask':mask,'macro_steps':1}
                    model.num_timesteps+=1;counters['steps']=counters.get('steps',0)+1;episode_start=False
                elif current is not None:
                    current['reward']+=(cfg['ppo']['gamma']**current['macro_steps'])*reward;current['macro_steps']+=1
                if current is not None and (term or trunc):
                    transitions.append(current);current=None
                if term or trunc:break
        except BaseException:
            interrupted=max(0,counters.get('steps',0)-episode_counter_start)
            counters['interrupted_episode_steps']=counters.get('interrupted_episode_steps',0)+interrupted
            raise
        metrics=train_adapter_episode(model,transitions,x,term,trunc,cfg) if train_actor else {'ticks':len(transitions),'updated':False}
        metrics.update(ticks=environment_ticks,environment_ticks=environment_ticks,cognitive_decisions=len(transitions),
                       forced_waits=environment_ticks-len(transitions))
        if mode in ('internal','frozen','evaluation') and actor_hash(model)!=actor_before:raise AssertionError('Actor changed in a frozen phase')
        if macro:raise AssertionError('Terminal episode retained unresolved motor credit')
        if args.smoke_steps and model.num_timesteps>=args.smoke_steps:raise StopTraining()
        return info,metrics,len(transitions)

    def evaluate(stage,condition,count):
        env.train_plasticity=False;results={};seed_slot=cfg['seeds'].index(env.seed_base)
        for test_stage in stages[:stages.index(stage)+1]:
            candidates=[m for m in env.scenarios if m['stage']==test_stage and m['split']=='validation'];outcomes=[];reasons=[]
            for i in range(count):
                mission=candidates[seed_slot*30+i]['name'];info,_,_=episode(test_stage,'evaluation',deterministic=True,mission=mission,split='validation')
                outcomes.append(bool(info['success']));reasons.append(info.get('truncation_reason'))
            results[test_stage]={'successes':sum(outcomes),'episodes':len(outcomes),'rate':sum(outcomes)/len(outcomes),
                                 'policy_collapses':sum(r=='policy_collapse' for r in reasons)}
        return results

    def evaluate_clean(stage,condition,count):
        """Evaluate from a clean checkpoint and restore even on stop/error."""
        nonlocal counters,evaluation_records
        saved_counters=json.loads(json.dumps(counters));saved_condition=env.condition;saved_plasticity=env.train_plasticity
        def save():
            persist()
            return point(current_name)
        def restore(saved):
            nonlocal counters
            counters=load_checkpoint(saved,brain,model,env);counters.update(saved_counters)
            env.condition=saved_condition;env.train_plasticity=saved_plasticity
        if evaluation_records is not None:raise RuntimeError('Nested evaluation transaction')
        transaction_id=f'{current_name}:{stage}:{uuid.uuid4().hex}'
        evaluation_records={'transaction_id':transaction_id,'steps':[],'episodes':[]}
        try:
            results=run_checkpointed_evaluation(save,restore,lambda:evaluate(stage,condition,count),set_checkpointing)
            buffered=evaluation_records;evaluation_records=None
            store.evaluation(transaction_id,buffered['steps'],buffered['episodes'])
            return results
        finally:
            evaluation_records=None

    env.on_episode=record_episode
    try:
      while True:
        stage=stages[min(schedule['stage_index'],len(stages)-1)];prepared=schedule['prepared'].setdefault(stage,[])
        # Common instruction is stage-specific and must pass held-out checks before branching.
        for seed in cfg['seeds']:
            if seed in prepared:continue
            env.seed_base=seed;env.condition='common_teaching';env.train_plasticity=False;current_name=f'common_{seed}';checkpoint_ready=False
            model=make_model(env,cfg,seed);counters={'steps':0,'guided':{},'dagger':{},'extra_dagger':{},'common_stage':stage,'examples_fitted_count':0};examples=[]
            common_point=point(current_name)
            if common_point.exists():
                counters=load_checkpoint(common_point,brain,model,env);examples=load_teaching_examples(common_point,campaign/'no_legacy_examples.npz')
                fitted=int(counters.setdefault('examples_fitted_count',len(examples)))
                if fitted<len(examples):
                    recovery=imitate(model,examples,cfg['imitation_epochs'],settings=cfg.get('imitation'),catalog=env.catalog);counters['examples_fitted_count']=len(examples)
                    store.update(imitation_recovery=recovery,recovered_unfitted_examples=len(examples)-fitted)
            else:brain.gains.fill(1.);brain.reset(seed)
            checkpoint_ready=True;store.update(resumed_from=str(common_point) if common_point.exists() else None,restored_actor_sha256=actor_hash(model),restored_steps=counters.get('steps',0),restored_examples_count=len(examples))
            for phase,target in [('guided',cfg['guided_episodes_per_stage']),('dagger',cfg['dagger_episodes_per_stage'])]:
                counts=counters.setdefault(phase,{})
                while counts.get(stage,0)<target:
                    before=len(examples);info,_,ticks=episode(stage,'teaching',instructor=phase=='guided',collect=True,
                      teacher_beta=cfg['dagger_teacher_beta'] if phase=='dagger' else 0.)
                    if phase=='guided' and not info['success']:del examples[before:]
                    counts[stage]=counts.get(stage,0)+1
                    metrics=imitate(model,examples,cfg['imitation_epochs'],settings=cfg.get('imitation'),catalog=env.catalog);counters['examples_fitted_count']=len(examples);store.update(imitation=metrics,teaching_stage=stage)
                    persist()
            # Evaluate from a checkpoint and restore it, so validation cannot contaminate learning state.
            passed=False;attempt=0
            while not passed:
                results=evaluate_clean(stage,'common_teaching',cfg['bootstrap_evaluation_episodes'])
                passed=all(r['rate']>=cfg['promotion_success'] for r in results.values())
                schedule['bootstrap_evaluations'][f'{stage}:{seed}:{attempt}']=results
                if passed:break
                if attempt>=cfg['max_extra_dagger_episodes']:raise CalibrationFailed(f'Bootstrap {stage}/{seed} did not reach {cfg["promotion_success"]:.0%}')
                info,_,ticks=episode(stage,'teaching',collect=True,teacher_beta=cfg['dagger_teacher_beta'])
                counters.setdefault('extra_dagger',{})[stage]=counters.setdefault('extra_dagger',{}).get(stage,0)+1
                metrics=imitate(model,examples,cfg['imitation_epochs'],settings=cfg.get('imitation'),catalog=env.catalog);counters['examples_fitted_count']=len(examples);store.update(imitation=metrics,teaching_stage=stage,bootstrap_retry=attempt+1);attempt+=1
            prepared.append(seed);persist()

        si=schedule['seed_index'];ci=schedule['condition_index'];seed=cfg['seeds'][si];condition=cfg['conditions'][ci]
        env.seed_base=seed;env.condition=condition;current_name=f'{condition}_{seed}';checkpoint_ready=False;model=make_model(env,cfg,seed);examples=[]
        condition_point=point(current_name);common_point=point(f'common_{seed}')
        restored=condition_point if condition_point.exists() else common_point
        counters=load_checkpoint(restored,brain,model,env);checkpoint_ready=True
        if not condition_point.exists():counters={'steps':0,'episodes':0,'evaluated_at':0,'common_stage':stage,'modes':{}}
        elif counters.get('common_stage')!=stage:
            common_examples=load_teaching_examples(common_point,campaign/'no_legacy_examples.npz')
            transfer=imitate(model,common_examples,cfg['stage_transfer_imitation_epochs'],settings=cfg.get('imitation'),catalog=env.catalog);counters['common_stage']=stage;store.update(stage_transfer=transfer)
        store.update(resumed_from=str(restored),restored_actor_sha256=actor_hash(model),restored_steps=counters.get('steps',0))
        mode='adapter' if condition=='adapter' or (condition=='combined' and counters.get('episodes',0)%2==0) else 'internal' if condition in ('internal','combined') else 'frozen'
        env.train_plasticity=mode=='internal'
        index=stages.index(stage);training_stage=stages[(counters.get('episodes',0)//4)%index] if index and counters.get('episodes',0)%4==0 else stage
        info,rollout,ticks=episode(training_stage,mode,train_actor=mode=='adapter')
        counters['episodes']=counters.get('episodes',0)+1;counters.setdefault('modes',{})[mode]=counters.setdefault('modes',{}).get(mode,0)+1
        store.update(last_rollout=rollout);persist()
        if counters['steps']-counters.get('evaluated_at',0)>=cfg['evaluation_interval_steps']:
            results=evaluate_clean(stage,condition,cfg['promotion_evaluation_episodes'])
            counters['evaluated_at']=counters['steps'];schedule['evaluations'].setdefault(stage,{})[current_name]={'stage':stage,'results':results,'steps':counters['steps']};persist()
        schedule['condition_index']+=1
        if schedule['condition_index']==len(cfg['conditions']):schedule['condition_index']=0;schedule['seed_index']+=1
        if schedule['seed_index']==len(cfg['seeds']):
            schedule['seed_index']=0;schedule['round']+=1;assess=schedule['evaluations'].get(stage,{})
            needed=[f'{condition}_{seed}' for condition in ('adapter','internal','combined') for seed in cfg['seeds']]
            if all(name in assess for name in needed) and all(all(result['episodes']>=cfg['promotion_evaluation_episodes'] and result['rate']>=cfg['promotion_success'] for result in assess[name]['results'].values()) for name in needed):
                if schedule['stage_index']<len(stages)-1:schedule['stage_index']+=1
                else:store.update(state='curriculum_complete',reason='Todas las habilidades superaron la puerta de promoción; falta evaluación final de 30 episodios');raise StopTraining()
        atomic_json(schedule_path,schedule)
    except CalibrationFailed as exc:
        persist();store.update(state='needs_calibration',reason=str(exc),checkpoint=str(latest_checkpoint))
    except (StopTraining,PilotFinished) as exc:
        persist();store.update(state='stopped',reason='Piloto terminado' if isinstance(exc,PilotFinished) else 'Detenido con checkpoint',checkpoint=str(latest_checkpoint))
    except Exception as exc:
        try:persist()
        except Exception:traceback.print_exc()
        store.update(state='error',reason=str(exc),traceback=traceback.format_exc());raise
    finally:keep_system_awake(False);env.close();store.close()

if __name__=='__main__':
    with WorkerLock():
        try:main()
        except (StopTraining,PilotFinished):pass
