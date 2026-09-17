"""Single-game, CPU pilot. Run from .venv-learning. See outputs/INFORME_APRENDIZAJE.md."""
from pathlib import Path
import sys,os,json,time,random,traceback,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
runtime_threads=str(json.loads((ROOT/'config/learning.json').read_text()).get('max_cpu_threads',2))
os.environ['OMP_NUM_THREADS']=runtime_threads;os.environ['MKL_NUM_THREADS']=runtime_threads
import numpy as np,torch
from learning_brain import LearningBrain
from learning_env import FlyOperatorEnv,ExerciseTeacher
from learning_policy import make_model,decision,imitate,internal_update,actor_hash
from learning_checkpoint import save_checkpoint,load_checkpoint,load_teaching_examples
from learning_runtime import ResourceGuard,WorkerLock,StopTraining,PilotFinished,LiveState
from lab_store import Store,RUNTIME,atomic_json,file_hash
from windows_session import keep_system_awake

REQUIRED={'move':'move','orient':'turn','switch':'equip','shoot':'fire','reload':'reload','door':'door_open','breach':'door_breach','grenade':'throw','elimination':'fire','rescue':'follow','defuse':'defuse','loadout':'loadout'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=24);parser.add_argument('--smoke-steps',type=int,default=0);args=parser.parse_args()
    cfg=json.loads((ROOT/'config/learning.json').read_text());store=Store(cfg.get('experiment_id','pilot-v2'));guard=ResourceGuard(store,cfg,args.hours)
    schedule_path=RUNTIME/'schedule.json';schedule=json.loads(schedule_path.read_text()) if schedule_path.exists() else {'version':2,'seed_index':0,'condition_index':0,'round':0,'stage_index':0,'prepared_seeds':[],'evaluations':{}}
    cap=json.loads((ROOT/'outputs/native_capabilities_v2.json').read_text());stages=[s for s in cfg['stage_order'] if REQUIRED[s] in cap['validated_actions']]
    store.update(state='preparing',reason='Cargando MaleCNS completo',enabled_stages=stages,pending_stages=[s for s in cfg['stage_order'] if s not in stages],pilot_hours=args.hours)
    guard();brain=LearningBrain();env=FlyOperatorEnv(brain,guard=guard);env.config=cfg;live=LiveState(brain)
    model=None;counters={};current_name=None;latest_checkpoint=None;examples=[];checkpoint_ready=False
    def persist():
        nonlocal latest_checkpoint
        if model is not None and current_name and checkpoint_ready:
            store.update(state='saving',reason='Guardando actividad, pesos, filtros y optimizadores')
            latest_checkpoint=save_checkpoint(current_name,brain,model,env,counters,store,teaching_examples=examples if env.condition=='common_teaching' else None)
            atomic_json(schedule_path,schedule)
    guard.on_pause=persist
    def reset(stage,split='train',mission=None):
        env.stage=stage;env.split=split;live.begin()
        try:return env.reset(options={'mission':mission} if mission else {})[0]
        finally:live.end(getattr(env,'last_counts',np.zeros(brain.n,np.int32)),env.condition,env.seed_base)
    def advance(x,mode,instructor=False,collect=False,deterministic=False):
        guard();mask=env.action_masks();action,value,logprob,detail=decision(model,x,mask,env.catalog,env.readout.body_ids,deterministic)
        teacher=ExerciseTeacher(env.catalog).choose(env) if collect or instructor else None
        if collect:examples.append((x.copy(),teacher,mask.copy()))
        if instructor:action=teacher
        detail['executed']=action
        detail['contributions_for_action']=detail['selected']
        detail['controller']='instructor' if instructor else 'neuronal_actor'
        live.begin()
        try:
            next_x,reward,term,trunc,info=env.step(action)
            update=internal_update(model,brain,x,next_x,reward,term,cfg['ppo']['gamma']) if mode=='internal' else {'changed_edges':0}
            info.update(decision=detail,learning=update,phase=mode,training_steps=counters.get('steps',0),actor_sha256=actor_hash(model))
            store.step(info)
            if guard.probe:
                # Detailed reproducible window starts at the next step, with a checkpoint.
                guard.probe=False;persist();env.neural_record=True
            elif env.neural_record:
                events=brain.last_events
                if events is not None:
                    path=RUNTIME/'probes';path.mkdir(exist_ok=True)
                    np.savez_compressed(path/f'{env.condition}_{env.seed_base}_{info["episode"]}_{info["sequence"]}.npz',indices=events[0],ticks=events[1],counts=env.last_counts)
                env.neural_record=False
        finally:live.end(getattr(env,'last_counts',np.zeros(brain.n,np.int32)),env.condition,env.seed_base)
        guard.after_step()
        store.update(condition=env.condition,seed=env.seed_base,stage=env.stage,phase=mode,episode=info['episode'],simulated_ms=env.total_simulated_ms,
          steps=counters.get('steps',0),neural_wall_seconds=info['activity']['wall_seconds'],last_reward=reward,graph_hash=brain.graph_hash)
        return next_x,reward,term,trunc,info,action,value,logprob,mask
    env.on_episode=store.episode
    try:
        while True:
            for seed in cfg['seeds']:
                if seed in schedule['prepared_seeds']:continue
                env.seed_base=seed;env.condition='common_teaching';env.train_plasticity=False;current_name=f'common_{seed}'
                checkpoint_ready=False
                model=make_model(env,cfg,seed);brain.gains.fill(1);counters={'guided':0,'dagger':0,'steps':0};examples=[]
                point=RUNTIME/'checkpoints'/current_name/'current.json';example_path=RUNTIME/f'examples_{seed}.npz'
                if point.exists():
                    counters=load_checkpoint(point,brain,model,env)
                    store.update(resumed_from=str(point),restored_actor_sha256=actor_hash(model),restored_steps=counters.get('steps',0))
                examples=load_teaching_examples(point,example_path)
                checkpoint_ready=True
                store.update(restored_examples_count=len(examples))
                for phase,total in [('guided',cfg['guided_episodes']),('dagger',cfg['dagger_episodes'])]:
                    while counters[phase]<total:
                        stage=stages[counters[phase]%len(stages)];x=reset(stage);examples_before_episode=len(examples)
                        while True:
                            x,r,t,u,info,*_=advance(x,'teaching',instructor=phase=='guided',collect=True);counters['steps']+=1
                            if args.smoke_steps and counters['steps']>=args.smoke_steps:raise StopTraining()
                            if t or u:break
                        if phase=='guided' and not info['success']:del examples[examples_before_episode:]
                        counters[phase]+=1;old_hash=actor_hash(model);metrics=imitate(model,examples,cfg['imitation_epochs'])
                        store.update(imitation=metrics,actor_before=old_hash,actor_after=actor_hash(model))
                        persist()
                schedule['prepared_seeds'].append(seed);persist()
            # Equal simulated decision budgets: one rollout per seed and condition per round.
            si=schedule['seed_index'];ci=schedule['condition_index'];seed=cfg['seeds'][si];condition=cfg['conditions'][ci]
            checkpoint_ready=False
            env.seed_base=seed;env.condition=condition;model=make_model(env,cfg,seed);current_name=f'{condition}_{seed}'
            point=RUNTIME/'checkpoints'/current_name/'current.json'
            restored_point=point if point.exists() else RUNTIME/'checkpoints'/f'common_{seed}'/'current.json'
            counters=load_checkpoint(restored_point,brain,model,env)
            checkpoint_ready=True
            store.update(resumed_from=str(restored_point),restored_actor_sha256=actor_hash(model),restored_steps=counters.get('steps',0))
            if not point.exists():counters={'steps':0,'rollouts':0,'evaluated_at':0}
            mode='adapter' if condition=='adapter' or (condition=='combined' and counters['rollouts']%2==0) else 'internal' if condition in ('internal','combined') else 'frozen'
            env.train_plasticity=mode=='internal';stage=stages[min(schedule['stage_index'],len(stages)-1)]
            # Exactly one of every four training episodes revisits previous stages.
            def training_stage():
                index=stages.index(stage)
                return stages[(env.episode_count//4)%index] if index and env.episode_count%4==0 else stage
            x=reset(training_stage());buf=model.rollout_buffer;buf.reset();episode_start=True;actor_before=actor_hash(model)
            for step in range(cfg['ppo']['n_steps']):
                env.truncate_next=step==cfg['ppo']['n_steps']-1
                previous=x.copy();x,r,t,u,info,action,value,logprob,mask=advance(x,mode)
                if mode=='adapter':
                    buffered=r
                    if u:
                        with torch.no_grad():buffered+=cfg['ppo']['gamma']*float(model.policy.predict_values(torch.as_tensor(x[None,:]))[0])
                    buf.add(previous[None,:],np.array([[action]]),np.array([buffered]),np.array([episode_start]),value,logprob,action_masks=mask[None,:])
                counters['steps']+=1;model.num_timesteps+=1;episode_start=t or u
                if episode_start and step<cfg['ppo']['n_steps']-1:x=reset(training_stage())
            if mode=='adapter':
                with torch.no_grad():last=model.policy.predict_values(torch.as_tensor(x[None,:]))
                buf.compute_returns_and_advantage(last_values=last,dones=np.array([episode_start]));model.train()
            if mode in ('internal','frozen') and actor_hash(model)!=actor_before:raise AssertionError('Frozen actor changed')
            counters['rollouts']+=1;persist()
            if counters['steps']-counters['evaluated_at']>=cfg['evaluation_interval_steps']:
                env.train_plasticity=False;results={};saved_rng=(torch.get_rng_state(),np.random.get_state(),random.getstate())
                for test_stage in stages[:stages.index(stage)+1]:
                    outcomes=[]
                    candidates=[m for m in env.scenarios if m['stage']==test_stage and m['split']=='validation']
                    for i in range(cfg['evaluation_episodes_per_seed']):
                        mission=candidates[cfg['seeds'].index(seed)*30+i]['name'];x=reset(test_stage,'validation',mission)
                        while True:
                            x,r,t,u,info,*_=advance(x,'evaluation',deterministic=True)
                            if t or u:outcomes.append(info['success']);break
                    results[test_stage]={'successes':sum(outcomes),'episodes':len(outcomes),'rate':sum(outcomes)/len(outcomes)}
                torch.set_rng_state(saved_rng[0]);np.random.set_state(saved_rng[1]);random.setstate(saved_rng[2])
                counters['evaluated_at']=counters['steps'];schedule['evaluations'][current_name]={'stage':stage,'results':results};persist()
            schedule['condition_index']+=1
            if schedule['condition_index']==len(cfg['conditions']):schedule['condition_index']=0;schedule['seed_index']+=1
            if schedule['seed_index']==len(cfg['seeds']):
                schedule['seed_index']=0;schedule['round']+=1
                assessments=[a for key,a in schedule['evaluations'].items() if not key.startswith('frozen_')]
                if len(assessments)==9 and all(a['stage']==stage and all(r['episodes']>=30 and r['rate']>=cfg['promotion_success'] for r in a['results'].values()) for a in assessments):
                    if schedule['stage_index']<len(stages)-1:schedule['stage_index']+=1;schedule['evaluations']={}
            atomic_json(schedule_path,schedule)
    except (StopTraining,PilotFinished) as e:
        persist();store.update(state='stopped',reason='Piloto terminado' if isinstance(e,PilotFinished) else 'Detenido con checkpoint',checkpoint=str(latest_checkpoint))
    except Exception as e:
        try:persist()
        except Exception:traceback.print_exc()
        store.update(state='error',reason=str(e),traceback=traceback.format_exc());raise
    finally:keep_system_awake(False);env.close();store.close()

if __name__=='__main__':
    with WorkerLock():
        try:main()
        except (StopTraining,PilotFinished):
            store=Store(json.loads((ROOT/'config/learning.json').read_text()).get('experiment_id','pilot-v2'));store.update(state='stopped',reason='Detenido antes de inicializar el entrenamiento');store.close()
        except Exception as exc:
            store=Store(json.loads((ROOT/'config/learning.json').read_text()).get('experiment_id','pilot-v2'));store.update(state='error',reason=str(exc),traceback=traceback.format_exc());store.close();raise
