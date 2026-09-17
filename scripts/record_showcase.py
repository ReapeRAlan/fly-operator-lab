"""Record showcase episodes: game frames plus the synchronized neural activity of every tick.

For each mission it plays the same scenario twice, with the same brain and the same masks:
an untrained actor and a taught actor. Nothing here trains; it only observes and records.
Outputs per run: trace.jsonl (one line per tick), spikes.npz (sparse spikes per tick) and
summary.json with the capture directory the game wrote its frames to.
"""
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from learning_brain import LearningBrain
from learning_env import FlyOperatorEnv
from learning_policy import make_model,decision,actor_hash
from learning_runtime import WorkerLock
from lab_store import atomic_json

STAGE_ACTIONS={'move':{'wait','move','stop','cancel'},'stance':{'wait','crouch'},'orient':{'wait','turn','stop','cancel'},
               'cancel':{'wait','cancel','door_breach'},'switch':{'wait','equip','cancel'}}

def load_actor(model,policy_path):
    """Load only the learned weights of one of our own checkpoints (same catalog and features)."""
    state=torch.load(policy_path,map_location='cpu',weights_only=False)
    model.policy.load_state_dict(state['policy']);return actor_hash(model)

def record(env,model,mission,stage,folder,max_ticks,label):
    folder.mkdir(parents=True,exist_ok=True)
    env.stage=stage;env.split=[m for m in env.scenarios if m['name']==mission][0]['split']
    env.curriculum_allowed=STAGE_ACTIONS[stage]
    x,_=env.reset(options={'mission':mission,'record':True})
    trace=[];indices=[];counts=[];offsets=[0];started=time.time()
    for tick in range(max_ticks):
        mask=env.action_masks()
        action,value,logprob,detail=decision(model,x,mask,env.catalog,env.readout.body_ids,deterministic=True)
        x,reward,terminated,truncated,info=env.step(action)
        spikes=np.flatnonzero(env.last_counts)
        indices.append(spikes.astype(np.int32));counts.append(env.last_counts[spikes].astype(np.int16));offsets.append(offsets[-1]+len(spikes))
        order=np.argsort(detail['probabilities'])[::-1][:5]
        trace.append({'tick':tick,'label':label,'mission':mission,'stage':stage,
          'action':info['action_label'],'action_index':int(action),'command':info['command']['action'],
          'requested':int(detail['selected']),'value':float(detail['value']),'entropy':float(detail['entropy']),
          'top_actions':[{'label':env.catalog.label(int(i)),'p':float(detail['probabilities'][int(i)])} for i in order if mask[int(i)]],
          'reward':float(reward),'success':bool(info['success']),'terminated':bool(terminated),'truncated':bool(truncated),
          'goal_distance':float(info['goal_distance']),'position':list(info['observation']['operator']['position']),
          'aim':list(info['observation']['operator']['aim']),'spikes':int(info['activity']['spikes']),
          'active_neurons':int(info['activity']['active_neurons']),'sim_time_ms':int(info['simulated_ms']),
          'capture_frame':info.get('capture',{}).get('capture_frame'),'decision_required':bool(info['decision_required']),
          'top_neurons':[{'body_id':n['body_id'],'spikes':n['spikes']} for n in info['neurons_readout'][:8]]})
        if terminated or truncated:break
    summary={'label':label,'mission':mission,'stage':stage,'ticks':len(trace),'success':bool(trace[-1]['success']),
             'truncation':info.get('truncation_reason'),'capture_directory':env.raw.get('capture_directory'),
             'actor_sha256':actor_hash(model),'wall_seconds':round(time.time()-started,1),
             'final_goal_distance':trace[-1]['goal_distance'],'episode_uid':env.episode_uid}
    (folder/'trace.jsonl').write_text('\n'.join(json.dumps(t,allow_nan=False) for t in trace),encoding='utf-8')
    np.savez_compressed(folder/'spikes.npz',indices=np.concatenate(indices),counts=np.concatenate(counts),offsets=np.array(offsets,np.int64))
    atomic_json(folder/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--missions',nargs='*',default=['fly_move_validation_1000','fly_move_validation_1001'])
    parser.add_argument('--stage',default='move')
    parser.add_argument('--policy',default='work/learning/campaigns/smoke-v4.1/checkpoints/common_7')
    parser.add_argument('--output',default='work/showcase')
    parser.add_argument('--max-ticks',type=int,default=200)
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config/learning.json').read_text())
    pointer=ROOT/args.policy/'current.json';manifest=json.loads(pointer.read_text())
    policy_path=pointer.parent/str(manifest['generation'])/'policy.pt'
    brain=LearningBrain();brain.plasticity=False
    env=FlyOperatorEnv(brain,seed=args.seed,sensory_schema=cfg.get('sensory_schema'));env.config=cfg;env.train_plasticity=False
    out=ROOT/args.output;runs=[]
    for mission in args.missions:
        torch.manual_seed(args.seed)
        untrained=make_model(env,cfg,args.seed)
        runs.append(record(env,untrained,mission,args.stage,out/f'{mission}__sin_entrenar',args.max_ticks,'SIN ENTRENAR'))
        trained=make_model(env,cfg,args.seed);load_actor(trained,policy_path)
        runs.append(record(env,trained,mission,args.stage,out/f'{mission}__entrenada',args.max_ticks,'DESPUES DE APRENDER'))
    env.close()
    atomic_json(out/'index.json',{'runs':runs,'policy':str(policy_path),'created':time.time()})
    print('done',len(runs),'runs')

if __name__=='__main__':
    with WorkerLock():main()
