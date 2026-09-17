"""Independent held-out evaluation. Stop the worker through the panel before using its game."""
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from learning_runtime import WorkerLock,ResourceGuard,LiveState
from learning_brain import LearningBrain
from learning_env import FlyOperatorEnv
from learning_policy import make_model,decision
from learning_checkpoint import load_checkpoint
from lab_store import Store,RUNTIME,atomic_json

def run():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--condition',choices=['frozen','adapter','internal','combined'],required=True);p.add_argument('--seed',type=int,choices=[7,19,43],required=True);p.add_argument('--stage',default='move');p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--episodes',type=int,default=30);p.add_argument('--record-first',action='store_true');a=p.parse_args()
    cfg=json.loads((ROOT/'config/learning.json').read_text());store=Store('heldout-'+str(int(time.time())));guard=ResourceGuard(store,cfg)
    b=LearningBrain();env=FlyOperatorEnv(b,condition=a.condition,seed=a.seed,stage=a.stage,split=a.split,guard=guard,on_episode=store.episode);model=make_model(env,cfg,a.seed)
    counters=load_checkpoint(a.checkpoint,b,model,env);env.train_plasticity=False;live=LiveState(b,store.run);rows=[]
    missions=[m for m in env.scenarios if m['stage']==a.stage and m['split']==a.split]
    if a.episodes>30 or a.episodes<1:raise ValueError('1..30 episodes per seed')
    for i in range(a.episodes):
        mission=missions[i+(cfg['seeds'].index(a.seed)*30 if a.split=='validation' else 0)]
        x,_=env.reset(options={'mission':mission['name'],'record':a.record_first and i==0})
        while True:
            action,_,_,detail=decision(model,x,env.action_masks(),env.catalog,env.readout.body_ids,True);live.begin()
            x,r,t,u,info=env.step(action);live.end(env.last_counts,a.condition,a.seed);info.update(decision=detail,phase='evaluation',training_steps=counters.get('steps',0));store.step(info)
            store.update(state='running',condition=a.condition,seed=a.seed,stage=a.stage,phase='evaluation',episode=i+1,simulated_ms=env.total_simulated_ms)
            if t or u:rows.append(info);break
    result={'condition':a.condition,'seed':a.seed,'stage':a.stage,'split':a.split,'checkpoint':a.checkpoint,'outcomes':rows,'successes':sum(r['success'] for r in rows),'episodes':len(rows),'instructor':False}
    path=ROOT/'outputs'/f'evaluation_{a.condition}_{a.seed}_{a.stage}_{int(time.time())}.json';atomic_json(path,result);store.update(state='stopped',reason='Evaluación terminada',report=str(path));env.close();store.close();print(path)
if __name__=='__main__':
    with WorkerLock():run()
