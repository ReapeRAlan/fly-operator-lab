"""Compare v3.1 and v3.2 imitation on replayed descending features with episode-grouped CV.

Input: work/probes/decodability_<schema>_move.npz from probe_decodability.py.
Output: outputs/imitation_offline_<schema>.json and an actor fitted on all examples
(work/probes/offline_actor_<schema>.pt) used by calibrate_motor_plasticity.py.
"""
from pathlib import Path
import sys,json,argparse,copy,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch,gymnasium as gym
from learning_adapter import ActionCatalog
from learning_policy import make_model,imitate
from offline_replay import within_tolerance
from lab_store import atomic_json

class FeatureEnv(gym.Env):
    def __init__(self,dimension,actions):
        self.observation_space=gym.spaces.Box(-5,5,shape=(dimension,),dtype=np.float32);self.action_space=gym.spaces.Discrete(actions)
    def reset(self,seed=None,options=None):return np.zeros(self.observation_space.shape,np.float32),{}
    def step(self,action):return np.zeros(self.observation_space.shape,np.float32),0,False,False,{}

def score(model,x,y,masks,catalog):
    with torch.no_grad():
        logits=model.policy.action_net(torch.from_numpy(x)).masked_fill(~torch.from_numpy(masks),-1e8);prediction=logits.argmax(1).numpy()
    return int(np.sum(prediction==y)),int(sum(within_tolerance(catalog,p,t) for p,t in zip(prediction,y)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--schema',default='v32');parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--settings',default=None,help='JSON imitation settings overriding config')
    args=parser.parse_args();started=time.time()
    cfg=json.loads((ROOT/'config/learning.json').read_text());catalog=ActionCatalog()
    settings=json.loads(args.settings) if args.settings else cfg.get('imitation') or {'learning_rate':.01,'weight_decay':1e-4,'max_epochs':300,'patience':30,'validation_fraction':.2,'angular_label_smoothing':.2,'min_epochs':5}
    data=np.load(ROOT/f'work/probes/decodability_{args.schema}_move.npz')
    x=data['features'].astype(np.float32);y=data['labels'].astype(int);masks=data['masks'];groups=data['groups']
    unique=np.unique(groups);order=np.random.default_rng(0).permutation(len(unique));fold_of={g:i%args.folds for i,g in zip(order,unique)}
    fold=np.array([fold_of[g] for g in groups]);env=FeatureEnv(x.shape[1],len(catalog.entries))
    totals={'v31':[0,0],'v32':[0,0]};epochs=[]
    for k in range(args.folds):
        train=np.where(fold!=k)[0];test=np.where(fold==k)[0]
        examples=[(x[i],int(y[i]),masks[i],'move',int(groups[i])) for i in train]
        for version in ('v31','v32'):
            torch.manual_seed(k);model=make_model(env,cfg,7)
            if version=='v31':
                # v3.1 refit all accumulated examples for 20 epochs after each of ~12 teaching episodes.
                for _ in range(12):imitate(model,[e[:4] for e in examples],cfg['imitation_epochs'])
            else:
                result=imitate(model,examples,settings=settings,catalog=catalog);epochs.append(result['epochs'])
            exact,within=score(model,x[test],y[test],masks[test],catalog);totals[version][0]+=exact;totals[version][1]+=within
    ceiling=json.loads((ROOT/f'outputs/decodability_probe_{args.schema}.json').read_text())['descending_to_action']
    n=len(y);torch.manual_seed(0);final=make_model(env,cfg,7)
    # The calibration actor uses whichever procedure generalized better in grouped CV.
    winner='v32' if totals['v32'][1]>totals['v31'][1] else 'v31';everything=[(x[i],int(y[i]),masks[i],'move',int(groups[i])) for i in range(n)]
    if winner=='v32':final_result=imitate(final,everything,settings=settings,catalog=catalog)
    else:
        for _ in range(12):final_result=imitate(final,[e[:4] for e in everything],cfg['imitation_epochs'])
        final_result.setdefault('within_tolerance',None)
    final_result['procedure']=winner
    torch.save({'action_net':final.policy.action_net.state_dict(),'schema':args.schema,'settings':settings if winner=='v32' else 'v31','procedure':winner},ROOT/f'work/probes/offline_actor_{args.schema}.pt')
    result={'version':1,'schema':args.schema,'samples':n,'episodes':int(len(unique)),'folds':args.folds,'settings':settings,
            'v31_imitation':{'exact':totals['v31'][0]/n,'within_45deg':totals['v31'][1]/n,'procedure':'12 successive 20-epoch fits with the PPO optimizer (lr 3e-4)'},
            'v32_imitation':{'exact':totals['v32'][0]/n,'within_45deg':totals['v32'][1]/n,'selected_epochs':epochs},
            'ridge_ceiling':ceiling,'criterion':'v3.2 within_45deg >= 0.9 x ridge ceiling within_45deg',
            'passed':totals['v32'][1]/n>=.9*ceiling['within_45deg'],'final_fit':{k:final_result.get(k) for k in ('procedure','examples','epochs','accuracy','within_tolerance')},
            'wall_seconds':time.time()-started}
    atomic_json(ROOT/f'outputs/imitation_offline_{args.schema}.json',result)
    print(json.dumps({k:result[k] for k in ('samples','episodes','v31_imitation','v32_imitation','passed','final_fit')},indent=2))
    print('ridge ceiling',ceiling['exact'],ceiling['within_45deg'])

if __name__=='__main__':main()
