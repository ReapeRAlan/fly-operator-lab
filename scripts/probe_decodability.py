"""How much of the instructor's choice is linearly decodable from sensors versus descending neurons.

Replays recorded v3.1 episodes open-loop through the complete brain with a chosen
sensory schema. Reports episode-grouped, nested ridge CV (exact and within 45°),
plus network activity. Features are saved for imitation/plasticity calibration.
"""
from pathlib import Path
import sys,json,time,argparse,os
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np

def worker(schema,episodes):
    sys.path.insert(0,str(ROOT/'src'))
    from learning_brain import LearningBrain
    from learning_adapter import SensoryEncoder,NeuralReadout,ActionCatalog
    from offline_replay import replay_episode
    brain=LearningBrain();encoder=SensoryEncoder(ROOT/schema);readout=NeuralReadout();catalog=ActionCatalog();out=[]
    for index,episode in episodes:
        rows,stats=replay_episode(brain,encoder,readout,catalog,episode,int(episode['trajectory_sha256'][:8],16))
        out.append((index,rows,stats))
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--schema',default='data/learning/sensory_ports_v31.json')
    parser.add_argument('--run',default='pilot-v3.1-curriculum');parser.add_argument('--stage',default='move')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--max-episodes',type=int,default=0)
    args=parser.parse_args();started=time.time()
    from offline_replay import load_episodes,grouped_ridge_cv
    from learning_adapter import ActionCatalog
    episodes=load_episodes(args.run,args.stage)
    if args.max_episodes:episodes=episodes[:args.max_episodes]
    # Balance workers by tick count.
    buckets=[[] for _ in range(args.workers)];loads=[0]*args.workers
    for index,episode in sorted(enumerate(episodes),key=lambda item:-len(item[1]['steps'])):
        slot=loads.index(min(loads));buckets[slot].append((index,episode));loads[slot]+=len(episode['steps'])
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(worker,[args.schema]*args.workers,buckets):results.extend(part)
    results.sort(key=lambda r:r[0])
    rows=[(index,row) for index,episode_rows,_ in results for row in episode_rows]
    x=np.stack([r['features'] for _,r in rows]);c=np.stack([r['channels'] for _,r in rows])
    y=np.array([r['label'] for _,r in rows]);m=np.stack([r['mask'] for _,r in rows]);g=np.array([i for i,_ in rows])
    catalog=ActionCatalog();ticks=np.array([s['ticks'] for _,_,s in results])
    weighted=lambda key:float(np.average([s[key] for _,_,s in results],weights=ticks))
    name=Path(args.schema).stem.replace('sensory_ports_','')
    result={'version':1,'schema':args.schema,'run':args.run,'stage':args.stage,'unique_episodes':len(episodes),'ticks':int(ticks.sum()),
            'decisions':int(len(y)),'label_counts':{catalog.label(int(k)):int(v) for k,v in zip(*np.unique(y,return_counts=True))},
            'sensors_to_action':grouped_ridge_cv(c,y,m,g,catalog),'descending_to_action':grouped_ridge_cv(x,y,m,g,catalog),
            'descending_100ms_filter_to_action':grouped_ridge_cv(x[:,:1314],y,m,g,catalog),
            'activity':{'dn_near_ceiling_fraction':weighted('dn_near_ceiling_fraction'),'active_neurons_mean':weighted('active_neurons_mean'),
                        'spikes_per_50ms_mean':weighted('spikes_mean')},
            'wall_seconds':time.time()-started,
            'method':'Open-loop replay of recorded observations; labels from ExerciseTeacher on the pre-decision observation and recorded mask; episode-grouped 5-fold ridge CV with inner grouped CV selecting the penalty; duplicate trajectories removed.'}
    features=ROOT/'work/probes';features.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(features/f'decodability_{name}_{args.stage}.npz',features=x.astype(np.float32),channels=c,labels=y,masks=m,groups=g)
    from lab_store import atomic_json
    atomic_json(ROOT/f'outputs/decodability_probe_{name}.json',result)
    print(json.dumps({k:result[k] for k in ('decisions','unique_episodes','ticks','sensors_to_action','descending_to_action','descending_100ms_filter_to_action','activity','wall_seconds')},indent=2))

if __name__=='__main__':main()
