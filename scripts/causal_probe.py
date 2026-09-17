"""Paired offline interventions from a full neural checkpoint. No claims about biological causality."""
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from learning_brain import LearningBrain
from learning_adapter import SensoryEncoder,NeuralReadout,ActionCatalog
from learning_policy import make_model,decision
from learning_checkpoint import load_checkpoint
from lab_store import RUNTIME,atomic_json
import gymnasium as gym
class OfflineEnv(gym.Env):
    action_space=gym.spaces.Discrete(len(ActionCatalog().entries));observation_space=gym.spaces.Box(0,np.inf,shape=(3942,),dtype=np.float32)
    def __init__(self):self.readout=NeuralReadout()
def run():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--observation',default=str(RUNTIME/'latest.json'));p.add_argument('--output',default=str(ROOT/'outputs/causal_probe_v2.json'));a=p.parse_args()
    cfg=json.loads((ROOT/'config/learning.json').read_text());env=OfflineEnv();brain=LearningBrain();model=make_model(env,cfg,7);encoder=SensoryEncoder();catalog=ActionCatalog()
    raw=json.loads(Path(a.observation).read_text());obs=raw.get('observation',raw);ids,rates,channels=encoder.encode(obs);mask=catalog.mask(obs);results={}
    for mode in ('normal','repeat','altered_input','silenced_descending'):
        load_checkpoint(a.checkpoint,brain,model,env);brain.plasticity=False
        if mode=='silenced_descending':brain.silence[env.readout.indices]=True
        applied=np.zeros_like(rates) if mode=='altered_input' else rates
        counts,activity=brain.step(ids,applied,50.);features=env.readout.update(counts);_,_,_,detail=decision(model,features,mask,catalog,env.readout.body_ids,True)
        results[mode]={'activity':activity,'probabilities':detail['probabilities'],'selected':detail['selected'],'filtered_feature_norm':float(np.linalg.norm(features))}
    assert results['normal']['activity']['spikes_sha256']==results['repeat']['activity']['spikes_sha256']
    base=np.array(results['normal']['probabilities']);effects={m:float(np.abs(np.array(v['probabilities'])-base).sum()) for m,v in results.items()}
    atomic_json(a.output,{'checkpoint':a.checkpoint,'observation':obs,'conditions':results,'probability_l1_effect':effects,'exact_repeat':True,'scope':'offline neural and adapter response; no game counterfactual or biological causal claim'});print(json.dumps(effects))
if __name__=='__main__':run()
