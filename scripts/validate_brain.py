"""Full graph dynamics, reproducibility, input intervention and benchmark."""
from pathlib import Path
import sys,json,time,platform
import numpy as np,psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flybrain import FlyBrain,Adapter
def main():
    brain=FlyBrain();a=Adapter();actor={'position':[0,0,0],'aim':[1,0,0],'ammo':30,'capacity':31,'speed':0,'recent_damage':0}
    obs={'operator':actor,'visible_enemies':[{'position':[6,0,2]}]}
    ids,rates,channels=a.encode(obs)
    runs={};arrays={}
    for label in ['normal','repeat','altered','silent']:
        brain.reset(7);series=[];counts=[]
        for i in range(20):
            r=rates if label in ['normal','repeat'] else (rates[::-1].copy() if label=='altered' else np.zeros_like(rates))
            spikes,stats=brain.step(ids,r,50.)
            command,scores=a.decode(spikes,obs,50.)
            series.append({**stats,'command':command,'scores':scores});counts.append(spikes)
        runs[label]=series;arrays[label]=np.stack(counts)
        print(label,'spikes',int(arrays[label].sum()),'wall',sum(x['wall_seconds'] for x in series),flush=True)
    assertions={'same_seed_exact_repeat':bool(np.array_equal(arrays['normal'],arrays['repeat'])),'altered_inputs_change_spikes':not np.array_equal(arrays['normal'],arrays['altered']),'silent_inputs_no_spikes':bool(arrays['silent'].sum()==0),'altered_inputs_change_actions':any(x['command']['action']!=y['command']['action'] for x,y in zip(runs['normal'],runs['altered'])),'all_neurons_loaded':len(brain.ids)==166700,'all_internal_edges_loaded':len(brain.indices)==25582938}
    result={'assertions':assertions,'adapter_channels':channels,'python':platform.python_version(),'rss_mib':psutil.Process().memory_info().rss/2**20,'runs':runs}
    (ROOT/'outputs/brain_benchmark.json').write_text(json.dumps(result,indent=2));print(assertions,flush=True)
    assert all(assertions.values())
if __name__=='__main__':main()
