"""Paired common-random-number validation from every spatial sector to descending neurons."""
from pathlib import Path
import sys,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from learning_adapter import SensoryEncoder,NeuralReadout
from learning_brain import LearningBrain
from lab_store import atomic_json


def run(seed,brain,encoder,readout,rates):
    brain.reset(seed);brain.plasticity=False
    counts,activity=brain.step(encoder.ids,rates,50.)
    dn=counts[readout.indices].copy()
    return counts,dn,activity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--schema',default='data/learning/sensory_ports_v31.json')
    parser.add_argument('--output',default='outputs/sensory_transfer_v3.json');args=parser.parse_args()
    encoder=SensoryEncoder(ROOT/args.schema);readout=NeuralReadout();brain=LearningBrain()
    base=np.full(len(encoder.ids),encoder.config['baseline_hz'],np.float64)
    channel_slices={name:slice(i*16,(i+1)*16) for i,name in enumerate(encoder.channels)}
    seeds=(7,19,43);state_names=['health','ammo','crouched','ready','busy','primary','secondary','flash','frag','tool','orientation_sin','goal_distance','time_remaining','action_in_progress']
    names=list(encoder.channels)
    records=[];signatures={name:np.zeros(len(readout.indices),np.int64) for name in names};started=time.perf_counter()
    ceiling=int(np.ceil(50./(brain.dynamics.refractory_ms+brain.dynamics.dt_ms)))
    saturation=[]
    for seed in seeds:
        all_base,dn_base,base_activity=run(seed,brain,encoder,readout,base)
        saturation.append(float(np.mean(all_base>=ceiling)))
        for name in names:
            # Each channel is driven at its own maximum rate (v3.2 schemas may lower gains per channel).
            rates=base.copy();rates[channel_slices[name]]=encoder.config['baseline_hz']+encoder.gains_hz[names.index(name)]
            all_active,dn_active,activity=run(seed,brain,encoder,readout,rates)
            difference=dn_active.astype(np.int64)-dn_base.astype(np.int64);signatures[name]+=difference
            records.append({'seed':seed,'channel':name,'changed_descending':int(np.count_nonzero(difference)),
                            'descending_l1_spikes':int(np.abs(difference).sum()),'descending_signed_spikes':int(difference.sum()),
                            'network_spikes':int(activity['spikes']),'baseline_network_spikes':int(base_activity['spikes']),
                            'network_saturation_fraction':float(np.mean(all_active>=ceiling)),
                            'signature_sha256':hashlib.sha256(difference.tobytes()).hexdigest()})
            saturation.append(float(np.mean(all_active>=ceiling)))
    goal_summary=[]
    for name in [f'goal_{i}' for i in range(8)]:
        rows=[r for r in records if r['channel']==name]
        goal_summary.append({'channel':name,'changed_descending_total':int(np.count_nonzero(signatures[name])),
                             'l1_total':int(np.abs(signatures[name]).sum()),
                             'changed_each_seed':[r['changed_descending'] for r in rows],
                             'l1_each_seed':[r['descending_l1_spikes'] for r in rows]})
    distinct=len({tuple(np.sign(signatures[f'goal_{i}'])) for i in range(8)})
    criteria={'every_goal_changes_descending':all(x['changed_descending_total']>=8 and min(x['changed_each_seed'])>0 for x in goal_summary),
              'goal_signatures_distinct':distinct==8,
              'all_98_channels_transfer':all(np.count_nonzero(signatures[name])>0 for name in names),
              'hostage_channels_transfer':all(np.count_nonzero(signatures[f'hostage_{i}'])>0 for i in range(8)),
              'enemy_channels_transfer':all(np.count_nonzero(signatures[f'enemy_{i}'])>0 for i in range(16)),
              'operator_state_channels_transfer':all(np.count_nonzero(signatures[name])>0 for name in state_names),
              'network_not_saturated':max(saturation)<.01}
    result={'version':1,'sensory_schema_version':encoder.config['version'],
            'sensory_schema_sha256':hashlib.sha256(encoder.path.read_bytes()).hexdigest(),
            'graph_hash':brain.graph_hash,'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seeds':list(seeds),'duration_ms':50,
            'method':'paired common random numbers; one active semantic channel versus baseline, full classified graph',
            'neurons':brain.n,'edges':brain.edges,'descending_neurons':len(readout.indices),'goal_summary':goal_summary,
            'distinct_goal_signatures':distinct,'validated_operator_state_channels':state_names,'max_network_saturation_fraction':max(saturation),
            'criteria':criteria,'passed':all(criteria.values()),'wall_seconds':time.perf_counter()-started,'records':records}
    path=ROOT/args.output;atomic_json(path,result)
    print(json.dumps({'path':str(path),'passed':result['passed'],'schema':result['sensory_schema_version'],
      'goal_signatures':distinct,'changed_descending_range':[min(v for x in goal_summary for v in x['changed_each_seed']),max(v for x in goal_summary for v in x['changed_each_seed'])],
      'max_network_saturation_fraction':max(saturation),'wall_seconds':result['wall_seconds']}))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
