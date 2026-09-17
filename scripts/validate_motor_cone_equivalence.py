"""Real-graph proof that motor-cone eligibility tracking is bit-identical and faster.

Loads one hash-verified v3.1 plastic checkpoint, then integrates the same recorded
sensory rates twice from that exact state: whole-graph eligibility (legacy) and
eligibility restricted to edges entering the 1,314 descending neurons.
"""
from pathlib import Path
import sys,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,psutil
from learning_brain import LearningBrain
from learning_adapter import SensoryEncoder,NeuralReadout
from lab_store import atomic_json,file_hash

DYNAMIC=('v','g','refractory','ring','ring_counts','gains','pretrace','posttrace','prelast','postlast','silence')

def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()

def run(brain,checkpoint,ids,rates,dn,ticks):
    brain.load(checkpoint);brain.plasticity=True
    credit=np.linspace(-1,1,len(dn)).astype(np.float32);records=[];walls=[]
    for tick in range(ticks):
        counts,activity=brain.step(ids,rates,50.);walls.append(activity['wall_seconds'])
        snapshot=brain.capture_motor_eligibility(dn)
        update=brain.reward_motor(snapshot,.5*(-1)**tick,dn,credit)
        records.append({'counts':digest(counts),'snapshot':digest(snapshot['edges'],snapshot['posts'],snapshot['values']),
                        'candidate_edges':int(len(snapshot['edges'])),'changed_edges':update['changed_edges'],
                        'eligible_edges_tracked':activity['eligible_edges'],'spikes':activity['spikes']})
    motor_edges=brain.plastic_post[np.asarray(brain.indices,dtype=np.int64)] if brain.restricted_plasticity else None
    return records,walls,{name:digest(getattr(brain,name)) for name in DYNAMIC},motor_edges

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='work/learning/campaigns/pilot-v3.1-curriculum/checkpoints/internal_7/current.json')
    parser.add_argument('--ticks',type=int,default=10);args=parser.parse_args()
    started=time.time();pointer=ROOT/args.checkpoint;manifest=json.loads(pointer.read_text())
    brain_path=pointer.parent/str(manifest['generation'])/'brain.npz'
    if file_hash(brain_path)!=manifest['brain_sha256']:raise ValueError('Checkpoint brain hash mismatch')
    latest=json.loads((ROOT/'work/learning/latest.json').read_text(encoding='utf-8'))
    encoder=SensoryEncoder();readout=NeuralReadout();brain=LearningBrain()
    values=np.array([latest['channels'][name] for name in encoder.channels],float)
    rates=np.repeat(encoder.config['baseline_hz']+encoder.config['gain_hz']*values,16)
    dn=readout.indices

    full_records,full_walls,full_state,_=run(brain,brain_path,encoder.ids,rates,dn,args.ticks)
    full_elig={name:np.asarray(getattr(brain,name)).copy() for name in ('eligibility','eligibility_last','listed')}
    full_active=int(brain.active_count)
    brain.set_plastic_posts(dn)
    restricted_records,restricted_walls,restricted_state,motor_edges=run(brain,brain_path,encoder.ids,rates,dn,args.ticks)
    eligibility_equal={name:bool(np.array_equal(full_elig[name][motor_edges],getattr(brain,name)[motor_edges])) for name in full_elig}
    del full_elig

    comparable=lambda records:[{k:v for k,v in r.items() if k!='eligible_edges_tracked'} for r in records]
    checks={'dynamic_state_equal':{name:full_state[name]==restricted_state[name] for name in DYNAMIC},
            'motor_eligibility_equal':eligibility_equal,
            'per_tick_counts_snapshots_updates_equal':comparable(full_records)==comparable(restricted_records)}
    passed=all(checks['dynamic_state_equal'].values()) and all(eligibility_equal.values()) and checks['per_tick_counts_snapshots_updates_equal']
    # First tick of each mode includes numba specialization; report warm medians.
    full_median=float(np.median(full_walls[1:]));restricted_median=float(np.median(restricted_walls[1:]))
    result={'version':1,'passed':passed,'checkpoint':str(pointer.relative_to(ROOT)),'brain_sha256':manifest['brain_sha256'],
            'graph_hash':brain.graph_hash,'neurons':brain.n,'edges':brain.edges,'motor_target_neurons':int(len(dn)),
            'motor_cone_edges':int(motor_edges.sum()),'ticks':args.ticks,'rates_source':'work/learning/latest.json channels',
            'checks':checks,'full_active_edges_end':full_active,'restricted_active_edges_end':int(brain.active_count),
            'full_wall_seconds':full_walls,'restricted_wall_seconds':restricted_walls,
            'full_warm_median_seconds':full_median,'restricted_warm_median_seconds':restricted_median,
            'speedup_warm_median':full_median/restricted_median,'per_tick':restricted_records,
            'rss_mib':psutil.Process().memory_info().rss/2**20,'wall_seconds':time.time()-started,
            'meaning':'Equality covers membrane, synaptic, refractory, delay-queue, trace, gain arrays, per-tick spike counts, motor eligibility snapshots and reward_motor updates.'}
    atomic_json(ROOT/'outputs/motor_cone_equivalence.json',result)
    print(json.dumps({k:result[k] for k in ('passed','motor_cone_edges','full_warm_median_seconds','restricted_warm_median_seconds','speedup_warm_median','full_active_edges_end','restricted_active_edges_end')},indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
