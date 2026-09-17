"""Build sensory schema v3.2: same 98 channels and gateway ranking, task-relevant allocation and tuned encoding.

Changes versus v3.1 (all engineered interface choices, recorded in the schema):
- allocation order: goal direction/proximity receive the strongest visual routes and
  dynamic body state the strongest body routes (v3.1 allocated in channel order, so the
  goal received ranks 769-896 while constant health/ammo received the strongest routes);
- sector channels use von Mises tuning instead of a single bin;
- goal direction stays salient at the goal and goal_distance encodes proximity;
- slowly varying state channels use a lower rate gain to reduce tonic drive.
"""
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from learning_adapter import SensoryEncoder,assign_ports,STAGES
from lab_store import atomic_json

CHANNELS=SensoryEncoder.channels
VISUAL_PRIORITY=([f'goal_{i}' for i in range(8)]+['goal_bearing_sin','goal_bearing_cos','goal_distance']+
                 [f'enemy_{i}' for i in range(16)]+[f'door_{i}' for i in range(8)]+[f'wall_{i}' for i in range(8)]+
                 [f'hostage_{i}' for i in range(8)]+[f'civilian_{i}' for i in range(8)]+[f'friend_{i}' for i in range(8)])
BODY_PRIORITY=(['busy','action_in_progress','ready','speed','crouched','damage','ammo','orientation_sin','orientation_cos',
                'health','time_remaining','primary','secondary','flash','frag','tool']+['task_'+s for s in STAGES]+['tonic'])
LOW_GAIN=('health','ammo','primary','secondary','flash','frag','tool','time_remaining','tonic')

def encoding(kappa,proximity,low_gain,task_gain):
    gains={k:low_gain for k in LOW_GAIN};gains.update({'task_'+s:task_gain for s in STAGES})
    return {'version':'3.2','sector_kappa':kappa,'goal_salience':1.0,'goal_proximity_m':proximity,'channel_gain_hz':gains}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='data/learning/sensory_ports_v32.json')
    parser.add_argument('--kappa',type=float,default=2.0);parser.add_argument('--proximity-m',type=float,default=1.5)
    parser.add_argument('--low-gain-hz',type=float,default=30.);parser.add_argument('--task-gain-hz',type=float,default=60.)
    parser.add_argument('--keep-v31-allocation',action='store_true',help='Ablation: v3.1 port allocation with v3.2 encoding')
    parser.add_argument('--verify-v31',action='store_true');args=parser.parse_args()
    order=VISUAL_PRIORITY+BODY_PRIORITY
    if sorted(order)!=sorted(CHANNELS):raise ValueError('Priority lists must cover every channel exactly once')
    if args.verify_v31:
        v31=json.loads((ROOT/'data/learning/sensory_ports_v31.json').read_text())
        regenerated=assign_ports(CHANNELS,CHANNELS)
        same=all(regenerated[k]['indices']==v31['mapping'][k]['indices'] for k in CHANNELS)
        print(json.dumps({'v31_allocation_reproduced':same}));
        if not same:raise SystemExit(1)
    allocation=list(CHANNELS) if args.keep_v31_allocation else order
    mapping=assign_ports(CHANNELS,allocation)
    ids=[i for k in CHANNELS for i in mapping[k]['indices']]
    if len(set(ids))!=len(ids):raise ValueError('Gateway neurons must be unique')
    config={'version':'3.2','mapping':mapping,'allocation_order':allocation,
            'encoding':encoding(args.kappa,args.proximity_m,args.low_gain_hz,args.task_gain_hz),
            'meaning':'Fixed engineered semantic channels injected through unique excitatory gateway neurons ranked by one/two-hop anatomical routes to descending neurons. Allocation order gives task-relevant channels stronger routes; sector channels use von Mises tuning; goal_distance encodes proximity exp(-d/goal_proximity_m). These are not biological weapon/action concepts. All classified neurons and edges remain in the simulation.',
            'port_selection':'published connectivity counts define ranking; transmitter sign prediction selects excitatory gateways; semantic assignment follows allocation_order',
            'baseline_hz':5.,'gain_hz':145.}
    atomic_json(ROOT/args.output,config)
    route=lambda keys:round(sum(sum(mapping[k]['descending_route_score']) for k in keys)/(16*len(keys)),1)
    print(json.dumps({'output':args.output,'sha256':hashlib.sha256((ROOT/args.output).read_bytes()).hexdigest(),
                      'mean_route_goal_sectors':route([f'goal_{i}' for i in range(8)]),'mean_route_enemy':route([f'enemy_{i}' for i in range(16)]),
                      'mean_route_health':route(['health']),'mean_route_busy':route(['busy'])},indent=2))

if __name__=='__main__':main()
