import json,sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from learning_adapter import SensoryEncoder,ActionCatalog,NeuralReadout,clean_observation,decision_required
from learning_brain import LearningBrain
from learning_curriculum import add_demonstration,forced_wait,train_adapter_episode
from learning_policy import make_model,actor_hash


def raw_observation(receipt=None,sim_time=50):
    return {'operator':{'id':1,'position':[0,0,0],'aim':[1,0,0],'health':75,'ammo':15,'capacity':30,
                        'weapon_state':5,'commands':0,'speed':0,'crouched':True},
            'humans':[{'id':2,'kind':2,'position':[2,0,1],'health':100}],
            'visible_ids':[2],'visible_enemies':[],
            'inventory':[{'slot':1,'equipped':True,'quantity':1,'type':1}],
            'objects':[],'sim_time_ms':sim_time,'action_receipt':receipt or {}}


def mission(stage='move'):
    return {'goal':[4,0,2],'goal_angle':0.,'stage':stage,'max_seconds':30,'geometry':[]}


def test_pending_move_is_a_choice_between_continuing_and_intervening():
    """Protocol 3.3: stopping is legal while the engine moves, far from the goal as well as near it."""
    catalog=ActionCatalog();obs=clean_observation(raw_observation({'status':'in_progress','action':'move'}),mission())
    mask=catalog.mask(obs)
    assert not forced_wait(mask) and decision_required(obs,mask)
    assert {catalog.entries[i]['action'] for i in np.flatnonzero(mask)}=={'wait','stop','cancel'}
    obs['goal']=[.1,0,0];near=catalog.mask(obs)
    assert np.array_equal(near,mask) # the mask no longer reveals when to stop
    start=clean_observation(raw_observation(None,sim_time=0),mission())
    start_mask=catalog.mask(start)
    assert forced_wait(start_mask) and not decision_required(start,start_mask)


def test_v3_encoder_exposes_people_goal_time_and_pending(tmp_path):
    mapping={};cursor=0
    for channel in SensoryEncoder.channels:
        ids=list(range(cursor,cursor+16));cursor+=16
        mapping[channel]={'indices':ids,'body_ids':ids}
    path=tmp_path/'ports.json';path.write_text(json.dumps({'version':3,'mapping':mapping,'baseline_hz':5.,'gain_hz':145.}))
    encoder=SensoryEncoder(path);obs=clean_observation(raw_observation({'status':'in_progress','action':'move'}),mission())
    ids,rates,channels=encoder.encode(obs)
    assert len(ids)==len(SensoryEncoder.channels)*16 and len(rates)==len(ids)
    assert channels['health']==.75 and channels['ammo']==.5
    assert channels['crouched']==1
    assert channels['busy']==1 and channels['action_in_progress']==1
    assert sum(channels[f'hostage_{i}'] for i in range(8))>0
    assert sum(channels[f'civilian_{i}'] for i in range(8))==0
    assert sum(channels[f'goal_{i}'] for i in range(8))>0
    assert 0<channels['goal_distance']<1 and channels['time_remaining']<1


def test_demonstrations_exclude_forced_waits_but_keep_real_choices():
    examples=[];x=np.ones(4,np.float32)
    assert not add_demonstration(examples,x,0,np.array([True,False,False]),'move')
    assert add_demonstration(examples,x,2,np.array([True,True,True]),'move')
    assert len(examples)==1 and examples[0][1]==2 and examples[0][3]=='move'
    assert np.array_equal(examples[0][2],[True,True,True])


def tiny_brain():
    graph=(np.array([10,20,30],np.int64),np.array([0,1,2,2],np.int64),
           np.array([1,2],np.int32),np.array([1,1],np.int32),np.ones(3,np.int8))
    return LearningBrain(graph=graph,seed=7)


def test_causal_motor_credit_only_changes_snapshot_edges():
    brain=tiny_brain();brain.plasticity=True
    pulses=np.zeros((100,3),bool);pulses[0,0]=True;pulses[30,1]=True
    brain.step([0,1,2],[0,0,0],10.,pulses=pulses)
    snapshot=brain.capture_motor_eligibility(np.array([1],np.int32))
    before=brain.gains.copy();result=brain.reward_motor(snapshot,1.,np.array([1],np.int32),np.array([1.],np.float32))
    assert result['rule']=='causal_motor_rstdp_v1' and result['candidate_edges']==1
    assert brain.gains[0]>before[0] and brain.gains[1]==before[1]


class SmallEnv(gym.Env):
    action_space=gym.spaces.Discrete(3)
    observation_space=gym.spaces.Box(-5,5,shape=(4,),dtype=np.float32)


def test_ppo_update_uses_complete_episode_without_truncating_world():
    cfg=json.loads((ROOT/'config/learning.json').read_text());model=make_model(SmallEnv(),cfg,7)
    transitions=[]
    for i in range(8):
        obs=np.array([i/8,1.,0.,-1.],np.float32);mask=np.ones(3,bool);x=torch.as_tensor(obs[None,:])
        with torch.no_grad():
            dist=model.policy.get_distribution(x,action_masks=mask[None,:]);action=torch.tensor([1])
            value=model.policy.predict_values(x);logprob=dist.log_prob(action)
        transitions.append({'observation':obs,'action':1,'reward':1. if i==7 else 0.,'episode_start':i==0,
                            'value':value,'logprob':logprob,'mask':mask})
    before=actor_hash(model)
    result=train_adapter_episode(model,transitions,np.zeros(4,np.float32),True,False,cfg)
    assert result['updated'] and result['ticks']==8 and actor_hash(model)!=before


def test_population_normalized_readout_is_finite_and_bounded():
    readout=NeuralReadout();counts=np.zeros(int(readout.indices.max())+1,np.int32)
    counts[readout.indices[::11]]=np.arange(len(readout.indices[::11]))%7
    features=readout.update(counts)
    assert np.isfinite(features).all() and features.min()>=-5 and features.max()<=5
    assert all(abs(features.reshape(3,-1)[i].mean())<1e-4 for i in range(3))
