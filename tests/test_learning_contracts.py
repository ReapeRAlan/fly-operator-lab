import sys,json,copy
from pathlib import Path
import numpy as np,torch,pytest,gymnasium as gym
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from learning_adapter import clean_observation,ActionCatalog
from learning_policy import make_model,imitate,actor_hash,internal_update
import lab_api
from lab_api import app
from fastapi.testclient import TestClient

def observation():
    return {'operator':{'id':1,'position':[0,0,0],'aim':[1,0,0],'health':100,'ammo':3,'capacity':31,'weapon_state':5,'commands':0},
      'humans':[{'id':2,'kind':1,'position':[99,0,99],'health':100,'ammo':30},{'id':3,'kind':1,'position':[1,0,0],'health':40,'ammo':2}],
      'visible_ids':[3],'visible_enemies':[{'id':3,'position':[1,0,0],'health':40,'ammo':2}], 'inventory':[],'objects':[],'sim_time_ms':50}

def test_hidden_information_never_enters_actor():
    raw=observation();mission={'goal':[2,0,0],'goal_angle':0,'stage':'move'}
    first=clean_observation(raw,mission);raw['humans'][0]['position']=[-123,0,700];raw['visible_enemies'][0]['ammo']=500
    assert clean_observation(raw,mission)==first
    assert set(first['visible_enemies'][0])=={'id','position','alive'}

def test_mask_mechanical_limits():
    c=ActionCatalog();o=clean_observation(observation(),{'goal':[2,0,0],'goal_angle':0,'stage':'move'})
    i=next(i for i,e in enumerate(c.entries) if e['action']=='fire')
    assert c.mask(o)[i];o['operator']['ammo']=0;assert not c.mask(o)[i]
    assert not any(c.mask(o)[j] for j,e in enumerate(c.entries) if e['action'] in ('throw','door_open','loadout'))
    assert list(np.flatnonzero(c.mask(o,{'wait'})))==[0]

def test_train_validation_test_layouts_disjoint():
    missions=json.loads((ROOT/'data/learning/scenarios.json').read_text())
    sets={split:{m['seed'] for m in missions if m['split']==split} for split in ('train','validation','test')}
    assert not sets['train']&sets['validation'] and not sets['test']&sets['validation'] and not sets['test']&sets['train']
    assert len([m for m in missions if m['stage']=='move' and m['split']=='validation'])==90

class SmallEnv(gym.Env):
    action_space=gym.spaces.Discrete(3);observation_space=gym.spaces.Box(0,np.inf,shape=(4,),dtype=np.float32)
    def reset(self,seed=None,options=None):return np.zeros(4,np.float32),{}
    def step(self,action):return np.zeros(4,np.float32),0,False,False,{}

def test_imitation_updates_actor_internal_updates_only_critic():
    cfg=json.loads((ROOT/'config/learning.json').read_text());model=make_model(SmallEnv(),cfg,7)
    x=np.ones(4,np.float32);before=actor_hash(model);imitate(model,[(x,1,np.ones(3,bool))]*8,epochs=2)
    assert actor_hash(model)!=before;before=actor_hash(model)
    class Brain:
        def reward(self,error):self.error=error;return {'changed_edges':1}
    b=Brain();internal_update(model,b,x,x,1,True,.999)
    assert actor_hash(model)==before and np.isfinite(b.error)

def test_dashboard_read_paths_and_origin_guard():
    with TestClient(app) as client:
        assert client.get('/api/parameters').status_code==200
        assert client.get('/api/neurons?q=10001').json()[0]['bodyId']==10001
        out=client.get('/api/neuron/10001?limit=3').json();assert len(out['connections'])==3 and out['total']>0
        assert all(e['pre']==10001 and e['anatomical_count']>0 for e in out['connections'])
        assert client.post('/api/control/stop',headers={'origin':'https://unrelated.example'}).status_code==403
        for kind in ('csv','svg','png','bundle'):assert client.get('/api/export/'+kind).status_code==200

def test_heavy_dashboard_cache_releases_only_after_idle_window():
    calls=[];lab_api._heavy_cache_last_access=100.
    assert not lab_api.release_heavy_cache_if_idle(now=219,idle_seconds=120,clearers=[lambda:calls.append('a')])
    assert calls==[]
    assert lab_api.release_heavy_cache_if_idle(now=220,idle_seconds=120,clearers=[lambda:calls.append('a'),lambda:calls.append('b')])
    assert calls==['a','b'] and lab_api._heavy_cache_last_access==0.
