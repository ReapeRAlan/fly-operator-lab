import sys,json,math,copy
from pathlib import Path
import numpy as np,torch,gymnasium as gym,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from learning_brain import LearningBrain
from learning_adapter import SensoryEncoder,ActionCatalog
from learning_policy import make_model,imitate,angular_neighbours,actor_hash
from learning_curriculum import add_demonstration

# --- causal_motor_rstdp_v2: presynaptic sign ------------------------------------------------

def motor_brain():
    # 0 excitatory -> 2, 1 inhibitory -> 2; neuron 2 is the credited motor target.
    graph=(np.array([10,11,12],np.int64),np.array([0,1,2,2],np.int64),np.array([2,2],np.int32),
           np.array([1,1],np.int32),np.array([1.,-1.,1.],np.float32))
    b=LearningBrain(graph=graph,seed=1);b.set_plastic_posts([2]);b.plasticity=True;return b

def snapshot(b):
    return {'version':1,'cursor':b.cursor,'edges':np.array([0,1],np.int32),'posts':np.array([2,2],np.int32),'values':np.ones(2,np.float32)}

@pytest.mark.parametrize('delta',[1.,-1.])
def test_sign_aware_rule_moves_effective_drive_in_credited_direction(delta):
    v1=motor_brain();v1.reward_motor(snapshot(v1),delta,np.array([2],np.int32),np.array([1.],np.float32))
    assert np.sign(v1.gains[0]-1)==np.sign(delta) and np.sign(v1.gains[1]-1)==np.sign(delta)
    v2=motor_brain();result=v2.reward_motor(snapshot(v2),delta,np.array([2],np.int32),np.array([1.],np.float32),sign_aware=True)
    assert result['rule']=='causal_motor_rstdp_v2'
    # Excitatory gain follows the credit; inhibitory gain moves opposite so descending drive follows it too.
    assert np.sign(v2.gains[0]-1)==np.sign(delta) and np.sign(v2.gains[1]-1)==-np.sign(delta)
    assert np.all(v2.signs==v1.signs) and np.all(v2.gains>0)

# --- sensory schema v3.2 ---------------------------------------------------------------------

def schema(tmp_path,encoding=None):
    channels=SensoryEncoder.channels
    mapping={k:{'indices':list(range(i*16,(i+1)*16))} for i,k in enumerate(channels)}
    config={'version':'3.2' if encoding else '3.1','mapping':mapping,'baseline_hz':5.,'gain_hz':145.}
    if encoding:config['encoding']=encoding
    path=tmp_path/'schema.json';path.write_text(json.dumps(config));return path

ENCODING={'version':'3.2','sector_kappa':2.,'goal_salience':1.,'goal_proximity_m':1.5,'channel_gain_hz':{'health':30.}}

def observation(goal,aim=(1,0,0)):
    return {'operator':{'position':[0,0,0],'aim':list(aim),'health':100,'ammo':30,'capacity':30},
            'visible_enemies':[],'visible_friendlies':[],'objects':[],'inventory':[],'goal':goal,'goal_angle':0.,
            'known_geometry':[],'stage':'move','time_limit_ms':30000,'sim_time_ms':500,'action_receipt':{}}

def test_v31_encoding_is_unchanged(tmp_path):
    encoder=SensoryEncoder(schema(tmp_path));_,rates,channels=encoder.encode(observation([3.,0,.1]))
    active=[k for k in channels if k.startswith('goal_') and k[5:].isdigit() and channels[k]>0]
    assert len(active)==1 and math.isclose(channels[active[0]],min(1.,math.hypot(3.,.1)/5.))
    values=np.array([channels[k] for k in encoder.channels]);assert np.array_equal(rates,np.repeat(5.+145.*values,16))

def test_v32_goal_tuning_is_graded_peaks_at_the_goal_sector_and_stays_salient_near_the_goal(tmp_path):
    encoder=SensoryEncoder(schema(tmp_path,ENCODING))
    for angle in np.linspace(-math.pi,math.pi,16,endpoint=False)+.13: # avoid exact sector boundaries (ties)
        _,_,channels=encoder.encode(observation([2*math.cos(angle),0,2*math.sin(angle)]))
        sectors=np.array([channels[f'goal_{i}'] for i in range(8)])
        assert int(sectors.argmax())==int(((angle+math.pi)/(2*math.pi))*8)%8
        assert np.count_nonzero(sectors>.1)>=3
    _,_,near=encoder.encode(observation([.05,0,0]))
    assert max(near[f'goal_{i}'] for i in range(8))>.7 and near['goal_distance']>.9

def test_v32_channel_gain_overrides_apply_per_channel(tmp_path):
    encoder=SensoryEncoder(schema(tmp_path,ENCODING));_,rates,channels=encoder.encode(observation([3.,0,0]))
    index=encoder.channels.index('health');ammo=encoder.channels.index('ammo')
    assert np.allclose(rates[index*16:(index+1)*16],5.+30.*channels['health'])
    assert np.allclose(rates[ammo*16:(ammo+1)*16],5.+145.*channels['ammo'])
    bad=json.loads(schema(tmp_path,ENCODING).read_text());bad['encoding']['channel_gain_hz']={'nonexistent':1.}
    (tmp_path/'bad.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='unknown sensory channel'):SensoryEncoder(tmp_path/'bad.json')

# --- imitation v3.2 --------------------------------------------------------------------------

class CatalogEnv(gym.Env):
    action_space=gym.spaces.Discrete(len(ActionCatalog().entries));observation_space=gym.spaces.Box(-5,5,shape=(16,),dtype=np.float32)
    def reset(self,seed=None,options=None):return np.zeros(16,np.float32),{}
    def step(self,action):return np.zeros(16,np.float32),0,False,False,{}

SETTINGS={'learning_rate':.01,'weight_decay':1e-4,'max_epochs':120,'patience':20,'validation_fraction':.2,'angular_label_smoothing':.2,'min_epochs':5}

def direction_examples(catalog,count=240,seed=0):
    rng=np.random.default_rng(seed);moves=[i for i,e in enumerate(catalog.entries) if e['action']=='move']
    mask=np.zeros(len(catalog.entries),bool);mask[0]=True;mask[moves]=True;examples=[]
    for k in range(count):
        angle=rng.uniform(0,2*math.pi);label=moves[int(round(angle/(math.pi/4)))%8]
        x=np.zeros(16,np.float32);x[0]=math.cos(angle);x[1]=math.sin(angle);x[2:]=rng.normal(0,.3,14)
        examples.append((x,label,mask.copy(),'move',k//12))
    return examples

def test_imitation_v32_fits_directions_without_touching_critic_or_ppo_optimizer():
    cfg=json.loads((ROOT/'config/learning.json').read_text());torch.manual_seed(3)
    model=make_model(CatalogEnv(),cfg,7);catalog=ActionCatalog()
    critic=copy.deepcopy(model.policy.value_net.state_dict());optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
    before=actor_hash(model);result=imitate(model,direction_examples(catalog),settings=SETTINGS,catalog=catalog)
    assert actor_hash(model)!=before and result['version']=='3.2'
    assert result['validation_episodes']>0 and result['validation_at_selected_epoch']['within_tolerance']>.9
    assert result['within_tolerance']>.95
    assert all(torch.equal(v,model.policy.value_net.state_dict()[k]) for k,v in critic.items())
    assert model.policy.optimizer.state_dict()['state']==optimizer['state']

def test_imitation_v32_is_deterministic_under_the_torch_seed():
    cfg=json.loads((ROOT/'config/learning.json').read_text());catalog=ActionCatalog();hashes=[]
    for _ in range(2):
        torch.manual_seed(5);model=make_model(CatalogEnv(),cfg,7);torch.manual_seed(9)
        imitate(model,direction_examples(catalog,120),settings=SETTINGS,catalog=catalog);hashes.append(actor_hash(model))
    assert hashes[0]==hashes[1]

def test_angular_neighbours_stay_within_one_action_family():
    catalog=ActionCatalog();neighbours=angular_neighbours(catalog)
    for index,items in neighbours.items():
        entry=catalog.entries[index]
        assert items and all(catalog.entries[j]['action']==entry['action'] and catalog.entries[j].get('slot')==entry.get('slot') for j in items)
    move=[i for i,e in enumerate(catalog.entries) if e['action']=='move']
    assert all(len(neighbours[i])==2 for i in move)

def test_demonstrations_can_record_their_episode():
    examples=[];mask=np.array([True,True]);x=np.ones(3,np.float32)
    assert add_demonstration(examples,x,1,mask,'move',episode=4) and examples[0][4]==4
    assert add_demonstration(examples,x,1,mask,'move') and len(examples[1])==4
