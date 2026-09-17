"""Protocol 3.3 contracts: PPO mechanics, masks, sectors, rewards, blocks and controls."""
import sys,json,math,copy
from pathlib import Path
import numpy as np,torch,gymnasium as gym,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from learning_adapter import ActionCatalog,SensorReadout,SensoryEncoder,sector_of,decision_required
from learning_curriculum import add_demonstration,block_passed,promotion_ready,training_stage,wilson_lower
from learning_env import reward_terms
from learning_policy import make_model,actor_hash,imitate
from learning_ppo import (RolloutCollector,episode_advantages,ppo_update,new_state,identity_check,install_group_optimizer,
                          group_learning_rates,anchor_beta,learning_rates_for)

CFG=json.loads((ROOT/'config/learning.json').read_text())
N_ACTIONS=6

class StubEnv(gym.Env):
    action_space=gym.spaces.Discrete(N_ACTIONS)
    observation_space=gym.spaces.Box(-5,5,shape=(8,),dtype=np.float32)

def model(seed=7):
    torch.manual_seed(seed);return make_model(StubEnv(),CFG,seed)

def settings(**overrides):
    return {**CFG['ppo_v33'],**overrides}

def collect(policy_model,episodes,rng,length=6,reward=.1,teacher=1,truncate=False):
    """Synthetic rollout recorded exactly as the trainer records it (single-sample forward)."""
    collector=RolloutCollector()
    for _ in range(episodes):
        transitions=[]
        for _ in range(length):
            x=rng.normal(size=8).astype(np.float32);mask=np.ones(N_ACTIONS,bool);mask[5]=False
            with torch.no_grad():
                distribution=policy_model.policy.get_distribution(torch.as_tensor(x[None,:]),action_masks=mask[None,:])
                action=distribution.get_actions(deterministic=False);logprob=distribution.log_prob(action)
                value=policy_model.policy.predict_values(torch.as_tensor(x[None,:]))
            transitions.append({'observation':x,'action':int(action[0]),'reward':reward,'value':float(value.reshape(-1)[0]),
                                'logprob':float(logprob.reshape(-1)[0]),'mask':mask,'macro_steps':1,'teacher':teacher})
        collector.add_episode(transitions,not truncate,truncate,0. if truncate else None)
    return collector

# --- GAE and collection -------------------------------------------------------------------

def test_two_one_decision_episodes_do_not_leak_advantage_across_the_boundary():
    collector=RolloutCollector()
    for _ in range(2):
        collector.add_episode([{'observation':np.zeros(8,np.float32),'action':0,'reward':1.,'value':0.,'logprob':0.,
                                'mask':np.ones(N_ACTIONS,bool),'macro_steps':1,'teacher':None}],True,False)
    batch=collector.batch(.99,.95)
    assert batch['advantages'].tolist()==[1.,1.]
    assert batch['returns'].tolist()==[1.,1.]

def test_truncation_bootstraps_and_termination_does_not():
    adv,ret=episode_advantages([1.],[0.],[1],.9,.95,bootstrap_value=.5)
    assert adv[0]==pytest.approx(1.45) and ret[0]==pytest.approx(1.45)
    adv,_=episode_advantages([1.],[0.],[1],.9,.95,bootstrap_value=None)
    assert adv[0]==pytest.approx(1.)

def test_macro_action_discounts_by_its_tick_count_and_lambda_once_per_decision():
    gamma,lam=.9,.5
    adv,_=episode_advantages([0.,1.],[.2,.4],[3,1],gamma,lam)
    delta1=1.-.4;delta0=0.+gamma**3*.4-.2
    assert adv[1]==pytest.approx(delta1)
    assert adv[0]==pytest.approx(delta0+gamma**3*lam*delta1)

def test_truncated_episode_requires_a_bootstrap_value_and_incidents_are_excluded():
    collector=RolloutCollector();item={'observation':np.zeros(8,np.float32),'action':0,'reward':0.,'value':0.,'logprob':0.,'mask':np.ones(N_ACTIONS,bool)}
    with pytest.raises(ValueError):collector.add_episode([item],False,True,None)
    assert not collector.add_episode([item],True,False,excluded=True)
    assert collector.decisions==0 and collector.excluded==1

# --- Identity, learning rates, warm-up, target_kl, anchor ---------------------------------

def test_identity_check_passes_for_the_collecting_policy_and_fails_after_a_change():
    m=model();collector=collect(m,3,np.random.default_rng(0))
    batch=collector.batch(.999,.95)
    result=identity_check(m.policy,batch)
    assert result['passed'] and result['max_ratio_deviation']<=1e-5 and result['approx_kl']<=1e-6
    with torch.no_grad():m.policy.action_net.weight[2].add_(.5)
    assert not identity_check(m.policy,batch)['passed']

def test_identity_failure_blocks_the_update():
    m=model();collector=collect(m,3,np.random.default_rng(1));before=actor_hash(m)
    with torch.no_grad():m.policy.action_net.bias[1].add_(1.)
    changed=actor_hash(m);state=new_state()
    result=ppo_update(m,collector,settings(),state)
    assert not result['updated'] and result['reason']=='identity_check_failed' and state['identity_failures']==1
    assert actor_hash(m)==changed!=before

def test_group_optimizer_covers_actor_and_critic_and_survives_updates():
    m=model();optimizer=m.policy.optimizer
    assert [g['name'] for g in optimizer.param_groups]==['actor','critic']
    state=new_state();state['phase']='actor'
    s=settings(actor_learning_rate=2e-5,critic_learning_rate=3e-4,actor_ramp_updates=1)
    for seed in range(2):
        ppo_update(m,collect(m,4,np.random.default_rng(seed)),s,state)
        assert group_learning_rates(m.policy.optimizer)=={'actor':2e-5,'critic':3e-4}
    assert m.policy.optimizer is optimizer

def test_actor_ramp_reaches_its_target_rate():
    s=settings(actor_learning_rate=1e-5,actor_ramp_updates=4);state=new_state();state['phase']='actor'
    rates=[]
    for k in range(5):
        state['actor_updates']=k;rates.append(learning_rates_for(state,s)['actor'])
    assert rates==pytest.approx([2.5e-6,5e-6,7.5e-6,1e-5,1e-5])
    state['phase']='critic_warmup';assert learning_rates_for(state,s)['actor']==0.

def test_critic_warmup_changes_only_the_critic_and_then_hands_over():
    m=model();state=new_state();s=settings(critic_warmup_min_rollouts=3,critic_warmup_max_rollouts=3)
    actor=actor_hash(m);critic=[p.detach().clone() for p in m.policy.value_net.parameters()]
    for seed in range(3):
        result=ppo_update(m,collect(m,4,np.random.default_rng(seed),reward=1.),s,state)
        assert result['updated'] and result['actor_delta_norm']==0.
        assert actor_hash(m)==actor
    assert any(not torch.equal(a,b) for a,b in zip(critic,m.policy.value_net.parameters()))
    assert state['phase']=='actor' and state['critic_warmup_updates']==3

def test_target_kl_stops_the_epochs():
    m=model();state=new_state();state['phase']='actor'
    s=settings(actor_learning_rate=1.,actor_ramp_updates=1,target_kl=1e-9,epochs=4,minibatch_size=8,anchor_beta=0.)
    result=ppo_update(m,collect(m,6,np.random.default_rng(3),reward=1.),s,state)
    assert result['stopped_by_target_kl'] and result['epochs_completed']<4

def test_anchor_pulls_the_actor_toward_the_instructor_label():
    m=model();state=new_state();state['phase']='actor'
    s=settings(actor_learning_rate=5e-2,actor_ramp_updates=1,anchor_beta=1.,target_kl=None,epochs=4,minibatch_size=16,vf_coef=0.)
    probe=torch.as_tensor(np.random.default_rng(9).normal(size=(64,8)).astype(np.float32));mask=np.ones((64,N_ACTIONS),bool);mask[:,5]=False
    def teacher_probability():
        with torch.no_grad():return float(m.policy.get_distribution(probe,action_masks=mask).distribution.probs[:,3].mean())
    before=teacher_probability()
    for seed in range(3):
        result=ppo_update(m,collect(m,6,np.random.default_rng(10+seed),reward=0.,teacher=3),s,state)
        assert result['anchor_loss'] is not None
    assert teacher_probability()>before+.1

def test_anchor_beta_decays_by_decisions_to_its_floor():
    s=settings(anchor_beta=1.,anchor_half_life_decisions=20000,anchor_beta_floor=.05)
    assert anchor_beta(0,s)==1. and anchor_beta(20000,s)==pytest.approx(.5) and anchor_beta(10**7,s)==.05

def test_update_frequency_control_skips_every_other_rollout():
    m=model();state=new_state();s=settings(update_every_rollouts=2)
    first=ppo_update(m,collect(m,3,np.random.default_rng(4)),s,state)
    second=ppo_update(m,collect(m,3,np.random.default_rng(5)),s,state)
    assert not first['updated'] and first['reason']=='update_frequency_control' and second['updated']

def test_ppo_state_is_json_serializable():
    m=model();state=new_state()
    ppo_update(m,collect(m,3,np.random.default_rng(6)),settings(),state)
    json.dumps(state,allow_nan=False)

def test_imitation_with_its_own_optimizer_leaves_the_ppo_optimizer_untouched():
    m=model();optimizer_state=copy.deepcopy(m.policy.optimizer.state_dict())
    mask=np.ones(N_ACTIONS,bool);examples=[(np.random.default_rng(i).normal(size=8).astype(np.float32),i%3,mask,'move') for i in range(30)]
    imitate(m,examples,5,own_optimizer=True)
    assert m.policy.optimizer.state_dict()['state']==optimizer_state['state']=={}

# --- Masks, sectors and rewards ---------------------------------------------------------

def observation(**overrides):
    obs={'operator':{'id':1,'position':[0.,0.,0.],'aim':[1.,0.,0.],'health':100,'commands':0,'weapon_state':5,'ammo':10,'capacity':30},
         'inventory':[{'slot':1,'type':1,'equipped':True,'quantity':1},{'slot':5,'type':1,'equipped':False,'quantity':1}],
         'visible_enemies':[],'visible_friendlies':[],'objects':[],'action_receipt':{},'sim_time_ms':500,'stage':'move','goal':[3.,0.,0.]}
    obs.update(overrides);return obs

def indices(catalog,action,**attributes):
    return [i for i,e in enumerate(catalog.entries) if e['action']==action and all(e.get(k)==v for k,v in attributes.items())]

def test_stop_and_cancel_are_legal_during_any_native_command_not_only_near_the_goal():
    catalog=ActionCatalog();busy=observation();busy['operator']=dict(busy['operator'],commands=1)
    legal=catalog.legal_mask(busy,{'wait','move','stop','cancel'})
    assert legal[indices(catalog,'stop')[0]] and legal[indices(catalog,'cancel')[0]]
    assert not legal[indices(catalog,'move')].any()
    assert decision_required(busy,legal)

def test_curriculum_mask_hides_families_but_never_depends_on_state():
    catalog=ActionCatalog();families={'wait','move'}
    used=catalog.mask(observation(),None,families)
    assert used[0] and used[indices(catalog,'move')].all() and not used[indices(catalog,'stop')].any()
    assert np.array_equal(catalog.family_mask(families),catalog.family_mask(families))

def test_stance_mask_offers_both_crouch_values():
    catalog=ActionCatalog();legal=catalog.legal_mask(observation(stage='stance'),{'wait','crouch'})
    assert legal[indices(catalog,'crouch',value=True)[0]] and legal[indices(catalog,'crouch',value=False)[0]]

def enemy(ident,x,z):return {'id':ident,'position':[x,0.,z],'alive':True}

def test_permuting_target_ids_keeps_the_same_spatial_target():
    catalog=ActionCatalog()
    positions=[(4.,0.1),(0.2,5.),(-3.,-.2),(6.,.3)]
    first=observation(visible_enemies=[enemy(i+10,*p) for i,p in enumerate(positions)])
    second=observation(visible_enemies=[enemy(99-i,*p) for i,p in enumerate(positions)])
    for i in indices(catalog,'aim_target'):
        a=catalog.target(catalog.entries[i],first);b=catalog.target(catalog.entries[i],second)
        assert (a is None)==(b is None)
        if a is not None:assert a['position']==b['position']
    assert catalog.legal_mask(first,{'wait','aim_target'}).tolist()==catalog.legal_mask(second,{'wait','aim_target'}).tolist()

def test_aim_sector_matches_the_sensory_enemy_channel_and_picks_the_nearest():
    catalog=ActionCatalog();obs=observation(visible_enemies=[enemy(5,8.,.2),enemy(6,4.,.1)])
    angle=math.atan2(.1,4.);sector=sector_of(angle,16)
    entry=catalog.entries[indices(catalog,'aim_target',sector=sector)[0]]
    assert catalog.target(entry,obs)['id']==6
    command=catalog.decode(catalog.entries.index(entry),obs)
    assert command['action']=='aim' and command['target_id']==6 if 'target_id' in command else command['action']=='aim'
    centre=-math.pi+(sector+.5)*2*math.pi/16
    assert abs(math.atan2(math.sin(angle-centre),math.cos(angle-centre)))<=math.pi/16+1e-9

def test_doors_are_addressed_by_sector_within_reach():
    catalog=ActionCatalog()
    door={'id':300,'kind':'door','template':'door_wood_01','position':[1.,0.,0.]}
    far={'id':301,'kind':'door','template':'door_wood_01','position':[0.,0.,9.]}
    legal=catalog.legal_mask(observation(objects=[far,door]),{'wait','door_open'})
    open_indices=[i for i in indices(catalog,'door_open') if legal[i]]
    assert len(open_indices)==1 and catalog.target(catalog.entries[open_indices[0]],observation(objects=[far,door]))['id']==300

def test_completing_the_task_beats_wandering_and_rejection_never_pays():
    config={'free_wait_penalty':.005}
    def episode_return(ticks,final_success,start_phi=-.15):
        total=0.;phi=start_phi
        for t in range(ticks):
            last=t==ticks-1;success=final_success and last
            new_phi=0. if success else (start_phi*(1-(t+1)/ticks) if final_success else start_phi)
            _,r=reward_terms(success,False,new_phi,phi,True,False,False,config);total+=r;phi=new_phi
        return total
    assert episode_return(40,True)>episode_return(600,False)
    _,wait=reward_terms(False,False,-.1,-.1,True,False,False,config)
    _,rejected=reward_terms(False,False,-.1,-.1,True,True,False,config)
    _,invalid=reward_terms(False,False,-.1,-.1,False,False,False,config)
    assert rejected<wait and invalid<wait

def test_wait_penalty_applies_only_to_idle_waits():
    terms,_=reward_terms(False,False,0.,0.,True,False,True,{'free_wait_penalty':.005})
    assert terms['wait_penalty']==-.005
    terms,_=reward_terms(False,False,0.,0.,True,False,False,{'free_wait_penalty':.005})
    assert terms['wait_penalty']==0.

# --- Controls, blocks and schedule --------------------------------------------------------

def test_sensor_readout_uses_the_three_readout_filters_on_channel_rates():
    readout=SensorReadout(['a','b','c','d'])
    first=readout.update(np.array([5.,150.,5.,77.]))
    assert first.shape==(12,) and np.isfinite(first).all()
    for _ in range(200):last=readout.update(np.array([5.,150.,5.,77.]))
    assert np.allclose(last.reshape(3,4)[0],last.reshape(3,4)[2],atol=1e-3)

def test_demonstrations_carry_sensor_features_of_the_same_tick():
    examples=[];mask=np.array([True,True])
    assert add_demonstration(examples,np.ones(4),1,mask,'move',episode=3,sensor_features=np.full(6,2.))
    assert len(examples[0])==6 and examples[0][4]==3 and np.array_equal(examples[0][5],np.full(6,2.,np.float32))
    with pytest.raises(ValueError):add_demonstration(examples,np.ones(4),1,mask,'move',sensor_features=np.ones(6))

def test_fixed_block_needs_24_of_25_and_retention_bars():
    promotion=CFG['promotion']
    good={'move':{'successes':9,'episodes':10},'stance':{'successes':24,'episodes':25}}
    assert block_passed(good,promotion,'stance')
    assert not block_passed({**good,'stance':{'successes':23,'episodes':25}},promotion,'stance')
    assert not block_passed({**good,'move':{'successes':7,'episodes':10}},promotion,'stance')
    assert not block_passed({'stance':{'successes':5,'episodes':5}},promotion,'stance')
    assert wilson_lower(24,25)==pytest.approx(.805,abs=1e-3)

def test_promotion_uses_the_latest_block_of_every_main_learner():
    blocks={'move':{'ppo_anchored_7':[{'passed':True}],'ppo_anchored_19':[{'passed':True},{'passed':False}],'ppo_anchored_43':[{'passed':True}]}}
    assert not promotion_ready(blocks,'move',['ppo_anchored'],[7,19,43])
    blocks['move']['ppo_anchored_19'].append({'passed':True})
    assert promotion_ready(blocks,'move',['ppo_anchored'],[7,19,43])
    assert not promotion_ready(blocks,'move',['ppo_anchored','other'],[7,19,43])

def test_retention_revisits_every_earlier_stage_equally():
    stages=['move','stance','cancel','orient']
    visits=[training_stage(stages,'orient',e) for e in range(4*12)]
    earlier=[s for s in visits if s!='orient']
    assert len(earlier)==12 and {s:earlier.count(s) for s in set(earlier)}=={'move':4,'stance':4,'cancel':4}
    assert all(training_stage(stages,'move',e)=='move' for e in range(8))

def test_stored_ticks_drop_redundant_masks_and_round_floats_without_touching_the_live_snapshot():
    from lab_store import durable_row
    info={'decision':{'action_labels':['wait','stop'],'mask':[True,False],'probabilities':[.123456789012,.876543210987]},
          'pre_action':{'mask':[True,False],'channels':{'health':.99999999}},
          'trace':{'legal_mask':[0,1],'curriculum_mask':[0,1],'used_mask':[0,1],'tick':3},'reward':-.0010000000001}
    row=durable_row(info)
    assert 'action_labels' not in row['decision'] and row['decision']['mask']==[True,False]
    assert 'mask' not in row['pre_action'] and 'used_mask' not in row['trace'] and row['trace']['legal_mask']==[0,1]
    assert row['decision']['probabilities']==[.12346,.87654] and row['reward']==-.001
    assert info['pre_action']['mask']==[True,False] and info['decision']['action_labels']==['wait','stop']
    assert len(json.dumps(row))<len(json.dumps(info))

def test_night_config_separates_tracks_and_every_condition_has_settings():
    tracks=CFG['tracks'];settings_=CFG['condition_settings']
    assert tracks['main']==['ppo_anchored'] and 'ppo_anchored' not in tracks['science']
    assert all(c in settings_ for c in tracks['main']+tracks['science'])
    assert {settings_[c]['brain_mode'] for c in settings_}<={'connectome','sensor_only','bias_only'}
    assert CFG['semantic_protocol_version']=='3.3'
