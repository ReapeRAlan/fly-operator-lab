"""Gymnasium interface to the local game and the complete classified MaleCNS."""
from pathlib import Path
import json,math,time
import numpy as np,psutil
import gymnasium as gym
from gymnasium import spaces
from bridge_client import Bridge
from learning_adapter import SensoryEncoder,NeuralReadout,ActionCatalog,clean_observation,bearing,native_progress,decision_required
ROOT=Path(__file__).resolve().parents[1]
PROCESS=psutil.Process()

def integrate(brain,ids,rates,record=False):
    """Step the brain and attach process page faults observed during integration."""
    faults=PROCESS.memory_info().num_page_faults
    counts,activity=brain.step(ids,rates,50.,record=record)
    activity['page_faults']=PROCESS.memory_info().num_page_faults-faults
    return counts,activity

class FlyOperatorEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,brain,bridge=None,condition='adapter',seed=7,stage='move',split='train',on_step=None,on_episode=None,guard=None,sensory_schema=None):
        self.brain=brain;self.bridge=bridge or Bridge();self.condition=condition;self.seed_base=seed
        self.encoder=SensoryEncoder(ROOT/sensory_schema) if sensory_schema else SensoryEncoder()
        self.readout=NeuralReadout();self.catalog=ActionCatalog()
        self.action_space=spaces.Discrete(len(self.catalog.entries));self.observation_space=spaces.Box(-5.,5.,shape=(3942,),dtype=np.float32)
        self.stage=stage;self.split=split;self.on_step=on_step;self.on_episode=on_episode;self.guard=guard
        self.scenarios=json.loads((ROOT/'data/learning/scenarios.json').read_text())
        self.episode_count=0;self.total_steps=0;self.last_features=np.zeros(3942,np.float32)
        self.train_plasticity=condition=='internal';self.record=False;self.last_info={};self.neural_record=False
        self.allowed=None;self.curriculum_allowed=None;self.pending_error=0.;self.last_plasticity={};self.total_simulated_ms=0
        self.free_wait_streak=0;self.forced_wait_steps=0;self.free_wait_steps=0;self.decision_steps=0
        self.no_progress_decisions=0;self.best_goal_distance=float('inf')
        cap=ROOT/'outputs/native_capabilities_v2.json'
        if cap.exists():self.allowed=set(json.loads(cap.read_text()).get('validated_actions',[]))|{'wait'}

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {}
        if self.guard:self.guard()
        if 'mission' in options:
            self.mission=next(m for m in self.scenarios if m['name']==options['mission'])
        else:
            candidates=[m for m in self.scenarios if m['stage']==self.stage and m['split']==self.split]
            offset=(self.seed_base+self.episode_count)%len(candidates)
            self.mission=candidates[offset]
        self.record=options.get('record',False)
        self.raw=self.bridge.new_mission(self.mission['name'],self.record)
        self.brain.reset(self.seed_base*100000+self.episode_count if seed is None else seed)
        self.brain.plasticity=self.train_plasticity;self.readout.reset()
        self.obs=clean_observation(self.raw,self.mission);self.start_raw=self.raw
        self.episode_reward=0.;self.steps=0;self.episode_wall=time.time();self.episode_start_ms=self.raw['sim_time_ms']
        self.initial_ammo=self.raw['operator']['ammo'];self.shot_ever=False;self.rescue_followed=False;self.reload_oriented=False;self.cancel_practiced=False
        self.rejections=0;self.initial_people={h['id']:h for h in self.raw['humans']}
        ids,rates,self.channels=self.encoder.encode(self.obs)
        counts,self.activity=integrate(self.brain,ids,rates)
        self.last_counts=counts;self.last_features=self.readout.update(counts)
        self.previous_phi=self.potential(self.obs);self.done=False;self.truncate_next=False
        self.free_wait_streak=0;self.forced_wait_steps=0;self.free_wait_steps=0;self.decision_steps=0
        self.no_progress_decisions=0;self.best_goal_distance=bearing(self.obs['operator'],self.mission['goal'])[1]
        return self.last_features.copy(),{'mission':self.mission,'neural_warmup_ms':50,'game_map_seed':self.raw.get('map_seed')}

    def action_masks(self):
        allowed=self.allowed if self.curriculum_allowed is None else self.allowed&set(self.curriculum_allowed)
        return self.catalog.mask(self.obs,allowed)

    def potential(self,obs):
        stage=self.mission['stage'];a=obs['operator']
        if stage in ('move','cancel','door','breach','rescue','defuse'):
            return -min(bearing(a,self.mission['goal'])[1],20.)/20.
        if stage=='orient':
            yaw=math.atan2(a['aim'][2],a['aim'][0]);d=(self.mission['goal_angle']-yaw+math.pi)%(2*math.pi)-math.pi
            return -abs(d)/math.pi
        return 0.

    def skill_success(self,raw):
        stage=self.mission['stage'];a=raw.get('operator',{})
        if a.get('health',0)<=0:return False
        receipt=raw.get('action_receipt',{});completed=receipt.get('status')=='completed'
        if stage=='move':return bearing(a,self.mission['goal'])[1]<.5 and completed and receipt.get('action')=='stop' and a.get('speed',0)<.05
        if stage=='stance':return completed and receipt.get('action')=='crouch' and a.get('crouched') is True
        if stage=='cancel':return (self.cancel_practiced and completed and receipt.get('action')=='cancel' and a.get('commands',0)==0 and
          any(o.get('id')==self.mission['target_id'] and o.get('door_state')!=0 for o in raw.get('objects',[])))
        if stage=='orient':return self.potential(clean_observation(raw,self.mission))>-.04
        if stage=='switch':return any(e['slot']==5 and e['equipped'] and e['state']>2 for e in raw['inventory'])
        if stage=='shoot':return raw.get('shots_accepted',0)>0
        if stage=='reload':return self.shot_ever and completed and receipt.get('action')=='reload'
        if stage in ('door','breach'):return any(o.get('id')==self.mission['target_id'] and o.get('door_state')==0 for o in raw['objects'])
        if stage=='grenade':return any(e['slot'] in (12,13) and e['quantity']<next(x['quantity'] for x in self.start_raw['inventory'] if x['slot']==e['slot']) for e in raw['inventory'])
        if stage=='loadout':return raw.get('mission_result')==1 and raw.get('loadout_choice',-1)>=0
        return raw.get('mission_result')==1

    def step(self,action):
        if self.done:raise RuntimeError('reset required after episode completion')
        if self.guard:self.guard()
        if not self.action_space.contains(action):raise ValueError('Action index out of range')
        action=int(action);mask=self.action_masks();before=self.obs;start=time.perf_counter()
        _,_,pre_channels=self.encoder.encode(before)
        if mask[action]:command=self.catalog.decode(action,before)
        else:command={'action':'wait'}
        forced_wait=bool(not decision_required(before,mask))
        raw=self.bridge.step(command,50);self.shot_ever |= raw.get('shots_accepted',0)>0
        if command['action']=='follow':self.rescue_followed=True
        if raw.get('action_receipt',{}).get('action')=='cancel' and raw.get('action_receipt',{}).get('status')=='completed':self.cancel_practiced=True
        self.obs=clean_observation(raw,self.mission);self.raw=raw;self.steps+=1;self.total_steps+=1;self.total_simulated_ms+=50
        if forced_wait:self.forced_wait_steps+=1;self.free_wait_streak=0
        else:
            self.decision_steps+=1
            if command['action']=='wait':self.free_wait_steps+=1;self.free_wait_streak+=1
            else:self.free_wait_streak=0
        if raw.get('action_receipt',{}).get('status')=='rejected' and command['action']!='wait':self.rejections+=1
        success=self.skill_success(raw);native_result=raw.get('mission_result',0)
        failure=raw.get('operator',{}).get('health',0)<=0 or native_result==2
        # A terminal native victory also ends isolated tasks, but only the tested skill earns success.
        terminated=bool(success or failure or native_result==1 or raw.get('game_state')==2)
        collapse_limit=int(getattr(self,'config',{}).get('policy_collapse_free_wait_steps',0))
        policy_collapse=bool(collapse_limit and self.free_wait_streak>=collapse_limit)
        goal_distance=bearing(self.obs['operator'],self.mission['goal'])[1]
        if not forced_wait and self.mission['stage'] in ('move','cancel','door','breach','rescue','defuse'):
            if goal_distance<self.best_goal_distance-.1:self.best_goal_distance=goal_distance;self.no_progress_decisions=0
            else:self.no_progress_decisions+=1
        progress_limit=int(getattr(self,'config',{}).get('policy_no_progress_decisions',0))
        no_progress=bool(progress_limit and self.no_progress_decisions>=progress_limit)
        truncated=bool(not terminated and (self.steps*50>=self.mission['max_seconds']*1000 or self.truncate_next or policy_collapse or no_progress))
        phi=0. if terminated else self.potential(self.obs)
        reward=float(success)-float(failure)-.001 + .999*phi-self.previous_phi
        self.previous_phi=phi
        if not mask[action]:reward-=.01
        if not forced_wait and command['action']=='wait':reward-=float(getattr(self,'config',{}).get('free_wait_penalty',0.))
        self.episode_reward+=reward
        ids,rates,self.channels=self.encoder.encode(self.obs)
        self.brain.plasticity=self.train_plasticity
        counts,self.activity=integrate(self.brain,ids,rates,record=self.neural_record)
        self.last_counts=counts;self.last_features=self.readout.update(counts)
        self.done=terminated or truncated
        info={'success':bool(success),'mission_result':native_result,'terminated':terminated,'truncated':truncated,
          'mission':self.mission['name'],'stage':self.mission['stage'],'split':self.mission['split'],'condition':self.condition,
          'seed':self.seed_base,'game_map_seed':raw.get('map_seed'),'episode':raw['episode'],'sequence':raw['sequence'],
          'game_session':f"{self.bridge.session.get('pid')}:{self.bridge.session.get('started_unix')}",
          'rendering':{'frames':raw.get('presentation_count'),'interval_ms':raw.get('presentation_interval_ms'),'client_view':raw.get('client_view'),'server_flags':raw.get('game_flags')},
          'simulated_ms':self.steps*50,'brain_simulated_ms':self.brain.cursor*self.brain.dynamics.dt_ms,
          'reward':reward,'observation':self.obs,'action_index':action,'action_label':self.catalog.label(action),
          'truncation_reason':('policy_collapse' if policy_collapse else 'no_progress' if no_progress else 'condition_switch_budget' if self.truncate_next else 'time_limit') if truncated else None,
          'command':command,'mask_valid':bool(mask[action]),'action_receipt':raw.get('action_receipt',{}),
          'forced_wait':forced_wait,'decision_required':not forced_wait,'free_wait_streak':self.free_wait_streak,
          'goal_distance':goal_distance,'best_goal_distance':self.best_goal_distance,'no_progress_decisions':self.no_progress_decisions,
          'pre_action':{'channels':pre_channels,'mask':mask.tolist(),'goal_distance':bearing(before['operator'],self.mission['goal'])[1],
                        'busy_reason':'command_queue' if before['operator'].get('commands',0)>0 else 'receipt_in_progress' if before.get('action_receipt',{}).get('status')=='in_progress' else 'initial_visibility' if before.get('sim_time_ms',0)==0 and before.get('stage')!='loadout' else 'free'},
          'activity':self.activity,'channels':self.channels,'neurons_readout':self.readout.summary(counts),
          'step_wall_seconds':time.perf_counter()-start}
        if 'capture_directory' in raw:info['capture']={k:raw[k] for k in ('capture_directory','capture_frame','frame') if k in raw}
        self.last_info=info
        if self.on_step:self.on_step(info)
        if self.done:
            self.episode_count+=1
            summary={k:info[k] for k in ('success','mission_result','terminated','truncated','truncation_reason','mission','stage','split','condition','seed','game_map_seed','game_session','episode','simulated_ms')}
            summary.update(reward=self.episode_reward,steps=self.steps,wall_seconds=time.time()-self.episode_wall,
                           operator_health=raw.get('operator',{}).get('health'),native_victory=bool(native_result==1),time=time.time(),
                           rejected_actions=self.rejections,consumables_used=sum(max(0,e['quantity']-next((v['quantity'] for v in raw['inventory'] if v['slot']==e['slot']),0)) for e in self.start_raw['inventory'] if e['type']==3),
                           protected_losses=sum(1 for h in raw['humans'] if h['kind']!=1 and h['health']<=0 and self.initial_people.get(h['id'],{}).get('health',0)>0),
                           forced_wait_steps=self.forced_wait_steps,free_wait_steps=self.free_wait_steps,decision_steps=self.decision_steps,
                           final_goal_distance=info['goal_distance'],best_goal_distance=self.best_goal_distance,
                           no_progress_decisions=self.no_progress_decisions)
            if self.on_episode:self.on_episode(summary)
        return self.last_features.copy(),reward,terminated,truncated,info

    def close(self):self.bridge.close()

class ExerciseTeacher:
    """Auditable training-only instructor. No hidden enemy state and no evaluation use."""
    def __init__(self,catalog):self.catalog=catalog
    def choose(self,env):
        obs=env.obs;a=obs['operator'];mask=env.action_masks();stage=env.mission['stage'];entries=self.catalog.entries
        def first(action,**attributes):
            return next((i for i,e in enumerate(entries) if mask[i] and e['action']==action and all(e.get(k)==v for k,v in attributes.items())),None)
        def move_to(point):
            delta,_=bearing(a,point)
            candidates=[i for i,e in enumerate(entries) if mask[i] and e['action']=='move']
            if not candidates:return 0
            return min(candidates,key=lambda i:abs((entries[i]['angle']-delta+math.pi)%(2*math.pi)-math.pi))
        if stage=='loadout' and obs['sim_time_ms']==0:return first('loadout',kit=env.mission['seed']%3) or 0
        if env.steps==0:return 0 # let the first native visibility update finish before choosing a direction
        receipt=obs.get('action_receipt',{})
        progressing=native_progress(obs)
        if receipt.get('action')=='cancel' and receipt.get('status')=='in_progress':return 0
        if stage=='move' and bearing(a,env.mission['goal'])[1]<.5:return first('stop') or 0
        if stage=='stance':return first('crouch',value=True) or 0
        if stage=='cancel':
            if progressing:return first('cancel') or 0
            return first('door_breach',slot=16) or 0
        if stage=='breach' and progressing and not env.cancel_practiced:return first('cancel') or 0
        if progressing:return 0
        if stage=='move':return move_to(env.mission['goal'])
        if stage=='orient':
            yaw=math.atan2(a['aim'][2],a['aim'][0]);delta=(env.mission['goal_angle']-yaw+math.pi)%(2*math.pi)-math.pi
            candidates=[i for i,e in enumerate(entries) if mask[i] and e['action']=='turn']
            return min(candidates,key=lambda i:abs(entries[i]['angle']-delta)) if candidates else 0
        if stage=='switch':return first('equip',slot=5) or 0
        if stage in ('door','breach','defuse'):
            action={'door':'door_open','breach':'door_breach','defuse':'defuse'}[stage]
            if bearing(a,env.mission['goal'])[1]>.6:return move_to(env.mission['goal'])
            return first(action) or 0
        if stage=='grenade':
            delta,_=bearing(a,env.mission['goal']);candidates=[i for i,e in enumerate(entries) if mask[i] and e['action']=='throw' and e['slot']==12]
            return min(candidates,key=lambda i:abs((entries[i]['angle']-delta+math.pi)%(2*math.pi)-math.pi)) if candidates else 0
        if stage=='rescue':
            if not env.rescue_followed:
                follow=first('follow')
                if follow is not None:return follow
                return move_to([-2.8,0,0])
            return move_to(env.mission['goal'])
        if stage=='reload':
            if env.shot_ever:return first('reload') or 0
            # Fire away from the opponent in this isolated exercise; rewards only completed reload.
            yaw=math.atan2(a['aim'][2],a['aim'][0])
            delta=(math.pi/2-yaw+math.pi)%(2*math.pi)-math.pi
            if not getattr(env,'reload_oriented',False):
                env.reload_oriented=True
                candidates=[i for i,e in enumerate(entries) if mask[i] and e['action']=='turn']
                return min(candidates,key=lambda i:abs(entries[i]['angle']-delta)) if candidates else 0
            if a.get('weapon_state')!=5:return 0
            return first('fire') or 0
        if a.get('ammo',0)==0:return first('reload') or 0
        targets=self.catalog.targets({'action':'aim_target'},obs)
        if targets:
            delta,dist=bearing(a,targets[0]['position'])
            if abs(delta)>.03 or a.get('weapon_state')!=5:return first('aim_target',target_index=0) or 0
            return first('fire') or 0
        # Training-only search rule: scan from the present observation, without hidden coordinates.
        if stage in ('shoot','elimination','loadout'):return first('turn',angle=math.pi/8) or 0
        return move_to(env.mission['goal'])
