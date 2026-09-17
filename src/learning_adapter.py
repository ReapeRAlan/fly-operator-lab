"""Versioned, deliberately engineered sensory ports and discrete action vocabulary."""
from pathlib import Path
import json,math
import numpy as np
import pyarrow.feather as feather
ROOT=Path(__file__).resolve().parents[1]
STAGES=['move','stance','cancel','orient','switch','shoot','reload','door','breach','grenade','elimination','rescue','defuse','loadout']

def clean_observation(raw,mission):
    """Allow-list. Never pass the debug all-humans list or enemy HP/ammo to the actor."""
    actor=raw.get('operator') or {}
    visible=set(raw.get('visible_ids',[]))
    enemies=[{'id':h['id'],'position':h['position'],'alive':h.get('health',0)>0} for h in raw.get('visible_enemies',[])]
    friends=[{'id':h['id'],'position':h['position'],'kind':h['kind']} for h in raw.get('humans',[]) if h['id'] in visible and h['kind']!=1 and h['id']!=actor.get('id')]
    return dict(operator=actor,visible_enemies=enemies,visible_friendlies=friends,objects=raw.get('objects',[]),
      inventory=raw.get('inventory',[]),sim_time_ms=raw.get('sim_time_ms',0),episode=raw.get('episode'),sequence=raw.get('sequence'),
      action_receipt=raw.get('action_receipt',{}),goal=mission['goal'],goal_angle=mission['goal_angle'],
      stage=mission['stage'],time_limit_ms=mission.get('max_seconds',30)*1000,
      known_geometry=mission.get('geometry',[]),loadout_choice=raw.get('loadout_choice',-1))

def bearing(actor,point):
    dx=point[0]-actor['position'][0];dz=point[2]-actor['position'][2]
    a=actor['aim'];heading=math.atan2(a[2],a[0])
    return ((math.atan2(dz,dx)-heading+math.pi)%(2*math.pi)-math.pi),math.hypot(dx,dz)


def native_progress(obs):
    """True while the engine is advancing an existing command or visibility tick."""
    receipt=obs.get('action_receipt',{})
    return bool((obs.get('sim_time_ms',0)==0 and obs.get('stage')!='loadout') or
                obs.get('operator',{}).get('commands',0)>0 or receipt.get('status')=='in_progress')


def decision_required(obs,mask):
    """A busy tick is cognitive only when an explicit intervention is available."""
    return bool(not native_progress(obs) or np.count_nonzero(np.asarray(mask,dtype=bool))>1)

SPATIAL_PREFIXES=('enemy_','friend_','hostage_','civilian_','door_','goal_','wall_')

def rank_gateways():
    """Excitatory gateway candidates ranked by one/two-hop anatomical routes to descending neurons."""
    df=feather.read_table(ROOT/'data/processed/neurons.feather').to_pandas()
    from scipy.sparse import csr_matrix
    folder=ROOT/'data/processed';indptr=np.load(folder/'indptr.npy',mmap_mode='r');indices=np.load(folder/'indices.npy',mmap_mode='r')
    weights=np.load(folder/'counts.npy',mmap_mode='r');signs=np.load(folder/'transmitter_sign.npy',mmap_mode='r')
    graph=csr_matrix((np.asarray(weights,np.float32),indices,indptr),shape=(len(df),len(df)))
    descending=(df.superclass=='descending_neuron').to_numpy(np.float32)
    direct=np.asarray(graph.dot(descending)).ravel();score=direct+np.asarray(graph.dot(np.log1p(direct))).ravel()
    def ranked(superclasses):
        candidates=df.index[df.superclass.isin(superclasses)].to_numpy()
        candidates=candidates[(signs[candidates]>0)&(score[candidates]>0)]
        # Stable bodyId tie-break makes regeneration exactly reproducible.
        return candidates[np.lexsort((df.loc[candidates,'bodyId'].to_numpy(),-score[candidates]))]
    return df,score,ranked(['visual_projection']),ranked(['vnc_sensory','cb_sensory'])

def assign_ports(channels,allocation_order,ports=16):
    """Give each channel unique gateways; channels earlier in allocation_order receive stronger routes."""
    if sorted(allocation_order)!=sorted(channels):raise ValueError('Allocation order must be a permutation of the channels')
    df,score,visual,body=rank_gateways();vi=bi=0;mapping={}
    for key in allocation_order:
        if key.startswith(SPATIAL_PREFIXES):indices=visual[vi:vi+ports];vi+=ports
        else:indices=body[bi:bi+ports];bi+=ports
        if len(indices)!=ports:raise ValueError('Insufficient annotated sensory neurons')
        mapping[key]={'indices':indices.tolist(),'body_ids':df.loc[indices,'bodyId'].astype(int).tolist(),
                      'gateway_superclasses':sorted(df.loc[indices,'superclass'].unique().tolist()),
                      'descending_route_score':[float(score[i]) for i in indices]}
    return {key:mapping[key] for key in channels}

class SensoryEncoder:
    channels=([f'enemy_{i}' for i in range(16)]+[f'friend_{i}' for i in range(8)]+
              [f'hostage_{i}' for i in range(8)]+[f'civilian_{i}' for i in range(8)]+
              [f'door_{i}' for i in range(8)]+[f'goal_{i}' for i in range(8)]+[f'wall_{i}' for i in range(8)]+
              ['health','ammo','speed','damage','ready','busy','crouched','primary','secondary','flash','frag','tool','tonic',
               'orientation_sin','orientation_cos','goal_bearing_sin','goal_bearing_cos','goal_distance','time_remaining','action_in_progress']+
              ['task_'+s for s in STAGES])
    def __init__(self,path=ROOT/'data/learning/sensory_ports_v31.json'):
        self.path=Path(path)
        if self.path.exists():config=json.loads(self.path.read_text())
        else:
            # Choose unique excitatory gateway neurons with measured anatomical
            # routes toward descending neurons. This is an engineered interface,
            # recorded in the schema; no anatomical edge is changed or omitted.
            mapping=assign_ports(self.channels,self.channels)
            config={'version':'3.1','mapping':mapping,'meaning':'Fixed engineered semantic channels injected through unique excitatory gateway neurons ranked by one/two-hop anatomical routes to descending neurons. These are not biological weapon/action concepts. Hostages and civilians are explicit. All classified neurons and edges remain in the simulation.',
                    'port_selection':'published connectivity counts define ranking; transmitter sign prediction selects excitatory gateways; semantic assignment is engineered',
                    'baseline_hz':5.,'gain_hz':145.}
            self.path.parent.mkdir(parents=True,exist_ok=True);self.path.write_text(json.dumps(config,indent=2))
        if list(config['mapping'])!=self.channels:raise ValueError('Sensory schema mismatch')
        self.config=config;self.ids=np.array([i for k in self.channels for i in config['mapping'][k]['indices']],np.int32)
        overrides=(config.get('encoding') or {}).get('channel_gain_hz',{})
        if set(overrides)-set(self.channels):raise ValueError('Gain override for unknown sensory channel')
        self.gains_hz=np.array([overrides.get(k,config['gain_hz']) for k in self.channels],float)
    def encode(self,obs):
        c=dict.fromkeys(self.channels,0.);a=obs['operator']
        encoding=self.config.get('encoding') or {}
        tuned=encoding.get('version')=='3.2'
        def sector(prefix,bins,angle,salience):
            if not tuned:
                i=int(((angle+math.pi)/(2*math.pi))*bins)%bins
                c[f'{prefix}_{i}']=max(c[f'{prefix}_{i}'],salience);return
            # von Mises tuning around each sector centre; kappa scales so neighbouring
            # sectors respond equally for 8 and 16 bins. Channel i keeps its v3.1 centre.
            kappa=encoding['sector_kappa']*(bins/8)**2
            for i in range(bins):
                centre=-math.pi+(i+.5)*2*math.pi/bins
                c[f'{prefix}_{i}']=max(c[f'{prefix}_{i}'],salience*math.exp(kappa*(math.cos(angle-centre)-1)))
        spatial=[('enemy',obs['visible_enemies'],16),('friend',obs['visible_friendlies'],8),
                 ('hostage',[x for x in obs['visible_friendlies'] if x.get('kind') in (2,4)],8),
                 ('civilian',[x for x in obs['visible_friendlies'] if x.get('kind')==3],8),
                 ('door',[x for x in obs['objects'] if x['kind']=='door'],8)]
        for prefix,values,bins in spatial:
            for obj in values:
                if obj.get('alive') is False:continue
                angle,dist=bearing(a,obj.get('initial_position',obj['position']))
                sector(prefix,bins,angle,1/(1+dist/10))
        angle,dist=bearing(a,obs['goal'])
        # v3.1 goal salience fell to zero at the goal; v3.2 keeps direction salient everywhere.
        sector('goal',8,angle,encoding['goal_salience'] if tuned else min(1.,dist/5.))
        # Known blueprint walls; eight egocentric ray intersections, range 12 m.
        origin=np.array(a['position'])[[0,2]];heading=math.atan2(a['aim'][2],a['aim'][0])
        for i in range(8):
            d=np.array([math.cos(heading+i*math.pi/4),math.sin(heading+i*math.pi/4)])
            best=12.
            for geo in obs['known_geometry']:
                points=geo.get('points',[])
                for p,q in zip(points,points[1:]):
                    p=np.asarray(p);segment=np.asarray(q)-p;den=d[0]*segment[1]-d[1]*segment[0]
                    if abs(den)<1e-8:continue
                    offset=p-origin;t=(offset[0]*segment[1]-offset[1]*segment[0])/den
                    u=(offset[0]*d[1]-offset[1]*d[0])/den
                    if 0<=t<=best and 0<=u<=1:best=t
            c[f'wall_{i}']=1-best/12.
        inventory={e['slot']:e for e in obs['inventory']}
        goal_angle,goal_distance=bearing(a,obs['goal']);duration=max(1.,float(obs.get('time_limit_ms',1.)))
        receipt=obs.get('action_receipt',{})
        c.update(health=np.clip(a.get('health',0)/100,0,1),ammo=np.clip(a.get('ammo',0)/max(1,a.get('capacity',1)),0,1),
          speed=np.clip(a.get('speed',0)/3,0,1),damage=np.clip(a.get('recent_damage',0)/100,0,1),ready=float(a.get('weapon_state')==5),
          busy=float(a.get('commands',0)>0 or receipt.get('status')=='in_progress'),crouched=float(a.get('crouched',False)),
          primary=float(inventory.get(1,{}).get('equipped',False)),secondary=float(inventory.get(5,{}).get('equipped',False)),
          flash=min(1,inventory.get(12,{}).get('quantity',0)/2),frag=min(1,inventory.get(13,{}).get('quantity',0)/2),tool=float(16 in inventory),tonic=.2,
          goal_bearing_sin=(math.sin(goal_angle)+1)/2,goal_bearing_cos=(math.cos(goal_angle)+1)/2,
          goal_distance=np.clip(goal_distance/20,0,1),time_remaining=np.clip(1-obs.get('sim_time_ms',0)/duration,0,1),
          action_in_progress=float(receipt.get('status')=='in_progress'))
        delta=obs['goal_angle']-heading;c['orientation_sin']=(math.sin(delta)+1)/2;c['orientation_cos']=(math.cos(delta)+1)/2
        c['task_'+obs['stage']]=1.
        if tuned:c['goal_distance']=math.exp(-goal_distance/encoding['goal_proximity_m']) # proximity: 1 at the goal
        values=np.clip(np.array([c[k] for k in self.channels],float),0,1)
        rates=np.repeat(self.config['baseline_hz']+self.gains_hz*values,16)
        return self.ids,rates,{k:float(v) for k,v in zip(self.channels,values)}

class NeuralReadout:
    def __init__(self):
        df=feather.read_table(ROOT/'data/processed/neurons.feather',columns=['superclass','bodyId']).to_pandas()
        self.indices=df.index[df.superclass=='descending_neuron'].to_numpy(np.int32)
        self.body_ids=df.loc[self.indices,'bodyId'].to_numpy(np.int64)
        if len(self.indices)!=1314:raise ValueError('Unexpected descending-neuron inventory')
        self.taus=np.array([100.,500.,2000.],np.float32);self.reset()
    def reset(self):self.filters=np.zeros((3,len(self.indices)),np.float32)
    def update(self,counts,dt=50.):
        rate=counts[self.indices].astype(np.float32)*1000./dt
        decay=np.exp(-dt/self.taus)[:,None];self.filters[:]=self.filters*decay+rate[None,:]*(1-decay)
        # Population normalization keeps the three time scales numerically
        # comparable without learning statistics from validation episodes.
        logged=np.log1p(self.filters)
        normalized=(logged-logged.mean(axis=1,keepdims=True))/np.maximum(logged.std(axis=1,keepdims=True),1e-4)
        return np.clip(normalized,-5.,5.).astype(np.float32).reshape(-1).copy()
    def summary(self,counts):
        order=np.argsort(counts[self.indices])[-24:][::-1]
        return [{'body_id':int(self.body_ids[i]),'index':int(self.indices[i]),'spikes':int(counts[self.indices[i]]),'rate_hz':float(self.filters[0,i])} for i in order]

class SensorReadout:
    """Brain-free control features: the injected channel rates through the DN readout filters.

    Same three time constants, log1p and per-filter population z-score as NeuralReadout,
    applied to 98 channel rates instead of 1,314 descending neurons (294 features).
    """
    def __init__(self,channels):
        self.channels=len(channels);self.taus=np.array([100.,500.,2000.],np.float32);self.reset()
    def reset(self):self.filters=np.zeros((3,self.channels),np.float32)
    def update(self,rates_hz,dt=50.):
        rate=np.asarray(rates_hz,np.float32)
        decay=np.exp(-dt/self.taus)[:,None];self.filters[:]=self.filters*decay+rate[None,:]*(1-decay)
        logged=np.log1p(self.filters)
        normalized=(logged-logged.mean(axis=1,keepdims=True))/np.maximum(logged.std(axis=1,keepdims=True),1e-4)
        return np.clip(normalized,-5.,5.).astype(np.float32).reshape(-1).copy()

def sector_of(angle,bins):
    """Egocentric sector index with the encoder's convention (sector i centred at -pi+(i+.5)*2pi/bins)."""
    return int(((angle+math.pi)/(2*math.pi))*bins)%bins

# Protocol 3.3: targeted actions address the egocentric sector the sensors represent,
# not the n-th object by sorted engine ID (which the actor cannot observe).
TARGET_SECTORS={'aim_target':16,'door_open':8,'door_breach':8,'defuse':8,'follow':8,'use':8,'clear_obstacle':8,'arrest':8,'spy_camera':8}
INTERVENTIONS=('stop','cancel')

class ActionCatalog:
    def __init__(self):
        self.entries=[{'action':x} for x in ('wait','stop','cancel','reload','fire')]
        self.entries +=[{'action':'equip','slot':s} for s in (1,5)]
        self.entries +=[{'action':'crouch','value':v} for v in (True,False)]
        self.entries +=[{'action':'move','angle':i*math.pi/4,'distance':.75} for i in range(8)]
        self.entries +=[{'action':'turn','angle':i*math.pi/8} for i in range(-8,8) if i]
        self.entries +=[{'action':'aim_target','sector':i,'bins':16} for i in range(16)]
        self.entries +=[{'action':'door_open','sector':i,'bins':8} for i in range(8)]
        self.entries +=[{'action':'door_breach','sector':i,'bins':8,'slot':s} for i in range(8) for s in (14,15,16)]
        self.entries +=[{'action':'throw','slot':s,'angle':i*math.pi/4,'distance':8.} for s in (12,13) for i in range(8)]
        self.entries +=[{'action':a,'sector':i,'bins':8} for a in ('defuse','follow','use','clear_obstacle','arrest') for i in range(8)]
        self.entries +=[{'action':'spy_camera','sector':i,'bins':8,'slot':14} for i in range(8)]
        self.entries +=[{'action':'evacuate'}]
        self.entries +=[{'action':'loadout','kit':i} for i in range(3)]
    def targets(self,entry,obs):
        """All candidate objects for a targeted action family, in no meaningful order."""
        a=entry['action']
        if a in ('aim_target','arrest'):return [x for x in obs['visible_enemies'] if x.get('alive',True)]
        if a=='follow':return list(obs['visible_friendlies'])
        if a in ('door_open','door_breach','spy_camera'):return [x for x in obs['objects'] if x['kind']=='door']
        if a=='defuse':return [x for x in obs['objects'] if 'bomb' in x['template'].lower()]
        return [x for x in obs['objects'] if x['kind']!='door' and 'bomb' not in x['template'].lower()]
    def target(self,entry,obs):
        """Nearest candidate inside the entry's egocentric sector, located as the encoder locates it."""
        actor=obs['operator'];best=None
        for item in self.targets(entry,obs):
            angle,distance=bearing(actor,item.get('initial_position',item['position']))
            if sector_of(angle,entry['bins'])!=entry['sector']:continue
            if best is None or distance<best[0]:best=(distance,item)
        return None if best is None else best[1]
    def sector_entry(self,action,obs,point,**attributes):
        """Index of the targeted entry whose sector contains point (used by the instructor)."""
        angle,_=bearing(obs['operator'],point)
        for i,e in enumerate(self.entries):
            if e['action']==action and 'sector' in e and e['sector']==sector_of(angle,e['bins']) and all(e.get(k)==v for k,v in attributes.items()):return i
        return None
    def family_mask(self,families=None):
        """Curriculum help: which action families a stage exposes. Never depends on state."""
        out=np.array([families is None or e['action'] in families for e in self.entries],bool);out[0]=True
        return out
    def legal_mask(self,obs,capabilities=None):
        """What the engine permits now (restricted to natively validated actions), with no task hints."""
        out=np.zeros(len(self.entries),bool);out[0]=True
        if obs['operator'].get('health',0)<=0:return out
        actor=obs['operator'];inventory={x['slot']:x for x in obs['inventory']};pending=obs.get('action_receipt',{}).get('status')=='in_progress'
        queue=actor.get('commands',0)>0
        # The first visibility tick is an engine transition: nothing can be issued yet.
        if obs.get('sim_time_ms',0)==0 and obs.get('stage')!='loadout':return out
        if queue or pending:
            # While a native command runs, continuing (wait), stopping and cancelling are all
            # legal whenever the engine allows them; the mask no longer decides when to stop.
            for i,e in enumerate(self.entries):
                if e['action'] in INTERVENTIONS and (capabilities is None or e['action'] in capabilities):out[i]=True
            return out
        for i,e in enumerate(self.entries):
            a=e['action'];valid=True
            if capabilities is not None and a not in capabilities:continue
            if a=='fire':valid=actor.get('weapon_state')==5 and actor.get('ammo',0)>0
            elif a=='reload':valid=0<=actor.get('ammo',-1)<actor.get('capacity',0) and not queue and actor.get('weapon_state')!=10
            elif a=='cancel':valid=queue or pending
            elif a=='equip':valid=e['slot'] in inventory
            elif a=='throw':valid=inventory.get(e['slot'],{}).get('type')==3 and inventory[e['slot']]['quantity']>0
            if 'sector' in e:
                target=self.target(e,obs);valid=valid and target is not None
                if valid and a!='aim_target':
                    valid=bearing(actor,target.get('initial_position',target['position']))[1]<=3.
            if a=='door_breach':valid=valid and inventory.get(e['slot'],{}).get('type') in (6,7,8,9,10,11,15,20)
            if a=='spy_camera':valid=valid and inventory.get(e['slot'],{}).get('type')==4
            if a=='evacuate':valid=False # requires an explicit native evacuation-zone proof before enabling
            if a=='loadout':valid=obs['sim_time_ms']==0 and obs.get('loadout_choice',-1)==-1 and obs['stage']=='loadout'
            if obs['stage']=='loadout' and obs['sim_time_ms']==0 and a not in ('loadout','wait'):valid=False
            out[i]=valid
        out[0]=True;return out
    def mask(self,obs,capabilities=None,families=None):
        return self.legal_mask(obs,capabilities)&self.family_mask(families)
    def decode(self,index,obs):
        e=self.entries[int(index)];a=e['action'];actor=obs['operator'];heading=math.atan2(actor['aim'][2],actor['aim'][0])
        result={k:v for k,v in e.items() if k not in ('angle','distance','sector','bins')}
        if a in ('move','turn','throw'):
            angle=heading+e['angle'];direction=[math.cos(angle),0.,math.sin(angle)]
            result['direction']=direction
            if a=='turn':result['action']='turn_left' if e['angle']<0 else 'turn_right'
            else:result['destination']=[actor['position'][0]+e['distance']*direction[0],actor['position'][1],actor['position'][2]+e['distance']*direction[2]]
        if 'sector' in e:
            target=self.target(e,obs)
            if target is None:raise ValueError('Targeted action without a target in its sector')
            result['target_id']=target['id']
            if a=='aim_target':
                dx=target['position'][0]-actor['position'][0];dz=target['position'][2]-actor['position'][2];norm=max(1e-8,math.hypot(dx,dz))
                result={'action':'aim','direction':[dx/norm,0.,dz/norm]}
            elif a=='spy_camera':result['direction']=actor['aim']
        return result
    def label(self,i):
        e=self.entries[int(i)];parts=[e['action']]
        if 'slot' in e:parts.append('slot '+str(e['slot']))
        if 'kit' in e:parts.append('kit '+str(e['kit']))
        if 'sector' in e:parts.append(f"sector {e['sector']}/{e['bins']}")
        if 'angle' in e:parts.append(str(round(math.degrees(e['angle'])))+'°')
        return ' · '.join(parts)
