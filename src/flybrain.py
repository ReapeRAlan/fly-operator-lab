"""Full MaleCNS LIF simulator. Anatomical graph fixed; dynamics are assumptions.

Reference: Shiu et al. 2024, Drosophila_brain_model (MIT).
This is not a validated physiological reconstruction, nor a trained tactical agent.
"""
from pathlib import Path
import json,time,math,hashlib
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[1]

@njit(cache=True)
def integrate(v,g,refractory,ring,ring_counts,cursor,indptr,indices,weights,signs,
              input_ids,pulses,dt,tau_m,tau_s,delay_steps,refractory_steps):
    n=len(v); counts=np.zeros(n,np.int32)
    em=math.exp(-dt/tau_m); es=math.exp(-dt/tau_s)
    coeff=tau_s/(tau_s-tau_m)*(es-em)
    for step in range(pulses.shape[0]):
        slot=cursor%ring.shape[0]
        for q in range(ring_counts[slot]):
            pre=ring[slot,q]; sign=signs[pre]
            for e in range(indptr[pre],indptr[pre+1]):
                g[indices[e]]+=weights[e]*sign
        ring_counts[slot]=0
        for j in range(len(input_ids)):
            if pulses[step,j]: v[input_ids[j]]+=68.75
        dest=(cursor+delay_steps)%ring.shape[0]
        for i in range(n):
            old_g=g[i]; g[i]=old_g*es
            if refractory[i]>0:
                refractory[i]-=1; v[i]=-52.0
            else:
                v[i]=-52.0+(v[i]+52.0)*em+old_g*coeff
                if v[i]>-45.0:
                    counts[i]+=1; v[i]=-52.0; g[i]=0.0
                    refractory[i]=refractory_steps
                    p=ring_counts[dest]; ring[dest,p]=i; ring_counts[dest]=p+1
        cursor+=1
    return cursor,counts

class FlyBrain:
    def __init__(self,directory=ROOT/'data/processed',seed=1,dt_ms=0.1):
        directory=Path(directory)
        self.ids=np.load(directory/'body_ids.npy',mmap_mode='r')
        self.indptr=np.load(directory/'indptr.npy',mmap_mode='r')
        self.indices=np.load(directory/'indices.npy',mmap_mode='r')
        counts=np.load(directory/'counts.npy',mmap_mode='r')
        self.weights=np.asarray(counts,dtype=np.float32)*np.float32(.275)
        self.signs=np.load(directory/'transmitter_sign.npy')
        self.dt=float(dt_ms)
        if not math.isfinite(self.dt) or self.dt<=0: raise ValueError('positive finite dt required')
        self.delay_steps=int(round(1.8/self.dt)); self.ref_steps=int(round(2.2/self.dt))
        if self.delay_steps<1 or abs(self.delay_steps*self.dt-1.8)>1e-6: raise ValueError('dt must divide 1.8 ms')
        self.seed=seed; self.reset(seed)
    def reset(self,seed=None):
        if seed is not None: self.seed=seed
        n=len(self.ids); self.v=np.full(n,-52.0,np.float32); self.g=np.zeros(n,np.float32)
        self.refractory=np.zeros(n,np.int32); self.ring=np.zeros((self.delay_steps+1,n),np.int32)
        self.ring_counts=np.zeros(self.delay_steps+1,np.int32); self.cursor=0
        self.rng=np.random.default_rng(self.seed)
    def step(self,input_ids,rates_hz,duration_ms=50.):
        input_ids=np.ascontiguousarray(input_ids,dtype=np.int32)
        rates=np.asarray(rates_hz,dtype=np.float64)
        if input_ids.ndim!=1 or rates.ndim!=1 or np.any(input_ids<0) or np.any(input_ids>=len(self.ids)): raise ValueError('input index out of range')
        if not math.isfinite(duration_ms) or duration_ms<=0: raise ValueError('positive finite duration required')
        if len(input_ids)!=len(rates) or len(np.unique(input_ids))!=len(input_ids): raise ValueError('unique inputs required')
        if not np.all(np.isfinite(rates)) or np.any(rates<0): raise ValueError('invalid rates')
        steps=int(round(duration_ms/self.dt))
        if abs(steps*self.dt-duration_ms)>1e-6: raise ValueError('duration must divide dt')
        pulses=self.rng.random((steps,len(input_ids))) < (-np.expm1(-rates*self.dt/1000.))
        start=time.perf_counter()
        self.cursor,counts=integrate(self.v,self.g,self.refractory,self.ring,self.ring_counts,self.cursor,self.indptr,self.indices,self.weights,self.signs,input_ids,pulses,self.dt,20.,5.,self.delay_steps,self.ref_steps)
        if not np.isfinite(self.v).all() or not np.isfinite(self.g).all(): raise FloatingPointError('non-finite neuronal activity')
        return counts,{'wall_seconds':time.perf_counter()-start,'simulated_ms':duration_ms,'neurons':len(self.ids),'edges':len(self.indices),'spikes':int(counts.sum()),'active_neurons':int(np.count_nonzero(counts)),'state_sha256':hashlib.sha256(self.v.tobytes()+self.g.tobytes()+self.refractory.tobytes()).hexdigest()}

class Adapter:
    """Explicit engineering mapping; action labels are NOT biological annotations."""
    ACTIONS=['move','stop','turn_left','turn_right','aim','fire','reload']
    def __init__(self,path=ROOT/'config/adapter.json'):
        self.config=json.loads(Path(path).read_text(encoding='utf-8'))
        self.inputs={k:np.asarray(v['indices'],np.int32) for k,v in self.config['inputs'].items()}
        self.outputs={k:np.asarray(v['indices'],np.int32) for k,v in self.config['outputs'].items()}
    def encode(self,obs):
        actor=obs['operator']; pos=np.array(actor['position']); facing=np.array(actor['aim'])
        angle=math.atan2(facing[2],facing[0]); channels={k:0. for k in self.inputs}
        for enemy in obs.get('visible_enemies',[]):
            d=np.array(enemy['position'])-pos; dist=float(np.linalg.norm(d[[0,2]]))
            bearing=(math.atan2(d[2],d[0])-angle+math.pi)%(2*math.pi)-math.pi
            side='left' if bearing<0 else 'right'
            channels['vision_'+side]=max(channels['vision_'+side],1./(1.+dist/10.))
        channels['proprioception']=min(1.,abs(actor.get('speed',0.))/2.)
        channels['damage']=min(1.,max(0.,actor.get('recent_damage',0.))/100.)
        channels['ammo']=max(0.,min(1.,actor.get('ammo',0)/max(1,actor.get('capacity',30))))
        channels['tonic']=.2
        ids=[]; rates=[]
        for key,indices in self.inputs.items(): ids.extend(indices); rates.extend([5.+145.*channels[key]]*len(indices))
        return np.asarray(ids,np.int32),np.asarray(rates,np.float64),channels
    def decode(self,spikes,obs,duration_ms):
        scores={k:float(spikes[idx].mean())*1000./duration_ms for k,idx in self.outputs.items()}
        # Highest normalized output population activity; fixed order breaks exact ties.
        action=max(self.ACTIONS,key=lambda k:scores[k]) if max(scores.values())>0 else 'stop'
        actor=obs['operator']; p=actor['position']; a=actor['aim']; angle=math.atan2(a[2],a[0])
        if action=='turn_left': angle-=math.pi/12
        if action=='turn_right': angle+=math.pi/12
        return {'action':action,'direction':[math.cos(angle),0.,math.sin(angle)],'destination':[p[0]+.4*math.cos(angle),p[1],p[2]+.4*math.sin(angle)]},scores
