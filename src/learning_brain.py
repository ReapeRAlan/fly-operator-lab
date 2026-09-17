"""Full classified MaleCNS LIF with event-based, reward-modulated STDP.

Connectivity/counts are immutable. Gains are separate engineering parameters.
STDP uses presynaptic *arrival* times (including the model's 1.8 ms delay).
No anatomical synapse or neuron is added, deleted, or dropped for performance.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import hashlib,json,math,time,os
import numpy as np
from numba import njit
from lossless_archive import atomic_npz

ROOT=Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Dynamics:
    dt_ms:float=.1
    tau_m_ms:float=20.
    tau_s_ms:float=5.
    rest_mv:float=-52.
    threshold_mv:float=-45.
    delay_ms:float=1.8
    refractory_ms:float=2.2
    input_pulse_mv:float=68.75
    unit_weight:float=.275
    trace_ms:float=20.
    eligibility_ms:float=2000.
    a_plus:float=1.
    a_minus:float=1.05
    learning_rate:float=1e-4
    gain_min:float=.25
    gain_max:float=4.

@njit(cache=True)
def incoming_index(indptr,indices,n):
    degree=np.zeros(n+1,np.int64)
    pre=np.empty(len(indices),np.int32)
    for i in range(n):
        for e in range(indptr[i],indptr[i+1]):
            degree[indices[e]+1]+=1;pre[e]=i
    for i in range(n):degree[i+1]+=degree[i]
    cursor=degree.copy();edges=np.empty(len(indices),np.int32)
    for e in range(len(indices)):
        j=indices[e];edges[cursor[j]]=e;cursor[j]+=1
    return degree,edges,pre

@njit(cache=True)
def tag_edge(e,change,clock,elig,last,listed,active,active_count,dt,tau):
    if change==0.:return active_count
    elig[e]=elig[e]*math.exp(-(clock-last[e])*dt/tau)+change
    last[e]=clock
    if not listed[e]:
        active[active_count]=e;active_count+=1;listed[e]=True
    return active_count

@njit(cache=True)
def integrate_learning(v,g,refractory,ring,ring_counts,cursor,indptr,indices,
                       baseline,gains,signs,input_ids,pulses,parameters,
                       inptr,inedges,edgepre,pretrace,posttrace,prelast,postlast,
                       elig,eliglast,listed,active,active_count,plasticity,
                       silence,event_ids,event_times,record_events,plastic_post):
    # plastic_post restricts eligibility bookkeeping to edges whose target may
    # learn. Membrane, synaptic and trace dynamics are identical for every neuron.
    dt,tm,ts,rest,threshold,pulse,delay,ref,ttrace,telig,ap,am=parameters
    em=math.exp(-dt/tm);es=math.exp(-dt/ts);coeff=ts/(ts-tm)*(es-em)
    n=len(v);counts=np.zeros(n,np.int32);events=0;fired=np.empty(n,np.int32)
    for step in range(pulses.shape[0]):
        slot=cursor%len(ring_counts)
        for q in range(ring_counts[slot]):
            pre=ring[slot,q]
            if silence[pre]:continue
            if plasticity:
                pretrace[pre]=pretrace[pre]*math.exp(-(cursor-prelast[pre])*dt/ttrace)+1.
                prelast[pre]=cursor
            for e in range(indptr[pre],indptr[pre+1]):
                post=indices[e]
                if silence[post]:continue
                g[post]+=baseline[e]*gains[e]*signs[pre]
                if plasticity and plastic_post[post]:
                    y=posttrace[post]*math.exp(-(cursor-postlast[post])*dt/ttrace)
                    active_count=tag_edge(e,-am*y,cursor,elig,eliglast,listed,active,active_count,dt,telig)
        ring_counts[slot]=0
        for j in range(len(input_ids)):
            if pulses[step,j] and not silence[input_ids[j]]:v[input_ids[j]]+=pulse
        dest=(cursor+int(delay))%len(ring_counts)
        # All neurons update before STDP postsynaptic updates: no index-order bias.
        nfired=0
        for i in range(n):
            oldg=g[i];g[i]=oldg*es
            if silence[i]:v[i]=rest;g[i]=0.;continue
            if refractory[i]>0:
                refractory[i]-=1;v[i]=rest
            else:
                v[i]=rest+(v[i]-rest)*em+oldg*coeff
                if v[i]>threshold:
                    counts[i]+=1;v[i]=rest;g[i]=0.;refractory[i]=int(ref)
                    p=ring_counts[dest];ring[dest,p]=i;ring_counts[dest]=p+1
                    fired[nfired]=i;nfired+=1
                    if record_events:
                        if events>=len(event_ids):raise ValueError('Spike event capacity exceeded')
                        event_ids[events]=i;event_times[events]=cursor;events+=1
        if plasticity:
            for q in range(nfired):
                post=fired[q]
                if plastic_post[post]:
                    for k in range(inptr[post],inptr[post+1]):
                        e=inedges[k];pre=edgepre[e]
                        if silence[pre]:continue
                        x=pretrace[pre]*math.exp(-(cursor-prelast[pre])*dt/ttrace)
                        active_count=tag_edge(e,ap*x,cursor,elig,eliglast,listed,active,active_count,dt,telig)
                posttrace[post]=posttrace[post]*math.exp(-(cursor-postlast[post])*dt/ttrace)+1.
                postlast[post]=cursor
        cursor+=1
    return cursor,counts,active_count,events

@njit(cache=True)
def modulate(gains,elig,last,active,active_count,clock,dt,tau,eta,delta,lo,hi):
    changed=0;absolute=0.;maximum=0.
    for k in range(active_count):
        e=active[k];value=elig[e]*math.exp(-(clock-last[e])*dt/tau)
        elig[e]=value;last[e]=clock
        old=gains[e];new=min(hi,max(lo,old+eta*delta*value))
        gains[e]=new;difference=abs(gains[e]-old)
        if difference>0.:changed+=1;absolute+=difference;maximum=max(maximum,difference)
    return changed,absolute,maximum

class LearningBrain:
    array_state=('v','g','refractory','ring','ring_counts','gains','pretrace','posttrace',
                 'prelast','postlast','eligibility','eligibility_last','listed','silence')
    def __init__(self,directory=ROOT/'data/processed',seed=7,dynamics=None,graph=None):
        self.directory=Path(directory);self.dynamics=dynamics or Dynamics()
        p=self.dynamics
        if not (0<p.dt_ms<min(p.tau_m_ms,p.tau_s_ms)) or p.tau_m_ms==p.tau_s_ms:raise ValueError('Invalid integration constants')
        if not (0<p.gain_min<=1<=p.gain_max) or p.learning_rate<0:raise ValueError('Invalid plasticity bounds')
        self.delay_steps=round(p.delay_ms/p.dt_ms);self.ref_steps=round(p.refractory_ms/p.dt_ms)
        if any(abs(round(x/p.dt_ms)*p.dt_ms-x)>1e-8 for x in (p.delay_ms,p.refractory_ms)):raise ValueError('dt must divide delay and refractory')
        if graph is None:
            self.ids=np.load(self.directory/'body_ids.npy',mmap_mode='r')
            self.indptr=np.load(self.directory/'indptr.npy',mmap_mode='r')
            self.indices=np.load(self.directory/'indices.npy',mmap_mode='r')
            self.counts=np.load(self.directory/'counts.npy',mmap_mode='r')
            self.signs=np.load(self.directory/'transmitter_sign.npy',mmap_mode='r')
        else:self.ids,self.indptr,self.indices,self.counts,self.signs=graph
        self.n=len(self.ids);self.edges=len(self.indices)
        self.graph_hash=self._graph_hash()
        if graph is None:
            baseline_path=self.directory/f'baseline_{self.graph_hash[:16]}_{p.unit_weight}.npy'
            if not baseline_path.exists():
                with baseline_path.with_suffix('.tmp').open('wb') as f:np.save(f,np.asarray(self.counts,np.float32)*p.unit_weight)
                os.replace(baseline_path.with_suffix('.tmp'),baseline_path)
            self.baseline=np.load(baseline_path,mmap_mode='r')
        else:self.baseline=np.asarray(self.counts,np.float32)*p.unit_weight
        self.gains=np.ones(self.edges,np.float32)
        if graph is None:
            cached=[self.directory/(name+'.npy') for name in ('incoming_indptr_v2','incoming_edges_v2','edge_pre_v2')]
            if not all(path.exists() for path in cached):
                arrays=incoming_index(self.indptr,self.indices,self.n)
                for path,array in zip(cached,arrays):
                    temporary=path.with_suffix('.tmp')
                    with temporary.open('wb') as f:np.save(f,array)
                    os.replace(temporary,path)
                del arrays
            self.inptr,self.inedges,self.edgepre=[np.load(path,mmap_mode='r') for path in cached]
        else:self.inptr,self.inedges,self.edgepre=incoming_index(self.indptr,self.indices,self.n)
        # Default: legacy whole-graph eligibility (global R-STDP). Causal motor rules
        # call set_plastic_posts so only edges entering learnable targets are tagged.
        self.plastic_post=np.ones(self.n,np.bool_);self.restricted_plasticity=False
        self.plasticity=False;self.updates=0;self.seed=int(seed);self.reset(seed)

    def set_plastic_posts(self,indices):
        """Restrict eligibility bookkeeping to edges whose postsynaptic neuron is listed.

        Spikes, membrane state, traces and every anatomical edge remain simulated.
        Eligibility values of retained edges are bit-identical to whole-graph tracking.
        """
        indices=np.asarray(indices,dtype=np.int64)
        if indices.ndim!=1 or len(indices)==0 or np.any(indices<0) or np.any(indices>=self.n):raise ValueError('Invalid plastic target indices')
        mask=np.zeros(self.n,np.bool_);mask[indices]=True
        self.plastic_post=mask;self.restricted_plasticity=True;self._restrict_active()

    def _restrict_active(self):
        if not self.restricted_plasticity or self.active_count==0:return
        candidates=self.active[:self.active_count]
        keep=self.plastic_post[np.asarray(self.indices[candidates],dtype=np.int64)]
        dropped=candidates[~keep]
        if len(dropped)==0:return
        self.eligibility[dropped]=0.;self.eligibility_last[dropped]=0;self.listed[dropped]=False
        retained=candidates[keep].copy();self.active_count=len(retained);self.active[:self.active_count]=retained

    def _graph_hash(self):
        h=hashlib.sha256()
        for a in (self.ids,self.indptr,self.indices,self.counts,self.signs):h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
        return h.hexdigest()

    def reset(self,seed=None):
        """Reset episodic activity. Learned gains deliberately persist."""
        if seed is not None:self.seed=int(seed)
        n=self.n;e=self.edges
        self.rng=np.random.default_rng(self.seed);self.cursor=0
        self.v=np.full(n,self.dynamics.rest_mv,np.float32);self.g=np.zeros(n,np.float32)
        self.refractory=np.zeros(n,np.int32)
        self.ring=np.zeros((self.delay_steps+1,n),np.int32);self.ring_counts=np.zeros(self.delay_steps+1,np.int32)
        self.pretrace=np.zeros(n,np.float32);self.posttrace=np.zeros(n,np.float32)
        self.prelast=np.zeros(n,np.int64);self.postlast=np.zeros(n,np.int64)
        self.eligibility=np.zeros(e,np.float32);self.eligibility_last=np.zeros(e,np.int32)
        self.listed=np.zeros(e,np.bool_);self.active=np.empty(e,np.int32);self.active_count=0
        self.silence=np.zeros(n,np.bool_);self.last_events=None

    def step(self,input_ids,rates_hz,duration_ms=50.,record=False,pulses=None):
        ids=np.ascontiguousarray(input_ids,dtype=np.int32);rates=np.asarray(rates_hz,float)
        if ids.ndim!=1 or rates.shape!=ids.shape or len(np.unique(ids))!=len(ids):raise ValueError('Unique one-dimensional input ports required')
        if np.any(ids<0) or np.any(ids>=self.n) or not np.isfinite(rates).all() or np.any(rates<0):raise ValueError('Invalid sensory stimulation')
        p=self.dynamics;steps=round(duration_ms/p.dt_ms)
        if not math.isfinite(duration_ms) or steps<1 or abs(steps*p.dt_ms-duration_ms)>1e-8:raise ValueError('Duration must be a positive multiple of dt')
        if self.cursor+steps>=np.iinfo(np.int32).max:raise ValueError('Episode exceeds eligibility timestamp capacity; reset before 59.6 hours of simulated time')
        if pulses is None:pulses=self.rng.random((steps,len(ids))) < -np.expm1(-rates*p.dt_ms/1000.)
        else:
            pulses=np.ascontiguousarray(pulses,dtype=np.bool_)
            if pulses.shape!=(steps,len(ids)):raise ValueError('Pulse shape mismatch')
        maximum=self.n*(math.ceil(steps/(self.ref_steps+1))+1) if record else 0
        event_ids=np.empty(maximum,np.int32);event_times=np.empty(maximum,np.int64)
        args=np.array([p.dt_ms,p.tau_m_ms,p.tau_s_ms,p.rest_mv,p.threshold_mv,p.input_pulse_mv,self.delay_steps,self.ref_steps,p.trace_ms,p.eligibility_ms,p.a_plus,p.a_minus])
        start=time.perf_counter()
        self.cursor,counts,self.active_count,nevents=integrate_learning(
          self.v,self.g,self.refractory,self.ring,self.ring_counts,self.cursor,self.indptr,self.indices,
          self.baseline,self.gains,self.signs,ids,pulses,args,self.inptr,self.inedges,self.edgepre,
          self.pretrace,self.posttrace,self.prelast,self.postlast,self.eligibility,self.eligibility_last,
          self.listed,self.active,self.active_count,self.plasticity,self.silence,event_ids,event_times,record,self.plastic_post)
        if not np.isfinite(self.v).all() or not np.isfinite(self.g).all():raise FloatingPointError('Non-finite neuronal state')
        self.last_events=(event_ids[:nevents].copy(),event_times[:nevents].copy()) if record else None
        return counts,dict(neurons=self.n,edges=self.edges,spikes=int(counts.sum()),active_neurons=int(np.count_nonzero(counts)),simulated_ms=duration_ms,
             wall_seconds=time.perf_counter()-start,plasticity=self.plasticity,eligible_edges=self.active_count,
             spikes_sha256=hashlib.sha256(counts.tobytes()).hexdigest())

    def reward(self,prediction_error):
        if not np.isfinite(prediction_error):raise ValueError('Non-finite reward prediction error')
        if not self.plasticity:return dict(changed_edges=0,absolute_change=0.,max_change=0.,delta=0.)
        if self.restricted_plasticity:raise ValueError('Global R-STDP requires whole-graph eligibility; plasticity is restricted to motor targets')
        p=self.dynamics;delta=float(np.clip(prediction_error,-1.,1.))
        changed,absolute,maximum=modulate(self.gains,self.eligibility,self.eligibility_last,self.active,self.active_count,self.cursor,p.dt_ms,p.eligibility_ms,p.learning_rate,delta,p.gain_min,p.gain_max)
        self.updates+=1
        return dict(changed_edges=int(changed),absolute_change=float(absolute),max_change=float(maximum),delta=delta)

    def capture_motor_eligibility(self,descending_indices):
        """Freeze causal eligibility at decision time for edges entering motor output neurons.

        The following observation is integrated only after this compact snapshot is
        made, so it cannot receive credit for an action that preceded it.
        """
        if not self.plasticity or self.active_count==0:
            return {'version':1,'cursor':int(self.cursor),'edges':np.empty(0,np.int32),
                    'posts':np.empty(0,np.int32),'values':np.empty(0,np.float32)}
        descending=np.asarray(descending_indices,dtype=np.int32)
        motor=np.zeros(self.n,np.bool_);motor[descending]=True
        candidates=self.active[:self.active_count]
        posts=np.asarray(self.indices[candidates],dtype=np.int32)
        keep=motor[posts]
        edges=np.asarray(candidates[keep],dtype=np.int32).copy();posts=posts[keep].copy()
        elapsed=(self.cursor-np.asarray(self.eligibility_last[edges],dtype=np.int64))*self.dynamics.dt_ms
        values=(np.asarray(self.eligibility[edges],dtype=np.float32)*
                np.exp(-elapsed/self.dynamics.eligibility_ms)).astype(np.float32)
        finite=np.isfinite(values)&(values!=0)
        return {'version':1,'cursor':int(self.cursor),'edges':edges[finite],
                'posts':posts[finite],'values':values[finite]}

    def reward_motor(self,snapshot,prediction_error,descending_indices,descending_credit,sign_aware=False):
        """Apply R-STDP only to causal motor-cone eligibilities, weighted by d log pi/ d DN.

        v1 moves every gain in the credited direction. For an inhibitory presynaptic
        neuron a larger gain *lowers* descending drive, so v2 (sign_aware) multiplies
        the change by the presynaptic sign: the effective weight then moves the
        descending neuron in the credited direction while the sign itself is kept.
        """
        rule='causal_motor_rstdp_v2' if sign_aware else 'causal_motor_rstdp_v1'
        if not np.isfinite(prediction_error):raise ValueError('Non-finite reward prediction error')
        if not self.plasticity:
            return dict(changed_edges=0,absolute_change=0.,max_change=0.,delta=0.,candidate_edges=0,rule=rule)
        if snapshot.get('version')!=1 or snapshot.get('cursor',self.cursor)>self.cursor:raise ValueError('Invalid eligibility snapshot')
        edges=np.asarray(snapshot['edges'],dtype=np.int32);posts=np.asarray(snapshot['posts'],dtype=np.int32)
        values=np.asarray(snapshot['values'],dtype=np.float32);descending=np.asarray(descending_indices,dtype=np.int32)
        credit=np.asarray(descending_credit,dtype=np.float32)
        if edges.shape!=posts.shape or edges.shape!=values.shape or credit.shape!=descending.shape:raise ValueError('Motor credit shape mismatch')
        if not np.isfinite(values).all() or not np.isfinite(credit).all():raise ValueError('Non-finite motor credit')
        delta=float(np.clip(prediction_error,-1.,1.));dense=np.zeros(self.n,np.float32);dense[descending]=credit
        credit_delay_ms=(self.cursor-int(snapshot['cursor']))*self.dynamics.dt_ms
        eligibility_decay=float(math.exp(-credit_delay_ms/self.dynamics.eligibility_ms))
        values=values*eligibility_decay
        old=np.asarray(self.gains[edges],dtype=np.float32).copy()
        change=self.dynamics.learning_rate*delta*values*dense[posts]
        if sign_aware:change=change*np.asarray(self.signs[np.asarray(self.edgepre[edges],dtype=np.int64)],dtype=np.float32)
        new=np.clip(old+change,self.dynamics.gain_min,self.dynamics.gain_max).astype(np.float32)
        self.gains[edges]=new;difference=np.abs(new-old);changed=int(np.count_nonzero(difference))
        self.updates+=1
        return dict(changed_edges=changed,absolute_change=float(difference.sum(dtype=np.float64)),
                    max_change=float(difference.max(initial=0.)),delta=delta,candidate_edges=int(len(edges)),
                    credit_nonzero=int(np.count_nonzero(credit)),credit_abs_mean=float(np.abs(credit).mean()),
                    snapshot_cursor=int(snapshot['cursor']),credit_delay_ms=float(credit_delay_ms),
                    eligibility_decay=eligibility_decay,rule=rule)

    def save(self,path,extra=None):
        """Atomic, non-pickle checkpoint including delay queue and RNG."""
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        metadata=dict(version=2,graph_hash=self.graph_hash,dynamics=asdict(self.dynamics),cursor=self.cursor,
          seed=self.seed,rng=self.rng.bit_generator.state,plasticity=self.plasticity,updates=self.updates,extra=extra or {})
        metadata['empty_eligibility']=self.active_count==0
        zero_names={'eligibility','eligibility_last','listed'} if self.active_count==0 else set()
        arrays={name:getattr(self,name) for name in self.array_state if name not in zero_names}
        arrays['active']=self.active[:self.active_count]
        arrays['metadata']=np.array(json.dumps(metadata,allow_nan=False))
        atomic_npz(path,arrays)
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1048576),b''):digest.update(chunk)
        return digest.hexdigest()

    def load(self,path):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['version']!=2 or meta['graph_hash']!=self.graph_hash:raise ValueError('Checkpoint graph/version mismatch')
            if meta['dynamics']!=asdict(self.dynamics):raise ValueError('Checkpoint dynamics mismatch')
            for name in self.array_state:
                if meta.get('empty_eligibility') and name in ('eligibility','eligibility_last','listed'):
                    if self.active_count:getattr(self,name).fill(0)
                    continue
                a=z[name]
                # Lossless migration of earlier v2 timestamp storage; clock range is checked.
                if name=='eligibility_last' and a.dtype==np.int64 and a.max(initial=0)<=np.iinfo(np.int32).max:a=a.astype(np.int32)
                if a.shape!=getattr(self,name).shape or a.dtype!=getattr(self,name).dtype:raise ValueError('Checkpoint array mismatch: '+name)
                if a.dtype.kind=='f' and not np.isfinite(a).all():raise ValueError('Invalid checkpoint state')
                getattr(self,name)[:]=a
            self.active_count=len(z['active']);self.active[:self.active_count]=z['active']
            self.cursor=meta['cursor'];self.seed=meta['seed'];self.updates=meta['updates']
            self.plasticity=meta['plasticity'];self.rng.bit_generator.state=meta['rng']
            if np.any(self.gains<self.dynamics.gain_min) or np.any(self.gains>self.dynamics.gain_max):raise ValueError('Checkpoint gain bound violation')
        # Whole-graph checkpoints remain loadable; non-learnable eligibilities are discarded.
        self._restrict_active()
        return meta['extra']
