from pathlib import Path
import json,os,time,shutil,msvcrt
import numpy as np,psutil
from lab_store import ROOT,RUNTIME,atomic_json
from windows_session import secondary_monitors,keep_system_awake
from runtime_config import load_runtime_config

class StopTraining(Exception):pass
class PilotFinished(Exception):pass

def configure_worker_process(process,config):
    """Apply scheduling limits; numerical thread count is configured separately."""
    available=process.cpu_affinity()
    requested=config.get('worker_cpu_affinity')
    if requested is None:
        affinity=available[-max(1,config.get('max_cpu_threads',2)):]
    else:
        if not isinstance(requested,list) or not requested or any(type(i) is not int or i not in available for i in requested):
            raise ValueError('worker_cpu_affinity must select available logical processors')
        affinity=sorted(set(requested))
    process.cpu_affinity(affinity)
    process.nice(psutil.IDLE_PRIORITY_CLASS if config.get('worker_priority')=='idle' else psutil.BELOW_NORMAL_PRIORITY_CLASS)

class ResourceGuard:
    def __init__(self,store,config,hours=None):
        self.store=store;self.config=config;self.start=time.time();self.deadline=self.start+3600*(hours or config['pilot_hours'])
        self.runtime_config=load_runtime_config();self.require_secondary_monitor=self.runtime_config['require_secondary_monitor']
        self.paused=False;self.stop=False;self.last_disk_check=0;self.disk_reason='';self.probe=False;self.display_missing=False
        self.on_pause=None;self.saved_pause=False;self.cooldown_until=0.;self.ram_low=False
        self.process=psutil.Process();configure_worker_process(self.process,config)
    OPERATIONAL_LIMITS=('minimum_available_ram_gb','resume_available_ram_gb','maximum_worker_ram_gb','minimum_free_disk_gb','artifact_quota_gb')
    def reload_limits(self):
        """Pick up edited resource limits without restarting; these keys never enter the semantic hash."""
        try:
            current=json.loads((ROOT/'config/learning.json').read_text(encoding='utf-8'))
            self.limits={k:current[k] for k in self.OPERATIONAL_LIMITS if k in current}
        except (OSError,ValueError):pass
    def reasons(self):
        if time.time()-getattr(self,'last_limits_check',0)>10:self.last_limits_check=time.time();self.reload_limits()
        c={**self.config,**getattr(self,'limits',{})};out=[];memory=psutil.virtual_memory()
        # Hysteresis avoids pause/save/resume flapping around a single RAM threshold.
        minimum=c['minimum_available_ram_gb'];resume=max(minimum,c.get('resume_available_ram_gb',minimum))
        self.ram_low=memory.available<(resume if getattr(self,'ram_low',False) else minimum)*1e9
        if self.ram_low:out.append(f"RAM disponible por debajo de {minimum} GB (reanuda con {resume} GB)")
        if self.process.memory_info().rss>c['maximum_worker_ram_gb']*1e9:out.append('Límite de memoria del entrenador')
        if time.time()-self.last_disk_check>10:
            self.last_disk_check=time.time();self.disk_reason=''
            self.display_missing=self.require_secondary_monitor and not secondary_monitors()
            if shutil.disk_usage(ROOT).free<c['minimum_free_disk_gb']*1e9:self.disk_reason='Espacio libre inferior a 8 GB'
            total=sum(p.stat().st_size for directory in (RUNTIME,ROOT/'work/frames',ROOT/'outputs') for p in directory.rglob('*') if p.is_file())
            if total>c['artifact_quota_gb']*1e9:self.disk_reason='Cuota de registros de 8 GB alcanzada'
        if self.disk_reason:out.append(self.disk_reason)
        if self.display_missing:out.append('Monitor secundario desconectado')
        return out
    def __call__(self):
        while True:
            for command in self.store.controls():
                a=command['action']
                if a=='pause':self.paused=True
                elif a=='resume':self.paused=False
                elif a=='stop':self.stop=True
                elif a=='probe':self.probe=True
            if self.stop:raise StopTraining()
            if time.time()>self.deadline:raise PilotFinished()
            reasons=self.reasons()
            if not reasons and not self.paused:
                remaining=self.cooldown_until-time.monotonic()
                if remaining>0:
                    time.sleep(min(.2,remaining));continue
                keep_system_awake(self.runtime_config['keep_display_awake'])
                self.saved_pause=False
                self.store.update(state='running',reason='',wall_seconds=time.time()-self.start,available_ram_gb=psutil.virtual_memory().available/1e9,
                  performance_profile=self.config.get('performance_profile','original'),cpu_affinity=self.process.cpu_affinity(),rest_after_step_seconds=self.config.get('minimum_rest_after_step_seconds',0),
                  worker_priority=self.config.get('worker_priority','below_normal'),max_cpu_threads=self.config.get('max_cpu_threads',2),pilot_deadline=self.deadline,
                  display_mode=self.runtime_config['display_mode'],require_secondary_monitor=self.require_secondary_monitor)
                return
            if not self.saved_pause and self.on_pause:
                self.on_pause();self.saved_pause=True
            keep_system_awake(False)
            self.store.update(state='paused',reason='Pausa solicitada' if self.paused else '; '.join(reasons),wall_seconds=time.time()-self.start,available_ram_gb=psutil.virtual_memory().available/1e9,
              display_mode=self.runtime_config['display_mode'],require_secondary_monitor=self.require_secondary_monitor)
            time.sleep(1)

    def after_step(self):
        self.cooldown_until=time.monotonic()+max(0.,self.config.get('minimum_rest_after_step_seconds',0.))

class WorkerLock:
    def __enter__(self):
        self.file=open(RUNTIME/'worker.lock','a+b');self.file.seek(0)
        if self.file.read(1)==b'':self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:self.file.close();raise RuntimeError('Ya existe un controlador de aprendizaje activo')
        return self
    def __exit__(self,*args):
        self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1);self.file.close()

class LiveState:
    """Read-only dashboard snapshots use a version marker; no shared game pointers."""
    def __init__(self,brain,run=''):
        self.brain=brain;self.run=str(run or '');self.revision=0;self.last_clock=-1;self.epoch=0
        self.neurons=np.lib.format.open_memmap(RUNTIME/'live_neurons.npy',mode='w+',dtype=np.float32,shape=(3,brain.n))
        self.audit_path=(RUNTIME/'campaigns'/self.run/'gain_audit.jsonl') if self.run else RUNTIME/'gain_audit.jsonl';self.audit_path.parent.mkdir(parents=True,exist_ok=True);self.audit_every=10
        self.gains=np.lib.format.open_memmap(RUNTIME/'live_gains.npy',mode='w+',dtype=np.float32,shape=(brain.edges,))
        self.gains[:]=brain.gains;brain.gains=self.gains
    def begin(self):atomic_json(RUNTIME/'live_version.json',{'writing':True,'revision':self.revision,'run':self.run})
    def end(self,counts,condition,seed):
        self.neurons[0]=self.brain.v;self.neurons[1]=self.brain.g;self.neurons[2]=counts;self.revision+=1
        if self.brain.cursor<=self.last_clock:self.epoch+=1
        self.last_clock=self.brain.cursor
        try:watched_neurons=json.loads((RUNTIME/'watch_neurons.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):watched_neurons=[]
        trace_path=RUNTIME/'neuron_trace.jsonl'
        with trace_path.open('a',encoding='utf-8') as f:
            for index in watched_neurons[:16]:
                if 0<=index<self.brain.n:f.write(json.dumps({'run':self.run,'body_id':int(self.brain.ids[index]),'v':float(self.brain.v[index]),'g':float(self.brain.g[index]),'spikes':int(counts[index]),'clock_ms':self.brain.cursor*self.brain.dynamics.dt_ms,'condition':condition,'seed':seed,'epoch':self.epoch,'wall_time':time.time()})+'\n')
        if trace_path.stat().st_size>5_000_000:
            rows=trace_path.read_text().splitlines()[-4000:];trace_path.write_text('\n'.join(rows)+'\n')
        try:watched=json.loads((RUNTIME/'watch_edges.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):watched=[]
        traces=[];b=self.brain;p=b.dynamics
        for edge in watched[:100]:
            if not 0<=edge<b.edges:continue
            pre=int(b.edgepre[edge]);post=int(b.indices[edge])
            traces.append({'edge_index':edge,'pre_trace':float(b.pretrace[pre]*np.exp(-(b.cursor-b.prelast[pre])*p.dt_ms/p.trace_ms)),
              'post_trace':float(b.posttrace[post]*np.exp(-(b.cursor-b.postlast[post])*p.dt_ms/p.trace_ms)),
              'eligibility':float(b.eligibility[edge]*np.exp(-(b.cursor-b.eligibility_last[edge])*p.dt_ms/p.eligibility_ms)),
              'gain':float(b.gains[edge]),'clock_ms':b.cursor*p.dt_ms,'condition':condition,'seed':seed,'run':self.run})
        atomic_json(RUNTIME/'live_edge_traces.json',traces)
        if self.revision==1 or self.revision%self.audit_every==0:
            gains=np.asarray(b.gains[::max(1,b.edges//65536)],dtype=np.float64)
            finite=gains[np.isfinite(gains)]
            if finite.size:
                stats={'run':self.run,'revision':self.revision,'epoch':self.epoch,'condition':condition,'seed':int(seed),'updated':time.time(),
                  'sample_size':int(finite.size),'mean':float(np.mean(finite)),'std':float(np.std(finite)),
                  'min':float(np.min(finite)),'p05':float(np.percentile(finite,5)),'median':float(np.median(finite)),
                  'p95':float(np.percentile(finite,95)),'max':float(np.max(finite)),
                  'below_floor':float(np.mean(finite<0.01)),'above_ceiling':float(np.mean(finite>10.0))}
                histogram,edges=np.histogram(np.clip(finite,0,10),bins=10,range=(0,10));stats['histogram']=[{'from':float(edges[i]),'to':float(edges[i+1]),'count':int(histogram[i])} for i in range(len(histogram))]
                with self.audit_path.open('a',encoding='utf-8') as f:f.write(json.dumps(stats,allow_nan=False)+'\n')
                if self.audit_path.stat().st_size>2_000_000:
                    rows=self.audit_path.read_text(encoding='utf-8').splitlines()[-2000:]
                    self.audit_path.write_text('\n'.join(rows)+'\n',encoding='utf-8')
        atomic_json(RUNTIME/'live_version.json',{'writing':False,'revision':self.revision,'run':self.run,'condition':condition,'seed':seed,'brain_clock_ms':self.brain.cursor*self.brain.dynamics.dt_ms,'updated':time.time()})
