"""Local read-only scientific data API plus an explicit trainer control queue."""
from pathlib import Path
from functools import lru_cache
from dataclasses import asdict
import io,json,sqlite3,time,math,zipfile,csv,threading,gc,os,psutil
import numpy as np,pyarrow.feather as feather
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,Response
from fastapi.staticfiles import StaticFiles
from lab_store import ROOT,RUNTIME,connect,atomic_json
from learning_brain import Dynamics
import lab_history
app=FastAPI(title='Fly Operator Observatory',docs_url='/api/docs')
HEAVY_CACHE_IDLE_SECONDS=120
_heavy_cache_last_access=0.

def read(path,default=None):
    # Windows can briefly deny a reader while atomic_json replaces a live file.
    # Retry that transient state so dashboard polling never surfaces a false 500.
    for attempt in range(4):
        try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except PermissionError:
            if attempt<3:time.sleep(.01);continue
        except (FileNotFoundError,json.JSONDecodeError):break
    return {} if default is None else default

LIVE_FRESH_SECONDS=10.
LIVE_STATES={'running','preparing','saving'}

def configured_run():
    return str(read(ROOT/'config/learning.json').get('experiment_id','pilot-v2'))

def _stored_latest(db,run):
    row=db.execute('SELECT id,payload FROM steps WHERE run=? ORDER BY id DESC LIMIT 1',(run,)).fetchone()
    if row is None:return {}
    item=json.loads(row[1]);item['record_id']=row[0]
    decision=item.get('decision') or {}
    if 'mask' in decision and 'action_labels' not in decision:
        from learning_adapter import ActionCatalog
        decision['action_labels']=[ActionCatalog().label(i) for i in range(len(decision['mask']))]
    return item

def _known_run(db,run):
    return db.execute('SELECT 1 FROM steps WHERE run=? LIMIT 1',(run,)).fetchone() is not None or db.execute('SELECT 1 FROM episodes WHERE run=? LIMIT 1',(run,)).fetchone() is not None

def run_context(db,requested_run=''):
    """Resolve one dashboard campaign without ever borrowing another run's live files."""
    configured=configured_run();selected=str(requested_run or configured)
    if selected!=configured and not _known_run(db,selected):
        raise HTTPException(404,'Unknown campaign')
    status=read(RUNTIME/'status.json');latest=read(RUNTIME/'latest.json');version=read(RUNTIME/'live_version.json')
    now=time.time();updated=version.get('updated');age=None if updated is None else max(0.,now-float(updated))
    source_runs=[value for value in (status.get('run'),latest.get('run'),version.get('run')) if value]
    source_run=source_runs[0] if source_runs and len(set(source_runs))==1 else None
    matching=bool(source_run and source_run==selected and len(source_runs)==3)
    coherent=bool(matching and version.get('revision') is not None and not version.get('writing',False))
    live=bool(selected==configured and matching and coherent and age is not None and age<=LIVE_FRESH_SECONDS and status.get('state') in LIVE_STATES)
    archived=_known_run(db,selected)
    if live:
        freshness='live';reason='Instantánea vigente de la campaña activa.'
    elif selected!=configured:
        freshness='histórico';reason='Consulta de registros persistentes; no hay actividad neuronal reconstruida.'
    elif source_runs:
        freshness='obsoleto';reason='La instantánea viva pertenece a otra campaña, está detenida o superó 10 segundos.'
    elif archived:
        freshness='obsoleto';reason='La campaña seleccionada tiene registros, pero no una instantánea viva vigente.'
    else:
        freshness='sin datos';reason='La campaña está lista para iniciar y aún no tiene registros.'
    return {'selected_run':selected,'configured_run':configured,'freshness':freshness,'source_run':source_run,
      'snapshot_updated':updated,'age_seconds':age,'coherent':coherent,'live':live,'archived':archived,
      'control_allowed':selected==configured,'reason':reason}

def campaign_audit_path(run):
    return RUNTIME/'campaigns'/run/'gain_audit.jsonl'

def live_counts(run,context=None):
    """Return counts only when live artifacts are coherent and belong to `run`."""
    if context is None:
        db=connect()
        try:context=run_context(db,run)
        finally:db.close()
    version=read(RUNTIME/'live_version.json')
    if not context.get('live') or version.get('run')!=run or not (RUNTIME/'live_neurons.npy').exists():
        return None,version
    before=version;counts=np.load(RUNTIME/'live_neurons.npy',mmap_mode='r')[2].copy();after=read(RUNTIME/'live_version.json')
    if before.get('revision')!=after.get('revision') or after.get('run')!=run:return None,after
    return counts,after

@app.get('/api/state')
def state(run:str=''):
    db=connect()
    try:
        context=run_context(db,run);selected=context['selected_run']
        raw_status=read(RUNTIME/'status.json');raw_latest=read(RUNTIME/'latest.json')
        if context['live']:
            status=raw_status;latest=raw_latest
        elif selected!=context['configured_run']:
            status={'state':'historical','reason':'Campaña archivada: consulta de solo lectura.','run':selected}
            latest=_stored_latest(db,selected)
        else:
            status={'state':'stopped','reason':f"La campaña {selected} está lista para iniciar",'run':selected}
            latest=_stored_latest(db,selected)
        rows=db.execute('SELECT payload FROM episodes WHERE run=? ORDER BY id DESC LIMIT 300',(selected,)).fetchall()
        groups=db.execute('SELECT condition,split,count(*),sum(success),sum(native_victory),avg(seconds),avg(reward) FROM episodes WHERE run=? GROUP BY condition,split',(selected,)).fetchall()
        checkpoints=db.execute("SELECT created,condition,seed,path,metadata FROM checkpoints WHERE json_extract(metadata,'$.run')=? ORDER BY id DESC LIMIT 12",(selected,)).fetchall()
        fields=('decision_required','command.action','forced_wait','decision.wait_probability','decision.entropy','activity.spikes',
                'activity.active_neurons','activity.wall_seconds','reward','episode','stage','recorded_at')
        columns=','.join(f"json_extract(payload,'$.{f}')" for f in fields)
        recent=[dict(zip(('id',)+fields,row)) for row in db.execute(f'SELECT id,{columns} FROM steps WHERE run=? ORDER BY id DESC LIMIT 256',(selected,)).fetchall()]
        decision_id=next((x['id'] for x in recent if x['decision_required']),None);last_decision=None
        if decision_id is not None:
            row=db.execute('SELECT payload FROM steps WHERE id=?',(decision_id,)).fetchone()
            if row:
                last_decision=json.loads(row[0]);last_decision['record_id']=decision_id;decision=last_decision.get('decision') or {}
                if 'mask' in decision and 'action_labels' not in decision:
                    from learning_adapter import ActionCatalog
                    decision['action_labels']=[ActionCatalog().label(i) for i in range(len(decision['mask']))]
    finally:
        db.close()
    decisions=[x for x in recent if x['decision_required']];free_waits=sum(x['command.action']=='wait' for x in decisions)
    forced=sum(bool(x['forced_wait']) for x in recent)
    wait_probs=[x['decision.wait_probability'] for x in decisions if x['decision.wait_probability'] is not None]
    entropies=[x['decision.entropy'] for x in decisions if x['decision.entropy'] is not None]
    free_rate=free_waits/len(decisions) if decisions else None
    learning_metrics={'window_steps':len(recent),'free_decisions':len(decisions),'free_waits':free_waits,
      'forced_waits':forced,'free_wait_rate':free_rate,
      'median_wait_probability':float(np.median(wait_probs)) if wait_probs else None,
      'mean_action_entropy':float(np.mean(entropies)) if entropies else None,
      'collapse_alarm':'critical' if len(decisions)>=64 and free_rate is not None and free_rate>=.8 and (np.median(wait_probs) if wait_probs else 0)>=.9 else 'warning' if len(decisions)>=64 and free_rate is not None and (free_rate>=.5 or (np.median(wait_probs) if wait_probs else 0)>=.75) else 'insufficient_data' if len(decisions)<64 else 'healthy',
      'meaning':'Rolling diagnostic over free cognitive decisions; forced native progress ticks are separate.'}
    series=[{'id':x['id'],'spikes':x['activity.spikes'],'active':x['activity.active_neurons'],'wall':x['activity.wall_seconds'],
             'reward':x['reward'],'episode':x['episode'],'stage':x['stage'],'decision':bool(x['decision_required']),'recorded_at':x['recorded_at']} for x in reversed(recent)]
    memory=psutil.virtual_memory();process=psutil.Process()
    return {'context':context,'activity_series':series,'last_decision':last_decision,'status':status,'latest':latest,'episodes':[json.loads(r[0]) for r in reversed(rows)],'groups':[dict(zip(['condition','split','n','successes','native_victories','seconds','reward'],r)) for r in groups],
      'checkpoints':[dict(created=r[0],condition=r[1],seed=r[2],path=r[3],metadata=json.loads(r[4])) for r in checkpoints],
      'capabilities':read(ROOT/'outputs/native_capabilities_v2.json'),'config':read(ROOT/'config/learning.json'),'audit':read(ROOT/'data/processed/audit.json'),'live_version':read(RUNTIME/'live_version.json') if context['live'] else {},
      'learning_metrics':learning_metrics,'sensory_transfer':read(ROOT/('outputs/sensory_transfer_v32.json' if str(read(ROOT/'config/learning.json').get('semantic_protocol_version'))=='3.2' else 'outputs/sensory_transfer_v3.json')),
      'runtime_config':read(ROOT/'config/runtime.json'),'runtime':{'available_ram_gb':memory.available/1e9,'ram_used_percent':memory.percent,'process_ram_gb':process.memory_info().rss/1e9,'cpu_percent':process.cpu_percent(interval=None)}}

@app.get('/api/campaigns')
def campaigns():
    db=connect();configured=configured_run()
    try:
        rows=db.execute("SELECT run,count(*),max(id),max(json_extract(payload,'$.recorded_at')) FROM steps GROUP BY run").fetchall()
        episodes=dict(db.execute('SELECT run,count(*) FROM episodes GROUP BY run').fetchall())
    finally:
        db.close()
    known={row[0]:{'run':row[0],'steps':int(row[1]),'last_step_id':row[2],'updated':row[3],'last_activity':row[3],'episodes':int(episodes.get(row[0],0))} for row in rows}
    for run,count in episodes.items():
        known.setdefault(run,{'run':run,'steps':0,'last_step_id':None,'updated':None,'last_activity':None,'episodes':int(count)})
    if configured not in known:known[configured]={'run':configured,'steps':0,'last_step_id':None,'updated':None,'last_activity':None,'episodes':0}
    status=read(RUNTIME/'status.json')
    for item in known.values():
        item['configured']=item['run']==configured
        item['runtime_state']=status.get('state') if status.get('run')==item['run'] else None
    return {'configured_run':configured,'campaigns':sorted(known.values(),key=lambda item:(not item['configured'],-float(item['updated'] or 0),item['run']))}

@app.get('/api/observatory')
def observatory(run:str=''):
    """Return one coherent supervisory snapshot for the selected campaign."""
    snapshot=state(run);context=snapshot['context'];now=time.time();latest=snapshot.get('latest') or {}
    latest_at=latest.get('recorded_at');runtime=snapshot['runtime'];status=snapshot.get('status') or {}
    return {
        'version':1,'context':context,'server_time':now,
        'snapshot':{'revision':snapshot.get('live_version',{}).get('revision'),'updated':context.get('snapshot_updated'),'age_seconds':context.get('age_seconds'),'coherent':context.get('coherent',False)},
        'latest':{'record_id':latest.get('record_id'),'recorded_at':latest_at,'age_seconds':None if latest_at is None else max(0.,now-latest_at),'episode':latest.get('episode'),'sequence':latest.get('sequence')},
        'runtime':{'state':status.get('state','stopped'),'reason':status.get('reason',''),'pid':status.get('pid'),**runtime},
        'learning':snapshot.get('learning_metrics',{}),
        'activity':{'latest':(snapshot.get('activity_series') or [])[-1] if snapshot.get('activity_series') else None,'series_count':len(snapshot.get('activity_series') or [])},
        'decision':snapshot.get('last_decision'),
        'condition':status.get('condition') or latest.get('condition'),'seed':status.get('seed') or latest.get('seed'),'stage':status.get('stage') or latest.get('stage'),
    }

@app.get('/api/plasticity')
def plasticity(limit:int=120,run:str=''):
    """Return only campaign-scoped gain audits; legacy global traces are never reused."""
    limit=min(300,max(1,limit));db=connect()
    try:context=run_context(db,run)
    finally:db.close()
    path=campaign_audit_path(context['selected_run']);history=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines()[-limit:]:
            try:
                row=json.loads(line)
                if row.get('run')==context['selected_run']:history.append(row)
            except json.JSONDecodeError:continue
    latest=history[-1] if history else None;alerts=[]
    if latest and latest.get('above_ceiling',0)>.01:alerts.append('Más del 1% de gains supera el techo de 10')
    if latest and latest.get('below_floor',0)>.05:alerts.append('Más del 5% de gains está cerca del suelo')
    if len(history)>=2 and latest and latest.get('std',0)>history[-2].get('std',0)*1.5:alerts.append('La dispersión de gains aumentó rápidamente')
    traces=read(RUNTIME/'live_edge_traces.json',[]) if context['live'] else []
    traces=[trace for trace in traces if trace.get('run')==context['selected_run']]
    return {'version':1,'context':context,'archive_available':path.exists(),'revision':read(RUNTIME/'live_version.json').get('revision') if context['live'] else None,
            'history':history,'latest':latest,'alerts':alerts,'traces':traces}

@app.get('/api/history')
def history(before:int=0,limit:int=60,stage:str='',condition:str='',run:str=''):
    db=connect()
    try:
        context=run_context(db,run);result=lab_history.page(db,context['selected_run'],before,limit,stage,condition);result['context']=context;return result
    finally:db.close()

@app.get('/api/history/{ident}')
def history_detail(ident:int,run:str=''):
    db=connect()
    try:
        context=run_context(db,run);result=lab_history.detail(db,context['selected_run'],ident)
    finally:db.close()
    if result is None:raise HTTPException(404,'Step outside this experiment')
    result['context']=context
    return result

@app.get('/api/sensory-schema')
def sensory_schema():
    cfg=read(ROOT/'config/learning.json')
    return read(ROOT/cfg.get('sensory_schema','data/learning/sensory_ports_v31.json'))

@app.post('/api/control/{action}')
async def control(action:str,request:Request):
    origin=request.headers.get('origin');host=request.headers.get('host')
    if origin and origin not in ('http://127.0.0.1:8766','http://localhost:8766'):raise HTTPException(403,'Local origin required')
    if host not in ('127.0.0.1:8766','localhost:8766','testserver'):raise HTTPException(403,'Local host required')
    if action not in ('pause','resume','stop','probe'):raise HTTPException(400,'Unknown command')
    db=connect();cursor=db.execute('INSERT INTO controls(created,command) VALUES(?,?)',(time.time(),json.dumps({'action':action})));db.commit();ident=cursor.lastrowid;db.close()
    return {'queued':True,'id':ident,'action':action,'meaning':'Acknowledged by worker at its next safe step boundary'}

@lru_cache(maxsize=1)
def graph():
    folder=ROOT/'data/processed';cols=['bodyId','type','instance','superclass','somaNeuromere','status','somaSide','somaLocation']
    annotations=feather.read_table(folder/'neurons.feather',columns=cols).to_pandas()
    return annotations,*[np.load(folder/(name+'.npy'),mmap_mode='r') for name in ['body_ids','indptr','indices','counts','transmitter_sign']]

@lru_cache(maxsize=1)
def transmitters():
    import pyarrow as pa,pyarrow.compute as pc
    raw=feather.read_table(ROOT/'data/raw/body-neurotransmitters-male-cns-v1.0.feather')
    raw=raw.filter(pc.is_in(raw['body'],value_set=pa.array(graph()[1])))
    table=raw.to_pandas()
    return table.set_index('body') if 'body' in table.columns else table.set_index('bodyId')

def touch_heavy_cache(now=None):
    global _heavy_cache_last_access
    _heavy_cache_last_access=time.monotonic() if now is None else float(now)

def release_heavy_cache_if_idle(now=None,idle_seconds=HEAVY_CACHE_IDLE_SECONDS,clearers=None):
    """Release graph exploration caches after the dashboard becomes idle."""
    global _heavy_cache_last_access
    current=time.monotonic() if now is None else float(now)
    if not _heavy_cache_last_access or current-_heavy_cache_last_access<idle_seconds:return False
    custom=clearers is not None
    for clear in (clearers if custom else (transmitters.cache_clear,graph.cache_clear,family_codes.cache_clear,incoming.cache_clear)):clear()
    _heavy_cache_last_access=0.;gc.collect()
    if not custom and os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);process=kernel.GetCurrentProcess()
        kernel.SetProcessWorkingSetSize(process,ctypes.c_size_t(-1),ctypes.c_size_t(-1))
    return True

def _cache_janitor():
    while True:
        time.sleep(30);release_heavy_cache_if_idle()

threading.Thread(target=_cache_janitor,name='fly-operator-cache-janitor',daemon=True).start()

@app.get('/api/neurons')
def search_neurons(q:str='',limit:int=25):
    touch_heavy_cache()
    annotations,*_=graph();limit=min(100,max(1,limit));q=q.strip()
    match=np.ones(len(annotations),bool)
    if q:
        match=annotations.bodyId.astype(str).str.contains(q,regex=False)
        for col in ('type','instance','superclass','somaNeuromere'):match |= annotations[col].fillna('').str.contains(q,case=False,regex=False)
    return json.loads(annotations[match].head(limit).to_json(orient='records'))

@app.get('/api/neuron/{body_id}')
def neuron(body_id:int,direction:str='out',limit:int=40,offset:int=0,run:str=''):
    db=connect()
    try:context=run_context(db,run)
    finally:db.close()
    touch_heavy_cache();df,ids,indptr,indices,counts,signs=graph();index=int(np.searchsorted(ids,body_id))
    if index>=len(ids) or ids[index]!=body_id:raise HTTPException(404,'Neuron outside classified inventory')
    if direction not in ('in','out'):raise HTTPException(400,'Use in or out')
    edges=np.arange(indptr[index],indptr[index+1],dtype=np.int64) if direction=='out' else np.flatnonzero(indices==index)
    order=np.argsort(counts[edges],kind='stable')[::-1];edges=edges[order];total=len(edges);edges=edges[max(0,offset):max(0,offset)+min(100,max(1,limit))]
    pres=np.full(len(edges),index) if direction=='out' else np.searchsorted(indptr,edges,side='right')-1;posts=indices[edges]
    live,version=live_counts(context['selected_run'],context);gains=None
    if live is not None and (RUNTIME/'live_gains.npy').exists():
        gains=np.load(RUNTIME/'live_gains.npy',mmap_mode='r')[edges].copy()
        after=read(RUNTIME/'live_version.json')
        if after.get('revision')!=version.get('revision') or after.get('run')!=context['selected_run']:gains=None
    nt=transmitters();rows=[]
    for k,e in enumerate(edges):
        pre=int(pres[k]);post=int(posts[k]);pred=nt.loc[int(ids[pre])].to_dict() if int(ids[pre]) in nt.index else {}
        pred=json.loads(json.dumps(pred,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
        pred={key:(None if isinstance(value,float) and not math.isfinite(value) else value) for key,value in pred.items()}
        gain=float(gains[k]) if gains is not None else None
        rows.append({'edge_index':int(e),'pre':int(ids[pre]),'post':int(ids[post]),'pre_type':df.iloc[pre]['type'],'post_type':df.iloc[post]['type'],
          'anatomical_count':int(counts[e]),'simulated_sign':float(signs[pre]),'initial_weight':float(counts[e])*.275*float(signs[pre]),'gain':gain,
          'effective_weight':None if gain is None else float(counts[e])*.275*float(signs[pre])*gain,'transmitter_prediction':pred,
          'source':'male-cns:v1.0 / connectome-weights-male-cns-v1.0-minconf-0.5.feather','classification':'count: published; sign and unit weight: assumed; gain: learned'})
    snapshot=None
    if live is not None:snapshot={'membrane_mv':float(np.load(RUNTIME/'live_neurons.npy',mmap_mode='r')[0,index]),'synaptic_state':float(np.load(RUNTIME/'live_neurons.npy',mmap_mode='r')[1,index]),'spikes_last_50ms':int(live[index]),**version}
    if context['live']:
        atomic_json(RUNTIME/'watch_edges.json',edges.astype(int).tolist());atomic_json(RUNTIME/'watch_neurons.json',[index])
    traces={r['edge_index']:r for r in (read(RUNTIME/'live_edge_traces.json',[]) if context['live'] else []) if r.get('run')==context['selected_run']}
    for row in rows:row['plasticity_trace']=traces.get(row['edge_index'])
    history=[];path=RUNTIME/'neuron_trace.jsonl'
    if path.exists():
        for line in path.read_text().splitlines()[-2000:]:
            try:item=json.loads(line)
            except json.JSONDecodeError:continue
            if item.get('run')==context['selected_run'] and item['body_id']==body_id:history.append(item)
    if history:history=[h for h in history if (h['epoch'],h['condition'],h['seed'])==(history[-1]['epoch'],history[-1]['condition'],history[-1]['seed'])][-120:]
    return {'context':context,'neuron':json.loads(df.iloc[[index]].to_json(orient='records'))[0],'direction':direction,'total':total,'connections':rows,'live':snapshot,'history':history,'snapshot_coherent':gains is not None}

# --- Circuit views: anatomy aggregated into eight readable families plus live activity ---------
FAMILIES=[('visual_projection','Proyección visual'),('sensory','Sensorial'),('optic','Lóbulo óptico'),('central','Cerebro central'),
          ('vnc','Cordón nervioso'),('ascending','Ascendentes'),('descending','Descendentes'),('other','Motor y otras')]
FAMILY_INDEX={key:i for i,(key,_) in enumerate(FAMILIES)}
INPUT_GROUPS=[('goal','Objetivo'),('enemy','Enemigos'),('people','Personas'),('door','Puertas'),('wall','Paredes'),('self','Estado propio'),('task','Tarea y tiempo')]

def family_key(superclass):
    s=str(superclass or '')
    if s.startswith('visual_projection'):return 'visual_projection'
    if s.startswith('descending_neuron'):return 'descending'
    if s.startswith('ascending_neuron'):return 'ascending'
    if 'sensory' in s:return 'sensory'
    if s.startswith('ol_') or s.startswith('visual_centrifugal'):return 'optic'
    if s=='cb_intrinsic':return 'central'
    if s in ('vnc_intrinsic','vnc_tbc'):return 'vnc'
    return 'other'

def text(value):
    return value if isinstance(value,str) else None

def goal_bearing_deg(channels):
    """Goal direction relative to the operator heading, recovered from the rescaled sin/cos channels."""
    if 'goal_bearing_sin' not in channels or 'goal_bearing_cos' not in channels:return None
    return math.degrees(math.atan2(2*channels['goal_bearing_sin']-1,2*channels['goal_bearing_cos']-1))

def input_group(channel):
    if channel.startswith(('goal_','orientation_')):return 'goal'
    if channel.startswith('enemy_'):return 'enemy'
    if channel.startswith(('friend_','hostage_','civilian_')):return 'people'
    if channel.startswith('door_'):return 'door'
    if channel.startswith('wall_'):return 'wall'
    if channel.startswith('task_') or channel in ('tonic','time_remaining'):return 'task'
    return 'self'

@lru_cache(maxsize=1)
def family_codes():
    df=graph()[0];return np.array([FAMILY_INDEX[family_key(s)] for s in df.superclass],np.int8)

@lru_cache(maxsize=1)
def incoming():
    folder=ROOT/'data/processed'
    return [np.load(folder/(name+'.npy'),mmap_mode='r') for name in ('incoming_indptr_v2','incoming_edges_v2','edge_pre_v2')]

@lru_cache(maxsize=1)
def circuit_structure():
    """Family-level synapse totals over every classified edge, cached on disk (anatomy is immutable)."""
    df,ids,indptr,indices,counts,signs=graph();codes=family_codes();F=len(FAMILIES)
    cache=ROOT/'data/processed/family_connectivity_v1.json'
    signature={'neurons':int(len(ids)),'edges':int(len(indices)),'families':FAMILIES}
    cached=read(cache)
    if cached.get('signature')==json.loads(json.dumps(signature)):return cached
    exc=np.zeros(F*F);inh=np.zeros(F*F);pairs=np.zeros(F*F,np.int64);degree=np.diff(np.asarray(indptr))
    for start in range(0,len(ids),8000):
        end=min(len(ids),start+8000);e0,e1=int(indptr[start]),int(indptr[end])
        pre=np.repeat(codes[start:end].astype(np.int16),degree[start:end]);post=codes[np.asarray(indices[e0:e1])].astype(np.int16)
        key=pre*F+post;synapses=np.asarray(counts[e0:e1],np.float64);excitatory=np.repeat(np.asarray(signs[start:end])>0,degree[start:end])
        exc+=np.bincount(key[excitatory],weights=synapses[excitatory],minlength=F*F);inh+=np.bincount(key[~excitatory],weights=synapses[~excitatory],minlength=F*F)
        pairs+=np.bincount(key,minlength=F*F)
    families=[{'key':key,'label':label,'neurons':int(np.sum(codes==i)),
               'superclasses':sorted({str(s) for s,c in zip(df.superclass,codes) if c==i})} for i,(key,label) in enumerate(FAMILIES)]
    result={'signature':signature,'families':families,'synapses_excitatory':exc.reshape(F,F).astype(int).tolist(),
            'synapses_inhibitory':inh.reshape(F,F).astype(int).tolist(),'connections':pairs.reshape(F,F).tolist(),
            'meaning':'Rows are presynaptic families, columns postsynaptic. Sign follows the predicted transmitter of the presynaptic neuron (model assumption).'}
    atomic_json(cache,result);return result

def _latest_for_context(context):
    db=connect()
    try:
        if context['live']:
            latest=read(RUNTIME/'latest.json')
            if latest.get('run')==context['selected_run']:
                decision=latest.get('decision') or {}
                if 'mask' in decision and 'action_labels' not in decision:
                    from learning_adapter import ActionCatalog
                    decision['action_labels']=[ActionCatalog().label(i) for i in range(len(decision['mask']))]
                return latest
        return _stored_latest(db,context['selected_run'])
    finally:
        db.close()

@app.get('/api/circuit/structure')
def circuit_structure_view():
    touch_heavy_cache()
    structure=circuit_structure();cfg=read(ROOT/'config/learning.json')
    schema=read(ROOT/cfg.get('sensory_schema','data/learning/sensory_ports_v31.json'));codes=family_codes()
    gateways={key:{} for key,_ in INPUT_GROUPS}
    for channel,item in schema.get('mapping',{}).items():
        group=gateways[input_group(channel)]
        for index in item.get('indices',[]):
            family=FAMILIES[int(codes[index])][0];group[family]=group.get(family,0)+1
    return {**{k:v for k,v in structure.items() if k!='signature'},'input_groups':[{'key':k,'label':l} for k,l in INPUT_GROUPS],'gateways':gateways}

@app.get('/api/brain/3d')
def brain_3d(limit:int=1400,family:str='',hemisphere:str='',region:str='',neuron_type:str='',active_only:bool=False,decision_only:bool=False,run:str=''):
    """Return a bounded point cloud, never attaching another campaign's live activity."""
    db=connect()
    try:context=run_context(db,run)
    finally:db.close()
    touch_heavy_cache();df,ids,indptr,indices,counts,signs=graph();codes=family_codes();live,version=live_counts(context['selected_run'],context)
    latest=_latest_for_context(context);decision=latest.get('decision') or {}
    readout={int(row.get('body_id')) for row in (latest.get('neurons_readout') or [])};contribution_ids={int(row.get('body_id')) for row in (decision.get('contributions') or [])}
    target_ids=readout|contribution_ids
    if not context['live']:active_only=False
    allowed={FAMILY_INDEX[family]} if family in FAMILY_INDEX else set(range(len(FAMILIES)))
    def matches(index):
        row=df.iloc[int(index)];side=str(row.get('somaSide') or '').lower();neuromere=str(row.get('somaNeuromere') or '').lower();kind=str(row.get('type') or '').lower();body=int(ids[index])
        hemi_ok=not hemisphere or (hemisphere.lower() in ('left','l') and side=='l') or (hemisphere.lower() in ('right','r') and side=='r') or (hemisphere.lower() in ('middle','m') and side=='m')
        return hemi_ok and (not region or region.lower() in neuromere) and (not neuron_type or neuron_type.lower() in kind) and (not active_only or (live is not None and int(live[index])>0)) and (not decision_only or body in target_ids)
    limit=min(2400,max(200,limit));selected=[]
    for family_index in sorted(allowed):
        candidates=[index for index in np.flatnonzero(codes==family_index) if matches(index)]
        quota=max(1,round(limit*len(candidates)/max(1,sum(np.sum(codes==i) for i in allowed))))
        selected.extend(np.asarray(candidates)[np.linspace(0,len(candidates)-1,min(quota,len(candidates)),dtype=int)].tolist() if candidates else [])
    selected=np.asarray(sorted(set(selected)),dtype=np.int64);node_index={};nodes=[]
    for index in selected:
        raw=df.iloc[int(index)].get('somaLocation');coord=list(raw) if isinstance(raw,(list,tuple,np.ndarray)) else []
        if len(coord)!=3:continue
        family_key=FAMILIES[int(codes[index])][0];body_id=int(ids[index]);node_index[int(index)]=len(nodes)
        nodes.append({'body_id':body_id,'x':int(coord[0]),'y':int(coord[1]),'z':int(coord[2]),'family':family_key,'type':str(df.iloc[int(index)]['type'] or ''),'region':str(df.iloc[int(index)].get('somaNeuromere') or ''),'hemisphere':str(df.iloc[int(index)].get('somaSide') or ''),'spikes':None if live is None else int(live[index]),'decision_relevant':body_id in target_ids})
    traces={int(row.get('edge_index')):row for row in (read(RUNTIME/'live_edge_traces.json',[]) if context['live'] else []) if row.get('run')==context['selected_run']};edges=[]
    for index in selected:
        if int(index) not in node_index:continue
        start,end=int(indptr[index]),int(indptr[index+1]);candidate_edges=[int(edge) for edge in range(start,end) if int(indices[edge]) in node_index]
        candidate_edges=sorted(candidate_edges,key=lambda edge:int(counts[edge]),reverse=True)[:4]
        for edge in candidate_edges:
            target_index=int(indices[edge]);trace=traces.get(edge,{})
            edges.append({'source':node_index[int(index)],'target':node_index[target_index],'edge_index':edge,'synapses':int(counts[edge]),'sign':float(signs[index]),'gain':trace.get('gain'),'effective_weight':None if trace.get('gain') is None else float(counts[edge])*.275*float(signs[index])*float(trace['gain'])})
    return {'version':1,'context':context,'coordinate_system':'connectome somaLocation voxels','total_neurons':int(len(ids)),'total_edges':int(len(indices)),'sampled_neurons':len(nodes),'sampled_edges':len(edges),'nodes':nodes,'edges':edges,'families':[{'key':key,'label':label} for key,label in FAMILIES],'activity_revision':version.get('revision') if context['live'] else None,'activity_available':bool(context['live'] and live is not None),'filters':{'family':family,'hemisphere':hemisphere,'region':region,'neuron_type':neuron_type,'active_only':active_only,'decision_only':decision_only},'model_status':'Observed anatomy. Activity appears only when the selected campaign owns a coherent live snapshot; decision relevance is an actor-logit inference.'}

@app.get('/api/causal/current')
def causal_current(run:str=''):
    db=connect()
    try:context=run_context(db,run)
    finally:db.close()
    latest=_latest_for_context(context);decision=latest.get('decision') or {}
    contributions=sorted(decision.get('contributions') or [],key=lambda row:abs(float(row.get('logit_contribution',0))),reverse=True)[:20]
    active_readout=latest.get('neurons_readout',[])[:24] if context['live'] else []
    return {'version':1,'context':context,'observed':{'condition':latest.get('condition'),'seed':latest.get('seed'),'stage':latest.get('stage'),'episode':latest.get('episode'),'sequence':latest.get('sequence'),'channels':latest.get('pre_action',{}).get('channels',{}),'active_readout':active_readout},'inferred':{'controller':decision.get('controller'),'action':latest.get('action_label'),'selected':decision.get('selected'),'contributions':contributions,'meaning':'Contribution ranking is an inference from actor logits, not a causal proof.'}}

@app.get('/api/export/brain3d')
def export_brain3d(limit:int=1400,family:str='',hemisphere:str='',region:str='',neuron_type:str='',active_only:bool=False,decision_only:bool=False,format:str='json',run:str=''):
    payload=brain_3d(limit,family,hemisphere,region,neuron_type,active_only,decision_only,run)
    if format=='csv':
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['body_id','type','family','region','hemisphere','x','y','z','spikes','decision_relevant']);writer.writeheader();writer.writerows(payload['nodes'])
        return Response(stream.getvalue().encode('utf-8-sig'),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=brain3d-selection.csv'})
    return Response(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8'),media_type='application/json',headers={'Content-Disposition':'attachment; filename=brain3d-selection.json'})

@app.get('/api/circuit/live')
def circuit_live(run:str=''):
    db=connect()
    try:context=run_context(db,run)
    finally:db.close()
    latest=_latest_for_context(context);decision=latest.get('decision') or {}
    base={'context':context,'coherent':False,'families':None,'input_drive_hz':{},'revision':None,
          'condition':latest.get('condition'),'seed':latest.get('seed'),'record_id':latest.get('record_id'),'sequence':latest.get('sequence'),
          'decision':{k:decision.get(k) for k in ('probabilities','mask','selected','executed','teacher_action','controller','entropy')},
          'action_labels':decision.get('action_labels'),'action_label':latest.get('action_label'),'stage':latest.get('stage'),
          'meaning':'Historical records retain decisions, but not a reconstructable full-neuron activity window.'}
    counts,version=live_counts(context['selected_run'],context)
    if counts is None:return base
    codes=family_codes();F=len(FAMILIES);cfg=read(ROOT/'config/learning.json')
    schema=read(ROOT/cfg.get('sensory_schema','data/learning/sensory_ports_v31.json'));gains=(schema.get('encoding') or {}).get('channel_gain_hz',{});drive={key:0. for key,_ in INPUT_GROUPS}
    for channel,value in (latest.get('channels') or {}).items():
        ports=len(schema.get('mapping',{}).get(channel,{}).get('indices',[])) or 16
        drive[input_group(channel)]+=float(gains.get(channel,schema.get('gain_hz',145.)))*float(value)*ports
    spikes=np.bincount(codes,weights=counts,minlength=F);active=np.bincount(codes,weights=counts>0,minlength=F);neurons=np.bincount(codes,minlength=F)
    return {**base,'coherent':True,'families':[{'key':key,'spikes':int(spikes[i]),'active':int(active[i]),'neurons':int(neurons[i])} for i,(key,_) in enumerate(FAMILIES)],'input_drive_hz':drive,'revision':version.get('revision'),'meaning':'Input drive is the current selected-run stimulus rate above baseline. Family activity is the current 50 ms window.'}

@app.get('/api/neuron/{body_id}/graph')
def neuron_graph(body_id:int,limit:int=10,run:str=''):
    db=connect()
    try:context=run_context(db,run)
    finally:db.close()
    touch_heavy_cache();df,ids,indptr,indices,counts,signs=graph();codes=family_codes();inptr,inedges,edgepre=incoming()
    index=int(np.searchsorted(ids,body_id))
    if index>=len(ids) or ids[index]!=body_id:raise HTTPException(404,'Neuron outside classified inventory')
    live,_=live_counts(context['selected_run'],context);limit=min(24,max(1,limit))
    def partners(edges,others,sign_of):
        edges=np.asarray(edges,np.int64);others=np.asarray(others,np.int64);synapses=np.asarray(counts[edges],np.int64)
        order=np.argsort(synapses,kind='stable')[::-1][:limit]
        by_family=np.bincount(codes[others],weights=synapses,minlength=len(FAMILIES)) if len(others) else np.zeros(len(FAMILIES))
        items=[{'body_id':int(ids[others[k]]),'type':text(df.iloc[int(others[k])]['type']),'family':FAMILIES[int(codes[others[k]])][0],
                'synapses':int(synapses[k]),'sign':float(sign_of(others[k])),'spikes':None if live is None else int(live[others[k]])} for k in order]
        return items,{'connections':int(len(edges)),'synapses':int(synapses.sum()),
                      'by_family':{FAMILIES[i][0]:int(v) for i,v in enumerate(by_family) if v}}
    out_edges=np.arange(indptr[index],indptr[index+1],dtype=np.int64);outputs,out_total=partners(out_edges,np.asarray(indices[out_edges]),lambda other:signs[index])
    in_edges=np.asarray(inedges[inptr[index]:inptr[index+1]],np.int64);pre=np.asarray(edgepre[in_edges],np.int64);inputs,in_total=partners(in_edges,pre,lambda other:signs[other])
    if len(in_edges):
        synapses=np.asarray(counts[in_edges],np.int64);negative=np.asarray(signs[pre])<0
        in_total['excitatory_synapses']=int(synapses[~negative].sum());in_total['inhibitory_synapses']=int(synapses[negative].sum())
    row=df.iloc[index]
    return {'context':context,'neuron':{'body_id':int(body_id),'type':text(row['type']),'instance':text(row['instance']),'superclass':text(row['superclass']),
                      'family':FAMILIES[int(codes[index])][0],'sign':float(signs[index]),'spikes':None if live is None else int(live[index])},
            'inputs':inputs,'outputs':outputs,'input_totals':in_total,'output_totals':out_total,
            'families':[{'key':k,'label':l} for k,l in FAMILIES]}

@app.get('/api/parameters')
def parameters():
    cfg=read(ROOT/'config/learning.json');rows=[]
    for key,value in asdict(Dynamics()).items():
        unit='ms' if key.endswith('_ms') else 'mV' if key.endswith('_mv') else 'dimensionless'
        rows.append({'name':'dynamics.'+key,'value':value,'unit':unit,'classification':'assumed','source':'config / Shiu-inspired LIF plus engineered R-STDP','version':cfg.get('version',3)})
    def flatten(obj,prefix='config'):
        for k,v in obj.items():
            name=prefix+'.'+k
            if isinstance(v,dict):flatten(v,name)
            else:rows.append({'name':name,'value':v,'unit':'ms' if k.endswith('_ms') else 'count' if k in ('classified_neurons','directed_edges') else 'see schema','classification':'published' if k in ('dataset','classified_neurons','directed_edges') else 'assumed','source':'config/learning.json','version':cfg.get('version',3)})
    flatten(cfg);return rows

@app.get('/api/equipment')
def equipment(q:str='',limit:int=30,run:str=''):
    db=connect()
    try:context=run_context(db,run)
    finally:db.close()
    catalog=read(ROOT/'data/learning/equipment_catalog.json');items=[v for k,v in catalog['items'].items() if q.lower() in k.lower() or q.lower() in v['family'].lower()]
    latest=_latest_for_context(context)
    return {'context':context,'total':len(items),'items':items[:min(100,max(1,limit))],'runtime':latest.get('observation',{}).get('inventory',[]),'bindings':catalog['bindings'] if q else []}

@app.get('/api/export/{kind}')
def export(kind:str,run:str=''):
    if kind=='video':
        path=ROOT/'outputs/FlyOperator_investigacion_v2_1080p30.mp4'
        if not path.exists():raise HTTPException(404,'No evaluation clip has been captured')
        return FileResponse(path,media_type='video/mp4',filename=path.name)
    if kind not in ('csv','svg','png','bundle'):raise HTTPException(400,'Unknown export type')
    db=connect()
    try:
        context=run_context(db,run);selected=context['selected_run'];rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM episodes WHERE run=? ORDER BY id',(selected,))]
    finally:db.close()
    if kind=='csv':
        f=io.StringIO();writer=csv.DictWriter(f,fieldnames=['condition','seed','stage','split','success','native_victory','reward','steps','simulated_ms','wall_seconds','operator_health'],extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        return Response(f.getvalue().encode('utf-8-sig'),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=episodes.csv'})
    if kind=='bundle':
        f=io.BytesIO()
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
            cfg=read(ROOT/'config/learning.json');campaign_dir=RUNTIME/'campaigns'/selected
            schedule=campaign_dir/'schedule.json'
            sensory=ROOT/cfg.get('sensory_schema','data/learning/sensory_ports_v31.json')
            for p in [ROOT/'config/learning.json',ROOT/'config/runtime.json',ROOT/'data/processed/audit.json',sensory,ROOT/'outputs/native_capabilities_v2.json',schedule,campaign_audit_path(selected)]:
                if p.exists():z.write(p,str(p.relative_to(ROOT)))
            if context['live']:
                for p in [RUNTIME/'status.json',RUNTIME/'latest.json',RUNTIME/'live_version.json']:
                    if p.exists():z.write(p,str(p.relative_to(ROOT)))
            z.writestr('context.json',json.dumps(context));z.writestr('episodes.json',json.dumps(rows));z.writestr('parameters.json',json.dumps(parameters()))
        return Response(f.getvalue(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename=FlyOperator-evidence.zip'})
    import matplotlib;matplotlib.use('Agg')
    from matplotlib.figure import Figure
    fig=Figure(figsize=(12.8,7.2),dpi=150);ax=fig.subplots()
    for condition in ['common_teaching','frozen','adapter','internal','combined']:
        values=[r for r in rows if r['condition']==condition]
        if values:ax.plot(np.arange(1,len(values)+1),[r['reward'] for r in values],label=condition)
    ax.set(xlabel='Episodio (por condición)',ylabel='Recompensa',title='Fly Operator · Datos observados; enseñanza y evaluación identificadas')
    if rows:ax.legend()
    else:ax.text(.5,.5,'Aún no hay episodios completos',ha='center',transform=ax.transAxes)
    ax.grid(alpha=.15);f=io.BytesIO();fig.savefig(f,format=kind)
    return Response(f.getvalue(),media_type='image/svg+xml' if kind=='svg' else 'image/png',headers={'Content-Disposition':f'attachment; filename=learning.{kind}'})

dist=ROOT/'dashboard/dist'
if dist.exists():app.mount('/',StaticFiles(directory=dist,html=True),name='dashboard')
