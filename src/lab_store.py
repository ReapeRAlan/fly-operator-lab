"""Durable single-writer experiment records and local dashboard control queue."""
from pathlib import Path
import json,sqlite3,time,os,hashlib,tempfile
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'work/learning';RUNTIME.mkdir(parents=True,exist_ok=True)

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,ensure_ascii=False,allow_nan=False)
    # Windows readers may briefly deny replacement while reading the old snapshot.
    # Unique temporaries also prevent simultaneous API requests sharing a .tmp file.
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        temporary=Path(f.name);f.write(payload);f.flush()
    deadline=time.monotonic()+3.;delay=.005
    try:
        while True:
            try:os.replace(temporary,path);break
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(delay);delay=min(.1,delay*2)
    finally:
        if temporary.exists():temporary.unlink()

def durable_row(info):
    """Static action-catalog labels are served by lab_history, not repeated per stored step."""
    decision=info.get('decision')
    if not isinstance(decision,dict) or 'action_labels' not in decision:return info
    row=dict(info);row['decision']={k:v for k,v in decision.items() if k!='action_labels'};return row

def connect():
    db=sqlite3.connect(RUNTIME/'experiments.sqlite',timeout=15)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY, run TEXT, condition TEXT, seed INTEGER, stage TEXT, split TEXT, success INTEGER, native_victory INTEGER, reward REAL, seconds REAL, payload TEXT);
      CREATE TABLE IF NOT EXISTS steps(id INTEGER PRIMARY KEY, run TEXT, episode INTEGER, sequence INTEGER, payload TEXT);
      CREATE INDEX IF NOT EXISTS steps_run_id ON steps(run,id);
      CREATE INDEX IF NOT EXISTS steps_run_episode_id ON steps(run,episode,id);
      CREATE TABLE IF NOT EXISTS controls(id INTEGER PRIMARY KEY, created REAL, command TEXT, consumed INTEGER DEFAULT 0);
      CREATE TABLE IF NOT EXISTS checkpoints(id INTEGER PRIMARY KEY, created REAL, condition TEXT, seed INTEGER, path TEXT, hash TEXT, metadata TEXT);
    ''');db.commit();return db

class Store:
    def __init__(self,run):self.run=run;self.db=connect();self.latest={};self.status={};self.last_save=0
    def step(self,info):
        info['recorded_at']=time.time()
        info['run']=self.run
        self.latest=info
        cursor=self.db.execute('INSERT INTO steps(run,episode,sequence,payload) VALUES(?,?,?,?)',(self.run,info['episode'],info['sequence'],json.dumps(durable_row(info),allow_nan=False)))
        self.db.commit()
        info['record_id']=cursor.lastrowid
        atomic_json(RUNTIME/'latest.json',info)
    def preview(self,info,transaction_id):
        """Publish live validation telemetry without inserting scientific rows."""
        item=dict(info);item['recorded_at']=time.time();item['run']=self.run
        item['evaluation_transaction_id']=transaction_id;item['evaluation_committed']=False;item['evaluation_preview']=True
        self.latest=item;atomic_json(RUNTIME/'latest.json',item)
    def episode(self,info):
        self.db.execute('INSERT INTO episodes(run,condition,seed,stage,split,success,native_victory,reward,seconds,payload) VALUES(?,?,?,?,?,?,?,?,?,?)',
           (self.run,info['condition'],info['seed'],info['stage'],info['split'],int(info['success']),int(info['native_victory']),info['reward'],info['simulated_ms']/1000,json.dumps(info)))
        self.db.commit()
    def evaluation(self,transaction_id,steps,episodes):
        """Publish a completed validation block in one SQLite transaction."""
        prepared_steps=[]
        for source in steps:
            info=dict(source);info['recorded_at']=time.time();info['run']=self.run
            info['evaluation_transaction_id']=transaction_id;info['evaluation_committed']=True;info['evaluation_preview']=False
            prepared_steps.append(info)
        prepared_episodes=[]
        for source in episodes:
            info=dict(source);info['evaluation_transaction_id']=transaction_id;info['evaluation_committed']=True
            prepared_episodes.append(info)
        # Serialize before BEGIN so malformed/non-finite data cannot leave a
        # partially committed validation block.
        step_rows=[(self.run,x['episode'],x['sequence'],json.dumps(durable_row(x),allow_nan=False)) for x in prepared_steps]
        episode_rows=[(self.run,x['condition'],x['seed'],x['stage'],x['split'],int(x['success']),int(x['native_victory']),x['reward'],x['simulated_ms']/1000,json.dumps(x,allow_nan=False)) for x in prepared_episodes]
        try:
            self.db.execute('BEGIN IMMEDIATE')
            last_step_id=None
            if step_rows:
                self.db.executemany('INSERT INTO steps(run,episode,sequence,payload) VALUES(?,?,?,?)',step_rows)
                last_step_id=self.db.execute('SELECT last_insert_rowid()').fetchone()[0]
            if episode_rows:self.db.executemany('INSERT INTO episodes(run,condition,seed,stage,split,success,native_victory,reward,seconds,payload) VALUES(?,?,?,?,?,?,?,?,?,?)',episode_rows)
            self.db.commit()
        except BaseException:
            self.db.rollback();raise
        if prepared_steps:
            prepared_steps[-1]['record_id']=last_step_id;self.latest=prepared_steps[-1];atomic_json(RUNTIME/'latest.json',self.latest)
    def update(self,**values):
        self.status.update(values);self.status.update(run=self.run,updated=time.time(),pid=os.getpid())
        atomic_json(RUNTIME/'status.json',self.status)
    def controls(self):
        rows=self.db.execute('SELECT id,command FROM controls WHERE consumed=0 ORDER BY id').fetchall()
        if rows:self.db.executemany('UPDATE controls SET consumed=1 WHERE id=?',[(r[0],) for r in rows]);self.db.commit()
        return [json.loads(r[1]) for r in rows]
    def checkpoint(self,condition,seed,path,digest,metadata):
        self.db.execute('INSERT INTO checkpoints(created,condition,seed,path,hash,metadata) VALUES(?,?,?,?,?,?)',(time.time(),condition,seed,str(path),digest,json.dumps(metadata)));self.db.commit()
    def close(self):self.db.close()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
