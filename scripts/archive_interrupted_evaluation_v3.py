"""Archive and remove pre-transaction validation rows from the active v3 run."""
from pathlib import Path
import gzip,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lab_store import RUNTIME,atomic_json,connect,file_hash
from learning_runtime import WorkerLock

RUN='pilot-v3.1-curriculum'
PREDICATE="run=? AND json_extract(payload,'$.split')='validation' AND coalesce(json_extract(payload,'$.evaluation_committed'),0)=0"


def main():
    with WorkerLock():
        schedule_path=RUNTIME/'campaigns'/RUN/'schedule.json';schedule=json.loads(schedule_path.read_text(encoding='utf-8-sig'))
        if schedule.get('bootstrap_evaluations') or schedule.get('stage_index')!=0 or schedule.get('prepared',{}).get('move'):
            raise ValueError('Curriculum has committed bootstrap results; automatic cleanup is no longer authorized')
        db=connect()
        try:
            steps=db.execute(f'SELECT id,payload FROM steps WHERE {PREDICATE} ORDER BY id',(RUN,)).fetchall()
            episodes=db.execute(f'SELECT id,payload FROM episodes WHERE {PREDICATE} ORDER BY id',(RUN,)).fetchall()
            if not steps and not episodes:
                print(json.dumps({'passed':True,'already_clean':True}));return
            archive_data={'version':1,'run':RUN,'reason':'Interrupted evaluation rows created before transactional publication',
              'schedule_sha256':file_hash(schedule_path),'steps':[{'id':i,'payload':json.loads(p)} for i,p in steps],
              'episodes':[{'id':i,'payload':json.loads(p)} for i,p in episodes]}
            raw=json.dumps(archive_data,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            folder=RUNTIME/'archives';folder.mkdir(parents=True,exist_ok=True)
            archive=folder/'interrupted_evaluation_pre_transaction_v3.json.gz'
            with archive.open('wb') as target:
                with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as compressed:compressed.write(raw)
            archive_sha=file_hash(archive)
            db.execute('BEGIN IMMEDIATE')
            current_steps=db.execute(f'SELECT count(*) FROM steps WHERE {PREDICATE}',(RUN,)).fetchone()[0]
            current_episodes=db.execute(f'SELECT count(*) FROM episodes WHERE {PREDICATE}',(RUN,)).fetchone()[0]
            if (current_steps,current_episodes)!=(len(steps),len(episodes)):raise RuntimeError('Evaluation rows changed during archive')
            db.execute(f'DELETE FROM steps WHERE {PREDICATE}',(RUN,));db.execute(f'DELETE FROM episodes WHERE {PREDICATE}',(RUN,));db.commit()
            latest=db.execute('SELECT id,payload FROM steps WHERE run=? ORDER BY id DESC LIMIT 1',(RUN,)).fetchone()
            if latest:
                payload=json.loads(latest[1]);payload['record_id']=latest[0];atomic_json(RUNTIME/'latest.json',payload)
            record={'version':1,'passed':True,'created':time.time(),'run':RUN,'archived_steps':len(steps),'archived_episodes':len(episodes),
              'step_id_range':[steps[0][0],steps[-1][0]] if steps else None,'episode_id_range':[episodes[0][0],episodes[-1][0]] if episodes else None,
              'archive':str(archive),'archive_sha256':archive_sha,'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),
              'schedule_sha256':archive_data['schedule_sha256'],'remaining_uncommitted_validation_steps':db.execute(f'SELECT count(*) FROM steps WHERE {PREDICATE}',(RUN,)).fetchone()[0],
              'remaining_uncommitted_validation_episodes':db.execute(f'SELECT count(*) FROM episodes WHERE {PREDICATE}',(RUN,)).fetchone()[0]}
            atomic_json(ROOT/'outputs/interrupted_evaluation_cleanup_v3.json',record);print(json.dumps(record,indent=2))
        except BaseException:
            db.rollback();raise
        finally:db.close()


if __name__=='__main__':main()
