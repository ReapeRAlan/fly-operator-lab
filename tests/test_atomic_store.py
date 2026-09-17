import sys,json,threading,time
import pytest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import lab_store
from lab_store import atomic_json

def test_windows_reader_does_not_interrupt_publishing(tmp_path):
    path=tmp_path/'state.json';atomic_json(path,{'revision':0})
    reader=path.open('rb');released=threading.Event()
    def release():
        time.sleep(.12);reader.close();released.set()
    thread=threading.Thread(target=release);thread.start()
    atomic_json(path,{'revision':1});thread.join()
    assert released.is_set() and json.loads(path.read_text())=={'revision':1}
    assert not list(tmp_path.glob('*.tmp'))

def test_concurrent_publishers_leave_complete_json(tmp_path):
    path=tmp_path/'watch.json';errors=[]
    def publish(index):
        try:
            for revision in range(12):atomic_json(path,{'writer':index,'revision':revision,'payload':'x'*8192})
        except BaseException as exc:errors.append(exc)
    threads=[threading.Thread(target=publish,args=(i,)) for i in range(4)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    result=json.loads(path.read_text())
    assert not errors and result['revision']==11 and len(result['payload'])==8192
    assert not list(tmp_path.glob('*.tmp'))


def test_evaluation_block_is_committed_and_tagged_atomically(tmp_path,monkeypatch):
    monkeypatch.setattr(lab_store,'RUNTIME',tmp_path);store=lab_store.Store('run')
    steps=[{'episode':9,'sequence':i,'value':i} for i in (1,2)]
    episodes=[{'condition':'common_teaching','seed':7,'stage':'move','split':'validation','success':True,
      'native_victory':False,'reward':1.0,'simulated_ms':100}]
    try:
        store.evaluation('tx-1',steps,episodes)
        step_rows=[json.loads(r[0]) for r in store.db.execute('SELECT payload FROM steps ORDER BY id')]
        episode_rows=[json.loads(r[0]) for r in store.db.execute('SELECT payload FROM episodes ORDER BY id')]
        assert len(step_rows)==2 and len(episode_rows)==1
        assert all(x['evaluation_transaction_id']=='tx-1' and x['evaluation_committed'] is True for x in step_rows+episode_rows)
        latest=json.loads((tmp_path/'latest.json').read_text())
        assert latest['sequence']==2 and latest['record_id']==store.db.execute('SELECT max(id) FROM steps').fetchone()[0]
    finally:store.close()


def test_invalid_evaluation_block_publishes_no_rows(tmp_path,monkeypatch):
    monkeypatch.setattr(lab_store,'RUNTIME',tmp_path);store=lab_store.Store('run')
    try:
        with pytest.raises(ValueError):store.evaluation('tx-bad',[{'episode':1,'sequence':1,'value':float('nan')}],[])
        assert store.db.execute('SELECT count(*) FROM steps').fetchone()[0]==0
        assert store.db.execute('SELECT count(*) FROM episodes').fetchone()[0]==0
    finally:store.close()


def test_evaluation_preview_is_live_but_not_persisted(tmp_path,monkeypatch):
    monkeypatch.setattr(lab_store,'RUNTIME',tmp_path);store=lab_store.Store('run')
    try:
        store.preview({'episode':3,'sequence':7},'tx-live')
        latest=json.loads((tmp_path/'latest.json').read_text())
        assert latest['evaluation_preview'] is True and latest['evaluation_committed'] is False
        assert latest['evaluation_transaction_id']=='tx-live'
        assert store.db.execute('SELECT count(*) FROM steps').fetchone()[0]==0
    finally:store.close()
