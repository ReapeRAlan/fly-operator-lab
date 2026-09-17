import sys,json,sqlite3
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lab_history import page,detail


def database():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE steps(id INTEGER PRIMARY KEY,run TEXT,episode INTEGER,sequence INTEGER,payload TEXT)')
    return db


def put(db,ident,run='pilot',session='one',episode=1,seed=7,stage='move'):
    item=dict(game_session=session,episode=episode,sequence=ident,simulated_ms=ident*50,seed=seed,condition='adapter',stage=stage,mission='m-'+stage,
      observation={'operator':{'position':[ident,0,0],'health':100}},action_label='wait',action_receipt={'status':'completed'})
    db.execute('INSERT INTO steps VALUES(?,?,?,?,?)',(ident,run,episode,ident,json.dumps(item)))


def test_history_pages_are_bounded_filtered_and_isolated():
    db=database()
    for i in range(1,7):put(db,i,stage='move' if i%2 else 'rescue')
    put(db,7,run='other')
    first=page(db,'pilot',limit=2);assert [x['id'] for x in first['items']]==[6,5]
    second=page(db,'pilot',before=first['next_before'],limit=2);assert [x['id'] for x in second['items']]==[4,3]
    assert [x['id'] for x in page(db,'pilot',stage='rescue')['items']]==[6,4,2]
    assert detail(db,'pilot',7) is None


def test_trail_never_crosses_game_sessions_or_future_steps():
    db=database();put(db,1,session='old');put(db,2);put(db,3);put(db,4)
    result=detail(db,'pilot',3)
    assert [p['id'] for p in result['trail']]==[2,3]
    assert result['snapshot']['observation']['operator']['position']==[3,0,0]


def test_legacy_without_session_does_not_invent_continuous_trail():
    db=database();put(db,1,session=None);put(db,2,session=None)
    assert [p['id'] for p in detail(db,'pilot',2)['trail']]==[2]
