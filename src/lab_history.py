"""Bounded, read-only views of durable perception records; no hidden game state."""
import json


def page(db,run,before=0,limit=60,stage='',condition=''):
    clauses=['run=?'];args=[run]
    if before:clauses.append('id<?');args.append(before)
    if stage:clauses.append("json_extract(payload,'$.stage')=?");args.append(stage)
    if condition:clauses.append("json_extract(payload,'$.condition')=?");args.append(condition)
    limit=min(100,max(1,limit))
    rows=db.execute('SELECT id,payload FROM steps WHERE '+' AND '.join(clauses)+' ORDER BY id DESC LIMIT ?',[*args,limit+1]).fetchall()
    result=[]
    for ident,payload in rows[:limit]:
        item=json.loads(payload);actor=item.get('observation',{}).get('operator',{})
        result.append(dict(id=ident,**{k:item.get(k) for k in ('recorded_at','game_session','episode','sequence','simulated_ms','mission','stage','condition','seed','action_label','reward','success')},
          controller=item.get('decision',{}).get('controller'),receipt=item.get('action_receipt',{}),health=actor.get('health'),position=actor.get('position')))
    return {'items':result,'next_before':result[-1]['id'] if len(rows)>limit else None,'run':run}


def detail(db,run,ident):
    row=db.execute('SELECT payload FROM steps WHERE run=? AND id=?',(run,ident)).fetchone()
    if row is None:return None
    item=json.loads(row[0]);item['record_id']=ident
    decision=item.get('decision')
    if isinstance(decision,dict) and 'action_labels' not in decision and 'mask' in decision:
        from learning_adapter import ActionCatalog
        catalog=ActionCatalog();decision['action_labels']=[catalog.label(i) for i in range(len(decision['mask']))]
    session=item.get('game_session')
    # Episode counters restart when the game process restarts. Never join their trails.
    rows=db.execute("SELECT id,payload FROM steps WHERE run=? AND episode=? AND id<=? AND json_extract(payload,'$.game_session') IS ? AND json_extract(payload,'$.seed') IS ? AND json_extract(payload,'$.condition') IS ? AND json_extract(payload,'$.mission') IS ? ORDER BY id DESC LIMIT 240",
      (run,item['episode'],ident,session,item.get('seed'),item.get('condition'),item.get('mission'))).fetchall()
    trail=[]
    # Legacy rows without a process/session ID cannot safely join across launches.
    if session is None:rows=[(ident,row[0])]
    for key,payload in reversed(rows):
        value=json.loads(payload);position=value.get('observation',{}).get('operator',{}).get('position')
        if position:trail.append({'id':key,'position':position,'sequence':value.get('sequence'),'simulated_ms':value.get('simulated_ms')})
    return {'snapshot':item,'trail':trail,'trail_limit':240}
