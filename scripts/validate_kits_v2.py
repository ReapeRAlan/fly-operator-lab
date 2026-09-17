from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bridge_client import Bridge
def main():
    results=[];session=None
    with Bridge() as b:
        session=dict(b.session)
        for kit in range(3):
            initial=b.new_mission('fly_loadout_train_7');o=b.step({'action':'loadout','kit':kit},50)
            for _ in range(100):
                if o['action_receipt']['status'] in ('completed','rejected','failed'):break
                o=b.step({'action':'wait'},50)
            primary=next(e for e in o['inventory'] if e['slot']==1)
            passed=primary['name']==('M16A4' if kit==0 else 'M4 Carbine') and o['operator']['ammo']>0 and o['action_receipt']['status']=='completed'
            equip_receipt=dict(o['action_receipt']);inventory=o['inventory']
            o=b.step({'action':'aim','direction':[0,0,1]},50)
            for _ in range(100):
                if o['operator']['weapon_state']==5:break
                o=b.step({'action':'wait'},50)
            before=o.get('shots_accepted',0);o=b.step({'action':'fire'},50)
            passed=passed and o.get('shots_accepted',0)>before
            results.append({'kit':kit,'passed':passed,'inventory':inventory,'receipt':equip_receipt,'fire_receipt':o['action_receipt'],'shots_accepted':o.get('shots_accepted',0)})
            print(json.dumps({'kit':kit,'passed':passed,'weapon':primary['name']}),flush=True)
        # Crouch and cancel are tested by motor flags and queue state.
        b.new_mission('fly_move_train_7');o=b.step({'action':'crouch','value':True},50)
        for _ in range(60):
            if o['action_receipt']['status'] in ('completed','rejected'):break
            o=b.step({'action':'wait'},50)
        results.append({'action':'crouch','passed':o['action_receipt']['status']=='completed' and o['operator'].get('crouched') is True,'receipt':o['action_receipt'],'crouched':o['operator'].get('crouched')})
        b.new_mission('fly_breach_train_7');o=b.step({'action':'door_breach','slot':16,'target_id':300},50);o=b.step({'action':'cancel'},50)
        for _ in range(300):
            if o['operator']['commands']==0:break
            o=b.step({'action':'wait'},50)
        results.append({'action':'cancel','passed':o['operator']['commands']==0 and not any(x.get('id')==300 and x.get('door_state')==0 for x in o['objects']),'receipt':o['action_receipt']})
    (ROOT/'outputs/native_kits_v2.json').write_text(json.dumps(results,indent=2))
    (ROOT/'outputs/native_kits_v2.meta.json').write_text(json.dumps({'version':1,'session':session},indent=2))
    from build_native_capabilities import build
    build()
    print(json.dumps(results[3:]),flush=True)
if __name__=='__main__':main()
