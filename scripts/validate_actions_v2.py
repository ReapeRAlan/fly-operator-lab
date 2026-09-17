from pathlib import Path
import sys,json,time,math,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bridge_client import Bridge

def main():
    results=[];b=Bridge();session=dict(b.session)
    (ROOT/'outputs/native_actions_v2.meta.json').write_text(json.dumps({'version':1,'session':session},indent=2))
    def test(stage,command,criterion,seconds=15):
        try:
            initial=b.new_mission(f'fly_{stage}_train_7');o=b.step(command,50);trace=[]
            for i in range(int(seconds*20)):
                trace.append({'time':o['sim_time_ms'],'receipt':o.get('action_receipt'),'operator':o.get('operator'),'objects':o.get('objects'),'result':o.get('mission_result')})
                if criterion(initial,o):break
                if o.get('action_receipt',{}).get('status') in ('rejected','failed') or o.get('game_state')==2:break
                o=b.step({'action':'wait'},50)
            ok=bool(criterion(initial,o));results.append({'stage':stage,'command':command,'passed':ok,'trace':trace,'inventory':o.get('inventory')})
            print(json.dumps({'stage':stage,'action':command['action'],'passed':ok,'receipt':o.get('action_receipt'),'result':o.get('mission_result')}),flush=True)
        except Exception as e:results.append({'stage':stage,'command':command,'passed':False,'error':str(e)});print(traceback.format_exc(),flush=True)
        (ROOT/'outputs/native_actions_v2.json').write_text(json.dumps(results,indent=2))
    done=lambda a,o:o.get('action_receipt',{}).get('status')=='completed'
    test('switch',{'action':'equip','slot':5},lambda a,o:any(e['slot']==5 and e['equipped'] and e['state']>2 for e in o['inventory']))
    test('move',{'action':'move','destination':[-3,0,0]},lambda a,o:math.dist(o['operator']['position'],[-3,0,0])<.2)
    test('door',{'action':'door_open','target_id':300},lambda a,o:any(x.get('id')==300 and x.get('door_state')==0 for x in o['objects']))
    test('breach',{'action':'door_breach','slot':16,'target_id':300},lambda a,o:any(x.get('id')==300 and x.get('door_state')==0 for x in o['objects']))
    for slot in (12,13):test('grenade',{'action':'throw','slot':slot,'destination':[4,0,0]},lambda a,o:any(x['slot']==slot and x['quantity']<2 for x in o['inventory']))
    test('defuse',{'action':'defuse','target_id':300},lambda a,o:o.get('mission_result')==1)
    # Follow must visibly move the hostage with the operator, not only empty a queue.
    initial=b.new_mission('fly_rescue_train_7');o=b.step({'action':'follow','target_id':300},50)
    o=b.step({'action':'move','destination':[-6,0,0]},50)
    for _ in range(300):
        if o.get('mission_result') or o.get('game_state')==2:break
        o=b.step({'action':'wait'},50)
    results.append({'stage':'rescue','command':{'action':'follow'},'passed':o.get('mission_result')==1,'final':o})
    (ROOT/'outputs/native_actions_v2.json').write_text(json.dumps(results,indent=2));b.close()
    from build_native_capabilities import build
    cap=build()
    print(json.dumps({'validated_actions':cap['validated_actions']}),flush=True)
if __name__=='__main__':main()
