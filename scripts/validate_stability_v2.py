from pathlib import Path
import sys,json,time,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bridge_client import Bridge
def main():
    cases=[('switch',{'action':'equip','slot':5}),('door',{'action':'door_open','target_id':300}),('breach',{'action':'door_breach','slot':16,'target_id':300}),('defuse',{'action':'defuse','target_id':300}),('grenade',{'action':'throw','slot':12,'destination':[4,0,0]})]
    rows=[];started=time.time()
    with Bridge() as b:
        for i in range(20):
            stage,command=cases[i%5];o=b.new_mission(f'fly_{stage}_train_{7+i}');episode=o['episode'];sequences=[]
            for step in range(300):
                o=b.step(command if step==0 else {'action':'wait'},50);sequences.append(o['sequence'])
                if o['action_receipt']['status'] in ('completed','rejected','failed'):break
            passed=o['action_receipt']['status']=='completed' and sequences==list(range(1,len(sequences)+1)) and o['episode']==episode
            rows.append({'episode':episode,'stage':stage,'passed':passed,'steps':len(sequences),'simulated_ms':o['sim_time_ms'],'receipt':o['action_receipt'],'mission_result':o['mission_result']})
            assert passed,rows[-1]
    report={'passed':True,'episodes':20,'scope':'native bridge action cycles; instructor commands, no neuronal learning claim','rows':rows,'session':json.loads((ROOT/'work/session.json').read_text()),'wall_seconds':time.time()-started}
    (ROOT/'outputs/stability_v2.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('rows','session')}))
if __name__=='__main__':main()
