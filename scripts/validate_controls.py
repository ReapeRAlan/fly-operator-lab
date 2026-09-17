"""Actual game integration checks; no synthetic observations or weapon edits."""
from pathlib import Path
import sys,json,time,math,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bridge_client import Bridge

def main():
    result={'session':json.loads((ROOT/'work/session.json').read_text()),'checks':{},'snapshots':{}}
    logpath=ROOT/'outputs/control_trace.jsonl'
    with logpath.open('w') as trace, Bridge() as b:
        o=b.new_episode(); result['snapshots']['initial']=o
        def steps(action,n=1,**args):
            nonlocal o
            for _ in range(n):
                prev=o['sim_time_ms'];o=b.step({'action':action,**args},50)
                assert o['sim_time_ms']-prev==50, 'Simulation clock mismatch'
                trace.write(json.dumps({'command':{'action':action,**args},'observation':o})+'\n')
            return o
        def check(name,passed):
            result['checks'][name]=bool(passed);result['snapshots'][name]=o
            print(name,passed,flush=True)
        steps('stop',60)
        check('no_automatic_shooting_visible_enemy',o['operator']['ammo']==31 and len(o['visible_enemies'])==1 and o['automatic_thinks_blocked']>=60)
        initial=o['operator']['position'];steps('move',10,destination=[-1,0,1])
        moved=o['operator']['position'];check('native_movement',math.dist(initial,moved)>.2)
        steps('stop');stop=o['operator']['position'];steps('aim',20,direction=[1,0,0])
        check('stop_mid_path',math.dist(stop,o['operator']['position'])<.02)
        steps('turn_left',20,direction=[0,0,-1]);check('turn_left',o['operator']['look'][2]<-.9)
        steps('turn_right',20,direction=[0,0,1]);check('turn_right',o['operator']['look'][2]>.9)
        steps('aim',20,direction=[0,0,1]);check('aim',o['operator']['aim'][2]>.9 and o['operator']['weapon_state']==5)
        # A second map fixture encloses the same armed enemy behind ordinary
        # walls, allowing a long empty-magazine/reload check without combat death.
        mapfile=ROOT/'work/profile/KillHouseGames/DoorKickers2/data/maps_wip/fly_validation.xml'
        original=mapfile.read_bytes()
        tree=ET.fromstring(original)
        wall=ET.SubElement(tree.find('Entities'),'Entity',{'template':'Wall White','id':'150','pos':'0 0 0','rot':'1 0 0 0 1 0 0 0 1'})
        shape=ET.SubElement(wall,'Wall',{'numTrees':'1','numNodes':'5'})
        ET.SubElement(shape,'Branch',{'parent':'0','p0':'2 -2','p1':'6 -2','p2':'6 2','p3':'2 2','p4':'2 -2'})
        try:
            mapfile.write_bytes(ET.tostring(tree));o=b.new_episode()
        finally:mapfile.write_bytes(original)
        result['snapshots']['weapon_fixture']=o
        steps('turn_right',20,direction=[0,0,1])
        before=o['operator']['ammo'];shots=o['shots_accepted'];steps('fire',20,direction=[0,0,1])
        check('fire_uses_ammunition_and_native_cooldown',0<before-o['operator']['ammo']<=12 and o['shots_accepted']-shots==before-o['operator']['ammo'])
        ammo=o['operator']['ammo'];steps('stop',20)
        check('no_firing_without_order',o['operator']['ammo']==ammo)
        steps('fire',80,direction=[0,0,1]);check('can_empty_magazine',o['operator']['ammo']==0)
        steps('stop',100);check('no_automatic_reload',o['operator']['ammo']==0 and o['reload_requests']==0)
        steps('reload');check('reload_not_instant',o['operator']['ammo']==0 and o['operator']['commands']>0)
        steps('aim',140,direction=[0,0,1]);check('native_reload_completed',o['operator']['ammo']>=30 and o['operator']['commands']==0)
        before=o;reject=b.request('step',episode=o['episode'],sequence=o['sequence'],dt_ms=50,command={'action':'fire'})
        o=b.request('observe');check('reject_duplicate_sequence',not reject['ok'] and o['sequence']==before['sequence'] and o['sim_time_ms']==before['sim_time_ms'])
        seq=o['sequence'];clock=o['sim_time_ms'];ammo=o['operator']['ammo']
    time.sleep(1)
    with Bridge() as b:
        o=b.request('observe');result['checks']['disconnect_pauses']=o['sim_time_ms']==clock and o['sequence']==seq and o['operator']['ammo']==ammo
        prior=o['episode'];o=b.new_episode(record=True)
        result['checks']['restart_new_episode']=o['episode']==prior+1 and o['sequence']==0 and o['sim_time_ms']==0
        for i in range(3):o=b.step({'action':'stop'},[33,33,34][i])
        result['snapshots']['capture']=o
        result['checks']['capture_client_matches_server']=o['frame']['client_sim_time_ms']==o['sim_time_ms']
    result['passed']=all(result['checks'].values())
    (ROOT/'outputs/control_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':result['passed'],'checks':result['checks']},indent=2))
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
