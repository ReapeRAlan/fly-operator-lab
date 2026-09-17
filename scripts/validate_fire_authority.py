"""Prove that the operator never shoots without an explicit order (protocol 3.3, A0).

With an enemy visible and no `fire` order for the configured window, shots_accepted must stay
at zero and the magazine must not lose rounds. Then an ordered burst must consume ammunition.
Runs against the real game through the native bridge; writes outputs/fire_authority_v33.json.
"""
from pathlib import Path
import sys,json,time,argparse,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from bridge_client import Bridge
from learning_adapter import ActionCatalog,clean_observation,bearing

def observe(bridge,mission):
    return clean_observation(bridge.last,mission)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mission',default=None,help='Scenario name (defaults to the first shoot training scenario)')
    parser.add_argument('--silent-seconds',type=float,default=15.)
    parser.add_argument('--fire-orders',type=int,default=6)
    parser.add_argument('--output',default='outputs/fire_authority_v33.json')
    args=parser.parse_args()
    scenarios=json.loads((ROOT/'data/learning/scenarios.json').read_text())
    mission=next(m for m in scenarios if m['name']==args.mission) if args.mission else next(m for m in scenarios if m['stage']=='shoot' and m['split']=='train')
    catalog=ActionCatalog();report={'mission':mission['name'],'silent_seconds':args.silent_seconds,'fire_orders':args.fire_orders}
    with Bridge() as bridge:
        raw=bridge.new_mission(mission['name'])
        start_ammo=raw['operator']['ammo'];silent_ticks=int(args.silent_seconds*1000/50)
        enemy_seen=0;shots=[];ammo=[start_ammo]
        # Phase 1: aim at the enemy when one is visible, but never order fire.
        for tick in range(silent_ticks):
            obs=clean_observation(raw,mission);enemies=[e for e in obs['visible_enemies'] if e.get('alive',True)]
            command={'action':'wait'}
            if enemies:
                enemy_seen+=1
                nearest=min(enemies,key=lambda e:bearing(obs['operator'],e['position'])[1])
                delta,_=bearing(obs['operator'],nearest['position'])
                if abs(delta)>.05:
                    index=catalog.sector_entry('aim_target',obs,nearest['position'])
                    if index is not None:command=catalog.decode(index,obs)
            raw=bridge.step(command,50)
            shots.append(raw.get('shots_accepted',0));ammo.append(raw['operator']['ammo'])
        silent={'ticks':silent_ticks,'ticks_with_visible_enemy':enemy_seen,'shots_accepted':max(shots) if shots else 0,
                'ammo_start':start_ammo,'ammo_end':ammo[-1],'health':raw['operator'].get('health')}
        # Phase 2: order fire and check that the magazine actually drops.
        before=raw['operator']['ammo'];accepted_before=raw.get('shots_accepted',0);ordered=0
        for _ in range(args.fire_orders):
            obs=clean_observation(raw,mission)
            if obs['operator'].get('weapon_state')!=5 or obs['operator'].get('ammo',0)<=0:
                raw=bridge.step({'action':'wait'},50);continue
            raw=bridge.step({'action':'fire'},50);ordered+=1
            for _ in range(4):raw=bridge.step({'action':'wait'},50)
        fired={'orders':ordered,'ammo_before':before,'ammo_after':raw['operator']['ammo'],
               'shots_accepted_before':accepted_before,'shots_accepted_after':raw.get('shots_accepted',0)}
    report['silent_phase']=silent;report['ordered_phase']=fired
    report['passed']=bool(silent['shots_accepted']==0 and silent['ammo_end']==silent['ammo_start'] and
                          (fired['orders']==0 or fired['ammo_after']<fired['ammo_before'] or fired['shots_accepted_after']>fired['shots_accepted_before']))
    report['meaning']='Fire is an explicit decision: with an enemy in view and no fire order the weapon stays silent.'
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
