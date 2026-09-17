"""Build the deterministic action-capability manifest from every native test artifact."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name):
    path=OUT/name
    if not path.exists():raise RuntimeError(f'Missing native evidence: {path}')
    return path,json.loads(path.read_text(encoding='utf-8'))

def build():
    control_path,control=load('control_validation.json')
    actions_path,actions=load('native_actions_v2.json')
    kits_path,kits=load('native_kits_v2.json')
    actions_meta_path,actions_meta=load('native_actions_v2.meta.json')
    kits_meta_path,kits_meta=load('native_kits_v2.meta.json')
    if not isinstance(actions,list) or not isinstance(kits,list):
        raise RuntimeError('Native evidence must be a JSON list')
    checks=control.get('checks',{})
    basic={
      'wait':['no_firing_without_order'],
      'stop':['stop_mid_path'],
      'turn':['turn_left','turn_right'],
      'aim_target':['aim'],
      'fire':['fire_uses_ammunition_and_native_cooldown'],
      'reload':['reload_not_instant','native_reload_completed'],
    }
    evidence={}
    for action,required in basic.items():
        if all(checks.get(check) is True for check in required):
            evidence[action]={'source':control_path.name,'checks':required}
    for row in actions:
        if not row.get('passed'):continue
        command=row.get('command',{})
        action=command.get('action')
        if action:
            evidence[action]={'source':actions_path.name,'stage':row.get('stage')}
    loadouts=kits[:3]
    if len(loadouts)==3 and all(row.get('passed') for row in loadouts):
        evidence['loadout']={'source':kits_path.name,'kits':[row.get('kit') for row in loadouts]}
    for row in kits[3:]:
        action=row.get('action')
        if action and row.get('passed'):
            evidence[action]={'source':kits_path.name}
    catalog={'wait','stop','cancel','reload','fire','equip','crouch','move','turn','aim_target','door_open','door_breach','throw','defuse','follow','use','clear_obstacle','arrest','spy_camera','evacuate','loadout'}
    sessions=[control.get('session',{}),actions_meta.get('session',{}),kits_meta.get('session',{})]
    identities={(s.get('exe_sha256'),s.get('dll_sha256')) for s in sessions}
    if len(identities)!=1 or None in next(iter(identities)):
        raise RuntimeError('Native evidence does not come from one executable/DLL build')
    sources=[]
    for path,meta_path,session in ((control_path,None,sessions[0]),(actions_path,actions_meta_path,sessions[1]),(kits_path,kits_meta_path,sessions[2])):
        item={'file':path.name,'sha256':digest(path),'session':{'pid':session.get('pid'),'started_unix':session.get('started_unix'),
              'exe_sha256':session.get('exe_sha256'),'dll_sha256':session.get('dll_sha256')}}
        if meta_path:item['metadata']={'file':meta_path.name,'sha256':digest(meta_path)}
        sources.append(item)
    generator=Path(__file__)
    result={
      'version':2,
      'manifest_version':1,
      'validated_actions':sorted(evidence),
      'pending':sorted(catalog-set(evidence)),
      'evidence':{key:evidence[key] for key in sorted(evidence)},
      'sources':sources,
      'generator':{'file':str(generator.relative_to(ROOT)).replace('\\','/'),'sha256':digest(generator)},
    }
    target=OUT/'native_capabilities_v2.json'
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    print(json.dumps(build(),indent=2))
