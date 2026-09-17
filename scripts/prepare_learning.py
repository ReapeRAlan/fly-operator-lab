"""Build the catalog and deterministic isolated lab scenarios; never edit game files."""
from pathlib import Path
import copy,json,re,hashlib,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
GAME=Path(json.loads((ROOT/'config/build.json').read_text())['game_path'])
MAPS=ROOT/'work/profile/KillHouseGames/DoorKickers2/data/maps_wip'
IDENT='1 0 0 0 1 0 0 0 1'

def parsed(path):
    text=path.read_text(encoding='utf-8-sig',errors='replace')
    # The game XML parser accepts bare ampersands in human-readable strings.
    text=re.sub(r'&(?!#\d+;|#x[\da-fA-F]+;|amp;|lt;|gt;|quot;|apos;)', '&amp;', text)
    return ET.fromstring(text)

def catalog():
    items={};bindings=[];sources=[];errors=[]
    for path in sorted((GAME/'data/equipment').glob('*.xml')):
        try:r=parsed(path)
        except ET.ParseError as e:errors.append({'file':str(path),'error':str(e)});continue
        sources.append({'path':str(path.relative_to(GAME)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for e in r:
            if e.tag in ('Bind','Unbind'):
                bindings.append({'operation':e.tag,**e.attrib,'to_children':[c.attrib for c in e]});continue
            if not e.get('name'):continue
            name=e.get('name');items[name]={'name':name,'family':e.tag,'attributes':e.attrib,'sections':{c.tag:c.attrib for c in e},'complete_xml':ET.tostring(e,encoding='unicode'),'source':str(path.relative_to(GAME))}
    out={'version':2,'meaning':'Stock XML templates; runtime effective parameters are recorded separately. XML file precedence may include later overrides.','items':items,'bindings':bindings,'sources':sources,'parse_errors':errors}
    (ROOT/'data/learning').mkdir(parents=True,exist_ok=True)
    (ROOT/'data/learning/equipment_catalog.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    return len(items),errors

def entity(parent,template,ident,pos,rotation=IDENT):
    return ET.SubElement(parent,'Entity',{'template':template,'id':str(ident),'pos':' '.join(map(str,pos)),'rot':rotation})

def scenario(stage,seed,split='train'):
    """Independent geometry RNG; engine's actual map RNG seed is logged separately."""
    rng=np.random.default_rng(seed);name=f'fly_{stage}_{split}_{seed}'
    tree=ET.parse(MAPS/'fly_validation.xml');root=tree.getroot();root.set('name',name)
    root.find('Terrain').set('widthMeters','64');root.find('Terrain').set('heightMeters','64')
    sc=root.find('Scenario');sc.set('type','HostageRescue' if stage=='rescue' else 'BombDefusal' if stage=='defuse' else 'ClearHostiles')
    es=root.find('Entities')
    for e in list(es):
        if int(e.get('id',0))>=100:es.remove(e)
    start=[-4.,0.,0.];angle=float(rng.uniform(-np.pi,np.pi));goal=[start[0]+float(np.cos(angle)*3),0.,float(np.sin(angle)*3)]
    entity(es,'Deploy Slot',102,[-4,.1,0],'0 0 1 0 1 0 -1 0 0')
    op=entity(es,'Ranger Assaulter',100,start,'0 0 1 0 1 0 -1 0 0');ET.SubElement(ET.SubElement(op,'Human',{'needsRescue':'false'}),'Traits')
    # Initial reload instruction isolates the native weapon cycle from incoming fire.
    # Combat remains active in shooting, elimination and loadout exercises.
    combat=stage in ['shoot','elimination','loadout']
    enemypos=[float(rng.uniform(3,7)),0.,float(rng.uniform(-3,3))] if combat else [24.,0.,24.]
    enemy=entity(es,'Grunt Insurgent',101,enemypos,'0 0 1 0 1 0 -1 0 0')
    ET.SubElement(ET.SubElement(enemy,'Human',{'needsRescue':'false'}),'Traits',{'chanceToInvestigate':'0','chanceToAmbush':'1','chanceToBlindFire':'0'})
    if stage=='defuse':es.remove(enemy)
    geometry=[]
    if not combat:
        wall=entity(es,'Wall White',200,[20,0,20]);points=[[0,0],[8,0],[8,8],[0,8],[0,0]]
        branch=ET.SubElement(ET.SubElement(wall,'Wall',{'numTrees':'1','numNodes':'5'}),'Branch',{'parent':'0'})
        for i,p in enumerate(points):branch.set(f'p{i}',f'{p[0]} {p[1]}')
        geometry.append({'kind':'wall','points':[[20+p[0],20+p[1]] for p in points]})
    target_id=None
    if stage in ('door','breach','cancel'):
        target_id=300;pos=[-2.,0.,0.]
        entity(es,'door_wood_01_locked' if stage in ('breach','cancel') else 'door_wood_01',target_id,pos,'0 0 1 0 1 0 -1 0 0')
        entity(es,'door_frame_wood_01',301,pos,'0 0 1 0 1 0 -1 0 0')
        goal=[-2.9,0,0]
    elif stage=='defuse':
        target_id=300;e=entity(es,'Timebomb Visible',300,[-2,0,0]);ET.SubElement(e,'Timebomb',{'kaboomTimeMsec':'60000','defuseTimeMsec':'5000'});goal=[-2.8,0,0]
    elif stage=='rescue':
        target_id=300;e=entity(es,'HostageVisible',300,[-2,0,0]);ET.SubElement(e,'Human',{'needsRescue':'true'})
        z=entity(es,'RescueZone',301,[-6,.03,0]);ET.SubElement(z,'RescueZone',{'type':'Hostage'})
        ET.SubElement(z,'Shape',{'type':'rectangle','width':'3','height':'3'})
        goal=[-6,0,0]
    elif stage=='grenade':goal=[2,0,float(rng.uniform(-2,2))]
    description=root.find('Description');description.set('text',f'Fly Operator research exercise: {stage}. Deterministic layout seed {seed}.')
    ET.indent(root);tree.write(MAPS/(name+'.xml'),encoding='utf-8',xml_declaration=True)
    descriptor={'name':name,'stage':stage,'seed':seed,'split':split,'goal':goal,'goal_angle':angle,'target_id':target_id,
      'geometry':geometry,'max_seconds':60 if stage in ('rescue','defuse','elimination','loadout') else 30,
      'victory_scope':'native_mission' if stage in ('rescue','defuse','elimination','loadout') else 'isolated_skill',
      'seed_scope':'layout and neuronal RNG; actual engine map_seed is recorded; engine determinism must be verified',
      'file_sha256':hashlib.sha256((MAPS/(name+'.xml')).read_bytes()).hexdigest()}
    return descriptor

def main():
    count,errors=catalog();manifest=[]
    stages=json.loads((ROOT/'config/learning.json').read_text())['stage_order']
    for stage in stages:
        for split,seeds in [('train',range(7,31)),('validation',range(1000,1090)),('test',range(2000,2030))]:
            for seed in seeds:manifest.append(scenario(stage,seed,split))
    (ROOT/'data/learning/scenarios.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'catalog_items':count,'catalog_parse_errors':errors,'scenarios':len(manifest)}))
if __name__=='__main__':main()
