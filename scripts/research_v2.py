"""Extract the additional PDB layouts and decompilation targets for protocol v2."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
symbols=json.loads((ROOT/'work/re/symbols.json').read_text())
names=['Human_Server::SetEquippedItem','Human_Server::ProcessCmdItemEquip',
 'Human_Server::CmdQueueDoorOpen','Human_Server::CmdQueueDoorBreach','Human_Server::CmdQueueGrenadeThrow',
 'Human_Server::CmdQueueUse','Human_Server::CmdCancelUse','Human_Server::CmdQueueTimebombDefuse',
 'Human_Server::CmdQueueArrest','Human_Server::CmdFollowToggle','Human_Server::CmdEvacuate',
 'Human_Server::CancelCommands','Human_Server::ClearCommands','Human_Server::GetEquipmentByType',
 'Inventory_Server::Equip','Scenario_Server::Evaluate','Scenario_Server::GatherStatistics',
 'Human_Server::ProcessCmdInteract','Human_Server::CmdQueueCrouch','Door_Server::Open']
(ROOT/'work/re/v2_targets.tsv').write_text('\n'.join(f'{0x140000000+symbols[n]["rva"]:x}\t{n}' for n in names if n in symbols))
types=(ROOT/'work/re/types.txt').read_text()
lookup={}
for rec in re.split(r'(?=\s+0x[0-9A-Fa-f]+ \| LF_)',types):
 m=re.match(r'\s*(0x[0-9A-Fa-f]+) \|',rec)
 if m:lookup[m[1]]=rec.strip()
classes={}
targets=['Equipment_Template','Door_Server','Scenario_Server','Equipment','Firearm','Entity_Template','Human_Server::eCommand','Human_Server','sScenario','ModifiableParams','Inventory_Server']
for k,v in lookup.items():
 m=re.search(r'LF_(?:CLASS|STRUCTURE|ENUM).*?`([^`]+)`',v,re.S)
 if m and m[1] in targets and 'forward ref' not in v:
  f=re.search('field list: (0x[0-9A-Fa-f]+)',v)
  classes[k]=v+'\n'+lookup.get(f[1],'') if f else v
(ROOT/'work/re/v2_classes.json').write_text(json.dumps(classes,indent=2))
print('Targets',len(names),'layouts',len(classes))
for k,v in classes.items():
 if re.search(r'`(?:Door_Server|Equipment_Template|Scenario_Server|Equipment|sScenario)`',v.split('\n')[0]):print(v[:14000])
