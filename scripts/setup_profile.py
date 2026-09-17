from pathlib import Path
import os,shutil,json,time,xml.etree.ElementTree as ET,zipfile
ROOT=Path(__file__).resolve().parents[1]
GAME=Path(r'D:\SteamLibrary\steamapps\common\DoorKickers2')
PROFILE=ROOT/'work/profile/KillHouseGames/DoorKickers2'

def main():
    original=Path(os.environ['LOCALAPPDATA'])/'KillHouseGames/DoorKickers2'
    backup=ROOT/'work/original_profile_backup'; backup.mkdir(parents=True,exist_ok=True)
    PROFILE.mkdir(parents=True,exist_ok=True)
    for p in original.glob('*'):
        if p.is_file() and p.suffix in ['.xml','.bak'] and not (backup/p.name).exists(): shutil.copy2(p,backup/p.name)
    options=ET.parse(original/'options.xml'); root=options.getroot()
    for child in list(root):
        if child.tag=='Mods': root.remove(child)
    root.find('Video').set('mode','0'); root.find('Video').set('width','1280'); root.find('Video').set('height','720')
    root.find('Video').set('skipLogo','true'); root.find('Video').set('maxFPS','60')
    root.find('Sound').set('musicVolume','0'); root.find('Game').set('lockCursorInBorderless','false')
    root.find('DevMode').set('value','false')
    mod=PROFILE/'mods/fly_operator_lab'; (mod/'maps_ugc').mkdir(parents=True,exist_ok=True)
    ET.SubElement(root,'Mods',{'compat':'35','path0':str(mod)})
    options.write(PROFILE/'options.xml',encoding='utf-8',xml_declaration=True)
    # Let the game create a stock roster. The user's roster may depend on mods.
    for n in ['keybinds.xml']:
        if not (PROFILE/n).exists(): shutil.copy2(original/n,PROFILE/n)
    (mod/'mod.xml').write_text('<Mod title="Fly Operator Lab" description="Local MaleCNS interoperability validation" author="Local experiment" tags="Missions" gameVersion="112"/>',encoding='utf-8')
    z=zipfile.ZipFile(GAME/'data/maps/data_store.zip'); src=ET.fromstring(z.read('tiny_trouble_06.xml'))
    m=ET.Element('Map',{'name':'Fly Operator - Validation Arena'})
    ET.SubElement(m,'Scenario',{'type':'ClearHostiles','numDeployableTroops':'1','referenceWinTimeSeconds':'60','challenges':'0','nightMission':'false'})
    m.append(src.find('Graphics'))
    terrain=ET.SubElement(m,'Terrain',{'widthMeters':'20','heightMeters':'12'}); ET.SubElement(terrain,'Materials',{'num':'1','mat0':'tiles_02'})
    ET.SubElement(m,'Description',{'text':'MaleCNS full-network experiment. One operator, one hostile. External step controller.'})
    ET.SubElement(m,'MusicTracks'); ents=ET.SubElement(m,'Entities')
    for e in src.find('Entities'):
        if e.get('template') in ['Floor','Light Ambiental','Light Directional','Default Effect (Outdoors)','Default Effect (Indoors)']:
            ents.append(e)
    ident='1 0 0 0 1 0 0 0 1'
    ET.SubElement(ents,'Entity',{'template':'Deploy Slot','id':'102','pos':'-4 0.1 0','rot':'0 0 1 0 1 0 -1 0 0'})
    op=ET.SubElement(ents,'Entity',{'template':'Ranger Assaulter','id':'100','pos':'-4 0 0','rot':'0 0 1 0 1 0 -1 0 0'})
    ET.SubElement(ET.SubElement(op,'Human',{'needsRescue':'false'}),'Traits')
    enemy=ET.SubElement(ents,'Entity',{'template':'Grunt Insurgent','id':'101','pos':'4 0 0','rot':'0 0 1 0 1 0 -1 0 0'})
    ET.SubElement(ET.SubElement(enemy,'Human',{'needsRescue':'false'}),'Traits',{'chanceToInvestigate':'0','chanceToAmbush':'1','chanceToBlindFire':'0'})
    ET.indent(m); ET.ElementTree(m).write(mod/'maps_ugc/fly_validation.xml',encoding='utf-8',xml_declaration=True)
    wip=PROFILE/'data/maps_wip';wip.mkdir(parents=True,exist_ok=True)
    shutil.copy2(mod/'maps_ugc/fly_validation.xml',wip/'fly_validation.xml')
    print(PROFILE)

if __name__=='__main__': main()
