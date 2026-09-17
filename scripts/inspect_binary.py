"""Repeatable static inspection of the authorized local game build."""
from pathlib import Path
import struct, uuid, hashlib, subprocess, re, json, ctypes
ROOT=Path(__file__).resolve().parents[1]
GAME=Path(r'D:\SteamLibrary\steamapps\common\DoorKickers2')
LLVM=Path(r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin')
OUT=ROOT/'work/re'; OUT.mkdir(parents=True,exist_ok=True)

def dump(*args):
    return subprocess.check_output([str(LLVM/'llvm-pdbutil.exe'),'dump',*args,str(GAME/'DoorKickers2.pdb')],text=True,errors='replace')

def pe_info():
    b=(GAME/'DoorKickers2.exe').read_bytes(); pe=struct.unpack_from('<I',b,0x3c)[0]; opt=pe+24
    machine,nsec=struct.unpack_from('<HH',b,pe+4); osz=struct.unpack_from('<H',b,pe+20)[0]
    base=struct.unpack_from('<Q',b,opt+24)[0]; sections=[]
    for i in range(nsec):
        p=opt+osz+40*i; name=b[p:p+8].rstrip(b'\0').decode(); vs,va,rs,rp=struct.unpack_from('<4I',b,p+8); sections.append(dict(name=name,rva=va,vsize=vs,raw_size=rs,raw=rp))
    def offset(rva):
        for s in sections:
            if s['rva']<=rva<s['rva']+max(s['vsize'],s['raw_size']): return s['raw']+rva-s['rva']
        raise ValueError(rva)
    dr,sz=struct.unpack_from('<II',b,opt+112+6*8)
    for p in range(offset(dr),offset(dr)+sz,28):
        _,_,_,_,kind,n,_,ptr=struct.unpack_from('<IIHHIIII',b,p)
        if kind==2 and b[ptr:ptr+4]==b'RSDS':
            guid=str(uuid.UUID(bytes_le=b[ptr+4:ptr+20])); age=struct.unpack_from('<I',b,ptr+20)[0]
    return dict(game_path=str(GAME),sha256=hashlib.sha256(b).hexdigest(),pdb_sha256=hashlib.sha256((GAME/'DoorKickers2.pdb').read_bytes()).hexdigest(),guid=guid,age=age,machine=machine,image_base=base,sections=sections)

def main():
    profile=pe_info(); (OUT/'summary.txt').write_text(dump('--summary'),encoding='utf-8')
    d=ctypes.WinDLL('dbghelp'); d.UnDecorateSymbolName.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint32,ctypes.c_uint32]
    def undec(s):
        b=ctypes.create_string_buffer(4096)
        return b.value.decode(errors='replace') if d.UnDecorateSymbolName(s.encode(),b,4096,0x1000) else s
    publics=dump('--publics'); symbols={}
    for r in re.split(r'(?=\s+\d+ \| S_PUB32)',publics):
        n=re.search(r'`([^`]+)`',r,re.S); a=re.search(r'addr = (\d+):(\d+)',r)
        if not(n and a): continue
        mangled=re.sub(r'\s+','',n[1]); name=undec(mangled); sec=int(a[1]); off=int(a[2])
        if sec and sec<=len(profile['sections']): symbols[name]={'rva':profile['sections'][sec-1]['rva']+off,'mangled':mangled}
    (OUT/'symbols.json').write_text(json.dumps(symbols,indent=2),encoding='utf-8')
    modules=dump('--modules'); (OUT/'modules.txt').write_text(modules,encoding='utf-8')
    wanted=['Brain','BrainPlayer','Human_Server','Game_Server','Game_Client','Replay','Firearm','Entity_Server','Entity_Common','AI','Pathfinder','Game','Game_Common','Vis','main','OS_Win32','Input']
    funcs={}
    for mod in re.split(r'(?=\s+Mod \d+)',modules):
        m=re.search(r'Mod (\d+) \| `([^`]+)`',mod)
        if not m or Path(m[2]).stem not in wanted: continue
        text=dump('--symbols','--modi='+str(int(m[1]))); stem=Path(m[2]).stem
        (OUT/(stem+'.symbols.txt')).write_text(text,encoding='utf-8')
        (OUT/(stem+'.files.txt')).write_text(dump('--files','--modi='+str(int(m[1]))),encoding='utf-8')
        for rec in re.split(r'(?=\s+\d+ \| S_)',text):
            f=re.search(r'S_(?:G|L)PROC32.*?`([^`]+)`.*?addr = (\d+):(\d+), code size = (\d+).*?type = `([^`]+)`',rec,re.S)
            if not f: continue
            rva=profile['sections'][int(f[2])-1]['rva']+int(f[3]); funcs[f[1]]=dict(rva=rva,size=int(f[4]),type=f[5],module=stem)
    (OUT/'functions.json').write_text(json.dumps(funcs,indent=2),encoding='utf-8')
    types=dump('--types'); (OUT/'types.txt').write_text(types,encoding='utf-8')
    lookup={}
    for rec in re.split(r'(?=\s+0x[0-9A-Fa-f]+ \| LF_)',types):
        m=re.match(r'\s*(0x[0-9A-Fa-f]+) \|',rec)
        if m: lookup[m[1]]=rec.strip()
    classes={}
    for key,rec in lookup.items():
        m=re.search(r'LF_(?:CLASS|STRUCTURE|ENUM).*?`([^`]+)`',rec,re.S)
        if m and any(x in m[1] for x in ['Human_Server','BrainPlayer','sFirearmTarget','sFirearmShoot','GameServer','HumanCommon','Human_Common','HumanShared','Human_Shared','Entity_Common','Entity_Server','Inventory','HumanStats','Firearm','sAwarenessEvent','sVisEntity']):
            if 'forward ref' not in rec:
                field=re.search('field list: (0x[0-9A-Fa-f]+)',rec)
                classes[key]=rec+'\n'+lookup.get(field[1],'') if field else rec
    (OUT/'classes.json').write_text(json.dumps(classes,indent=2),encoding='utf-8')
    profile['symbols']={k:v for k,v in symbols.items() if any(x in k for x in ['g_pServer','g_pClient','g_game','GameServer::Update','BrainPlayer::Think','Human_Server::Cmd','Human_Server::ReplaceWaypoints'])}
    (ROOT/'config/build.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    targets=['AI::BrainPlayer::Think','AI::BrainPlayer::DoEngage','Human_Server::Update','Human_Server::CmdFirearmShoot','Human_Server::ReplaceWaypoints','Human_Server::OnVisEntityInView','Human_Server::OnVisEnd','Human_Server::UpdateFOV','GameServer::Update','GameServer::UpdateGame','GameServer::LoadMap','GameServer::Pause','GameClient::UpdateReplay','GameClient::RequestReplayStart','Replay::StopRecording','Firearm::Shoot','GameServer::EvaluateWinLoseConditions','GameServer::IsGamePaused']
    rows=[]
    for name in targets:
        if name not in funcs: continue
        f=funcs[name]; addr=profile['image_base']+f['rva']; safe=name.replace('::','_')
        asm=subprocess.check_output([str(LLVM/'llvm-objdump.exe'),'-d','--x86-asm-syntax=intel','--start-address='+hex(addr),'--stop-address='+hex(addr+f['size']),str(GAME/'DoorKickers2.exe')],text=True)
        (OUT/(safe+'.asm.txt')).write_text(asm,encoding='utf-8'); rows.append(f'{addr:x}\t{name}')
    (OUT/'decompile_targets.tsv').write_text('\n'.join(rows),encoding='utf-8')
    print(json.dumps({'functions':len(funcs),'symbols':len(symbols),'types':len(lookup),'profile':str(ROOT/'config/build.json')},indent=2))

if __name__=='__main__': main()
