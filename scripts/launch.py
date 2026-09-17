"""Launch the hash-pinned local game suspended, install the bridge, then resume.

The bridge redirects the game's Windows LocalAppData query before game startup.
No game executable or original save file is patched.
"""
from pathlib import Path
import ctypes as C
from ctypes import wintypes as W
import hashlib,json,time,sys
ROOT=Path(__file__).resolve().parents[1]
k=C.WinDLL('kernel32',use_last_error=True)
class STARTUPINFO(C.Structure):
    _fields_=[('cb',W.DWORD),('reserved',W.LPWSTR),('desktop',W.LPWSTR),('title',W.LPWSTR),('x',W.DWORD),('y',W.DWORD),('xs',W.DWORD),('ys',W.DWORD),('xc',W.DWORD),('yc',W.DWORD),('fill',W.DWORD),('flags',W.DWORD),('show',W.WORD),('reserved2',W.WORD),('reservedp',C.c_void_p),('stdin',W.HANDLE),('stdout',W.HANDLE),('stderr',W.HANDLE)]
class PROCESSINFO(C.Structure):
    _fields_=[('process',W.HANDLE),('thread',W.HANDLE),('pid',W.DWORD),('tid',W.DWORD)]
def api(name,restype,args):
    f=getattr(k,name);f.restype=restype;f.argtypes=args;return f
create=api('CreateProcessW',W.BOOL,[W.LPCWSTR,W.LPWSTR,C.c_void_p,C.c_void_p,W.BOOL,W.DWORD,C.c_void_p,W.LPCWSTR,C.POINTER(STARTUPINFO),C.POINTER(PROCESSINFO)])
event=api('CreateEventW',W.HANDLE,[C.c_void_p,W.BOOL,W.BOOL,W.LPCWSTR])
alloc=api('VirtualAllocEx',C.c_void_p,[W.HANDLE,C.c_void_p,C.c_size_t,W.DWORD,W.DWORD])
write=api('WriteProcessMemory',W.BOOL,[W.HANDLE,C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)])
module=api('GetModuleHandleW',W.HMODULE,[W.LPCWSTR]);proc=api('GetProcAddress',C.c_void_p,[W.HMODULE,C.c_char_p])
remote=api('CreateRemoteThread',W.HANDLE,[W.HANDLE,C.c_void_p,C.c_size_t,C.c_void_p,C.c_void_p,W.DWORD,C.c_void_p])
wait=api('WaitForSingleObject',W.DWORD,[W.HANDLE,W.DWORD]);resume=api('ResumeThread',W.DWORD,[W.HANDLE])
close=api('CloseHandle',W.BOOL,[W.HANDLE]);terminate=api('TerminateProcess',W.BOOL,[W.HANDLE,W.UINT])
free=api('VirtualFreeEx',W.BOOL,[W.HANDLE,C.c_void_p,C.c_size_t,W.DWORD])
def checked(value):
    if not value:raise C.WinError(C.get_last_error())
    return value
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def launch():
    sys.path.insert(0,str(ROOT/'src'))
    from windows_session import secondary_monitors
    from runtime_config import load_runtime_config
    runtime=load_runtime_config()
    if runtime['require_secondary_monitor'] and not secondary_monitors():raise RuntimeError('Conecta el monitor secundario antes de abrir Fly Operator')
    # The isolated profile may have retained a borderless mode from manual play.
    import xml.etree.ElementTree as ET
    options=ROOT/'work/profile/KillHouseGames/DoorKickers2/options.xml'
    if options.exists():
        tree=ET.parse(options);video=tree.getroot().find('Video')
        video.set('mode','0');video.set('width','1280');video.set('height','720');video.set('maxFPS','20')
        tree.write(options,encoding='utf-8',xml_declaration=True)
    profile=json.loads((ROOT/'config/build.json').read_text());game=Path(profile['game_path']);exe=game/'DoorKickers2.exe';dll=ROOT/'work/build/flybridge.dll'
    if digest(exe)!=profile['sha256'] or digest(game/'DoorKickers2.pdb')!=profile['pdb_sha256']:raise RuntimeError('Executable/PDB pair changed; repeat reverse engineering before use')
    si=STARTUPINFO();si.cb=C.sizeof(si);pi=PROCESSINFO();ready=None;loader=None;buffer=None
    checked(create(str(exe),C.create_unicode_buffer('"'+str(exe)+'"'),None,None,False,4,None,str(game),C.byref(si),C.byref(pi)))
    try:
        ready=checked(event(None,True,False,'Local\\FlyBridgeReady_'+str(pi.pid)))
        raw=(str(dll.resolve())+'\0').encode('utf-16-le');buffer=checked(alloc(pi.process,None,len(raw),0x3000,4));written=C.c_size_t()
        checked(write(pi.process,buffer,raw,len(raw),C.byref(written)))
        loader=checked(remote(pi.process,None,0,proc(module('kernel32.dll'),b'LoadLibraryW'),buffer,0,None))
        if wait(loader,30000)!=0:raise RuntimeError('DLL loader did not finish')
        if wait(ready,30000)!=0:raise RuntimeError('Bridge did not signal ready; inspect work/bridge.log')
        if resume(pi.thread)==0xffffffff:raise C.WinError(C.get_last_error())
        result={'pid':pi.pid,'pipe':'\\\\.\\pipe\\FlyOperator_'+str(pi.pid),'exe_sha256':profile['sha256'],'pdb_guid':profile['guid'],'pdb_age':profile['age'],'profile':str(ROOT/'work/profile'),'started_unix':time.time(),'dll_sha256':digest(dll),'display_mode':runtime['display_mode']}
        (ROOT/'work/session.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        return result
    except BaseException:
        terminate(pi.process,1);raise
    finally:
        if buffer:free(pi.process,buffer,0,0x8000)
        for h in [loader,ready,pi.thread,pi.process]:
            if h:close(h)
if __name__=='__main__':launch()
