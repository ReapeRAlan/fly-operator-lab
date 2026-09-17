"""Read-only Win32 verification of the configured laboratory display policy."""
from pathlib import Path
import ctypes as C
from ctypes import wintypes as W
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from runtime_config import load_runtime_config
u=C.WinDLL('user32',use_last_error=True)
class MI(C.Structure):
    _fields_=[('size',W.DWORD),('monitor',W.RECT),('work',W.RECT),('flags',W.DWORD),('device',W.WCHAR*32)]
u.GetWindowThreadProcessId.argtypes=[W.HWND,C.POINTER(W.DWORD)]
u.GetWindowRect.argtypes=[W.HWND,C.POINTER(W.RECT)]
u.MonitorFromWindow.argtypes=[W.HWND,W.DWORD];u.MonitorFromWindow.restype=W.HANDLE
u.GetMonitorInfoW.argtypes=[W.HANDLE,C.POINTER(MI)]
u.IsWindowVisible.argtypes=[W.HWND]
CALLBACK=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
def main():
    runtime=load_runtime_config()
    session=json.loads((ROOT/'work/session.json').read_text());found=[]
    @CALLBACK
    def enum(hwnd,param):
        pid=W.DWORD();u.GetWindowThreadProcessId(hwnd,C.byref(pid))
        if pid.value==session['pid'] and u.IsWindowVisible(hwnd):found.append(hwnd)
        return True
    u.EnumWindows(enum,0)
    results=[]
    for hwnd in found:
        rect=W.RECT();u.GetWindowRect(hwnd,C.byref(rect));mi=MI();mi.size=C.sizeof(mi)
        if not u.GetMonitorInfoW(u.MonitorFromWindow(hwnd,2),C.byref(mi)):raise C.WinError(C.get_last_error())
        results.append({'window':int(hwnd),'rect':[rect.left,rect.top,rect.right,rect.bottom],'monitor':mi.device,'primary':bool(mi.flags&1),'inside_monitor':rect.left>=mi.monitor.left and rect.top>=mi.monitor.top and rect.right<=mi.monitor.right and rect.bottom<=mi.monitor.bottom})
    inside=bool(results) and all(r['inside_monitor'] for r in results)
    placement=inside and (not runtime['require_secondary_monitor'] or all(not r['primary'] for r in results))
    result={'pid':session['pid'],'display_mode':runtime['display_mode'],'require_secondary_monitor':runtime['require_secondary_monitor'],
      'windows':results,'passed':placement}
    (ROOT/'outputs/display_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert result['passed']
if __name__=='__main__':main()
