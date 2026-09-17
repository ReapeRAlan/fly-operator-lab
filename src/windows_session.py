"""Read-only display inventory and process-scoped Windows sleep prevention."""
import ctypes as C
from ctypes import wintypes as W

def secondary_monitors():
    user=C.WinDLL('user32',use_last_error=True)
    class MonitorInfo(C.Structure):
        _fields_=[('size',W.DWORD),('monitor',W.RECT),('work',W.RECT),('flags',W.DWORD)]
    callback=C.WINFUNCTYPE(W.BOOL,W.HANDLE,W.HDC,C.POINTER(W.RECT),W.LPARAM)
    user.GetMonitorInfoW.argtypes=[W.HANDLE,C.POINTER(MonitorInfo)]
    found=[]
    @callback
    def visit(handle,dc,rect,param):
        info=MonitorInfo();info.size=C.sizeof(info)
        if user.GetMonitorInfoW(handle,C.byref(info)) and not info.flags&1:
            found.append([info.work.left,info.work.top,info.work.right,info.work.bottom])
        return True
    user.EnumDisplayMonitors(None,None,visit,0)
    return found

def keep_system_awake(enabled):
    # No global power-plan changes; this request disappears when the worker exits.
    # Keep the display path alive too: some HDMI/USB-C monitors disappear from
    # EnumDisplayMonitors when Windows powers them down, moving the game to the
    # primary screen and invalidating an overnight secondary-monitor run.
    flags=0x80000000 | (0x1|0x2 if enabled else 0)
    C.WinDLL('kernel32').SetThreadExecutionState(W.DWORD(flags))
