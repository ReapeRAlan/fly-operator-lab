"""Hold a process-scoped Windows system/display wake request for an unattended run."""
from pathlib import Path
import argparse,ctypes,json,time

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'work/learning/status.json'
kernel=ctypes.WinDLL('kernel32')
kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]
kernel.SetThreadExecutionState.restype=ctypes.c_uint

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=24);args=parser.parse_args()
    deadline=time.time()+max(0.1,args.hours)*3600
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    if not kernel.SetThreadExecutionState(0x80000003):raise ctypes.WinError()
    try:
        while time.time()<deadline:
            try:status=json.loads(STATUS.read_text(encoding='utf-8-sig'))
            except (FileNotFoundError,json.JSONDecodeError,PermissionError):status={}
            if status.get('state') in ('stopped','error','needs_calibration','curriculum_complete'):break
            time.sleep(15)
    finally:
        kernel.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
