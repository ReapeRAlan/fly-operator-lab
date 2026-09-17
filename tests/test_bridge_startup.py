import sys,time,os
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import bridge_client
from bridge_client import Bridge

def bridge(tmp_path,started):
    b=Bridge.__new__(Bridge);b.session={'profile':str(tmp_path),'started_unix':started};return b

def log(tmp_path,text,mtime=None):
    path=tmp_path/'KillHouseGames/DoorKickers2/log.txt';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    if mtime is not None:os.utime(path,(mtime,mtime))
    return path

def test_startup_waits_for_ui_marker_of_this_session(tmp_path,monkeypatch):
    sleeps=[];monkeypatch.setattr(bridge_client.time,'sleep',lambda s:sleeps.append(s))
    now=time.time();log(tmp_path,'CL 016:631 - Loading maps took 12126\nCL 017:900 - Loading UI took 16 + 476 msec\n')
    b=bridge(tmp_path,now-60);b.wait_for_startup(timeout=1,poll=0);assert b._startup_ready and sleeps==[2.]
    b.wait_for_startup(timeout=0);assert sleeps==[2.]

def test_startup_rejects_ui_still_loading_or_previous_session_log(tmp_path,monkeypatch):
    monkeypatch.setattr(bridge_client.time,'sleep',lambda s:None)
    now=time.time();log(tmp_path,'CL 016:631 - Loading maps took 12126\n')
    with pytest.raises(TimeoutError):bridge(tmp_path,now).wait_for_startup(timeout=0,poll=0)
    log(tmp_path,'Loading UI took 1 msec\n',mtime=now-3600)
    with pytest.raises(TimeoutError):bridge(tmp_path,now).wait_for_startup(timeout=0,poll=0)
