"""Length-prefixed JSON transport over the local Windows named pipe."""
from pathlib import Path
import json,struct,time
ROOT=Path(__file__).resolve().parents[1]
class Bridge:
    def __init__(self,session=None):
        self.session=session or json.loads((ROOT/'work/session.json').read_text())
        self.file=open(self.session['pipe'],'r+b',buffering=0)
        self.last=None
    def exact(self,n):
        out=b''
        while len(out)<n:
            chunk=self.file.read(n-len(out))
            if not chunk:raise EOFError('Game bridge disconnected')
            out+=chunk
        return out
    def request(self,op,**kwargs):
        raw=json.dumps({'op':op,**kwargs},allow_nan=False).encode()
        if len(raw)>65536:raise ValueError('Oversize request')
        packet=struct.pack('<I',len(raw))+raw
        while packet:
            n=self.file.write(packet)
            if not n:raise EOFError('Write failed')
            packet=packet[n:]
        n=struct.unpack('<I',self.exact(4))[0]
        if n>1048576:raise ValueError('Oversize response')
        self.last=json.loads(self.exact(n));return self.last
    def step(self,command,dt_ms=33):
        if self.last is None: self.request('observe')
        out=self.request('step',episode=self.last['episode'],sequence=self.last['sequence']+1,dt_ms=dt_ms,command=command)
        if not out['ok']:raise RuntimeError(out)
        if 'capture_frame' in out:
            directory=Path(out['capture_directory']); index=out['capture_frame'];deadline=time.monotonic()+15
            path=directory/'timestamps.jsonl'
            while time.monotonic()<deadline:
                if path.exists():
                    rows=path.read_text().splitlines()
                    if len(rows)>index:
                        frame=json.loads(rows[index])
                        if not frame['saved'] or frame['sim_time_ms']!=out['sim_time_ms']:raise RuntimeError('Invalid capture timestamp')
                        out['frame']=frame;break
                time.sleep(.005)
            else:raise TimeoutError('Renderer did not finish frame capture')
        return out

    def new_episode(self,record=False):
        old=self.request('observe').get('map_load_id')
        self.request('load_validation')
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            time.sleep(.05);o=self.request('observe')
            if o.get('map_load_id')!=old and o.get('client0_state')==5 and len(o['humans'])==2 and o['game_flags']&512:break
        else:raise TimeoutError('Validation mission did not load')
        out=self.deploy()
        # Let the normal client deployment notification and network packets settle.
        time.sleep(.1)
        out=self.request('start',record=record)
        if not out['ok'] or out['operator']['ammo']<1:raise RuntimeError('Stock operator unavailable')
        return out
    def close(self):self.file.close()
    def wait_for_startup(self,timeout=180.,poll=.5):
        """Block until this game session finished loading its UI.

        Client map indices appear before all startup resources finish loading, and
        Game::RestartMap during that window crashes the game. A fixed 10 s guard was
        measured with a warm cache; enumerating 2,017 laboratory maps cold took 12 s
        and the UI loaded later still. The game log marks the end of UI loading.
        """
        if getattr(self,'_startup_ready',False):return
        started=float(self.session.get('started_unix',0))
        log=Path(self.session.get('profile',ROOT/'work/profile'))/'KillHouseGames/DoorKickers2/log.txt'
        deadline=time.monotonic()+timeout
        while True:
            try:
                # The previous session's log is renamed at startup; require this session's file.
                if log.stat().st_mtime>=started-5 and b'Loading UI took' in log.read_bytes():break
            except OSError:pass
            if time.monotonic()>=deadline:raise TimeoutError('Game UI did not finish loading: '+str(log))
            time.sleep(poll)
        time.sleep(max(2.,started+10-time.time()))
        self._startup_ready=True
    def new_mission(self,name,record=False):
        self.wait_for_startup()
        old=self.request('observe').get('map_load_id')
        result=self.request('load_mission',mission=name)
        if not result.get('ok'):raise RuntimeError(result)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            time.sleep(.05);o=self.request('observe')
            if o.get('map_load_id')!=old and o.get('client0_state')==5 and o['game_flags']&512 and any(h['kind']==0 for h in o['humans']):break
        else:raise TimeoutError('Laboratory mission did not load: '+name)
        result=self.deploy()
        time.sleep(.1)
        result=self.request('start',record=record)
        if not result.get('ok') or not result.get('operator'):raise RuntimeError('Operator unavailable')
        return result
    def deploy(self):
        result=self.request('deploy');deadline=time.monotonic()+10
        while result.get('ok') and time.monotonic()<deadline:
            view=result.get('client_view',{})
            if not result.get('game_flags',0)&512 and not view.get('game_flags',0)&512 and not result.get('deployment_pending',False):return result
            time.sleep(.05);result=self.request('observe')
        raise RuntimeError('Deployment did not synchronize client and server: '+str(result.get('client_view',result)))
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
if __name__=='__main__':
    import sys
    with Bridge() as b: print(json.dumps(b.request(sys.argv[1] if len(sys.argv)>1 else 'observe'),indent=2))
