"""One save/stop/restart audit of the real trainer; leaves the authorized pilot running."""
from pathlib import Path
import json,time,urllib.request,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
url='http://127.0.0.1:8766'
def state():
    with urllib.request.urlopen(url+'/api/state',timeout=10) as r:return json.load(r)
def until(test):
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        current=state()
        if current['status'].get('state')=='error':raise RuntimeError(current['status'].get('reason'))
        if test(current):return current
        time.sleep(.5)
    raise TimeoutError('Trainer transition timeout')
def start():
    # Use a file, not PIPE: descendants can inherit a pipe and keep communicate()
    # waiting after PowerShell exits, even though the trainer is already running.
    with (ROOT/'work/checkpoint_launch_audit.log').open('ab') as output:
        subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/start_learning.ps1'),'-Hours','24'],check=True,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=30)
before=state()['status']
if before['state']=='stopped':
    start()
    first=until(lambda s:s['status'].get('pid')!=before['pid'] and s['status'].get('state')=='running' and s['status'].get('steps',0)>=s['status'].get('restored_steps',10**9)+2)
else:
    assert before['state'] in ('running','saving'),'Resume trainer before this audit'
    first=until(lambda s:s['status'].get('state')=='running' and s['status'].get('steps',0)>=before.get('steps',0)+2)
with urllib.request.urlopen(urllib.request.Request(url+'/api/control/stop',method='POST'),timeout=10) as r:json.load(r)
stopped=until(lambda s:s['status'].get('pid')==first['status']['pid'] and s['status'].get('state')=='stopped')
point=Path(stopped['status']['checkpoint']);manifest=json.loads(point.read_text());examples=point.parent/str(manifest['generation'])/'examples.npz'
assert manifest.get('examples_sha256')==hashlib.sha256(examples.read_bytes()).hexdigest()
time.sleep(1)
start()
restored=until(lambda s:s['status'].get('pid')!=stopped['status']['pid'] and s['status'].get('state')=='running' and s['status'].get('steps',0)>manifest['counters']['steps'])
status=restored['status']
assert status['restored_steps']==manifest['counters']['steps']
assert status['restored_actor_sha256']==manifest['actor_sha256']
assert status['restored_examples_count']==manifest['examples_count']
result={'passed':True,'checkpoint':str(point),'actor_sha256':manifest['actor_sha256'],'steps_saved':manifest['counters']['steps'],'steps_after_restart':status['steps'],
 'examples_restored':status['restored_examples_count'],'examples_sha256':manifest['examples_sha256'],'pid':status['pid'],'game_session':restored['latest'].get('game_session'),
 'scope':'Real checkpoint and teaching examples restored together; pilot left running for up to 24 hours'}
(ROOT/'outputs/checkpoint_io_live_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
