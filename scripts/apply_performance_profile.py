"""One audited resource-only transition, preserving the active pilot deadline."""
from pathlib import Path
import json,sys,time,subprocess,urllib.request,psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lab_store import atomic_json,file_hash

runtime=ROOT/'work/learning'
audit=json.loads((ROOT/'work/resource_profile_change.json').read_text())
original=audit['original_status']
deadline=original['updated']-original['wall_seconds']+original['pilot_hours']*3600

def state():return json.loads((runtime/'status.json').read_text())

def until(test,seconds=100):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        current=state()
        if current.get('state')=='error':raise RuntimeError(current.get('reason'))
        if test(current):return current
        time.sleep(.5)
    raise TimeoutError('Trainer transition timeout')

before=state();old_pid=before['pid']
assert before['state'] in ('running','paused') and before['phase']=='internal','Apply at internal phase; no partial PPO buffer'
assert psutil.sensors_battery().power_plugged,'Laptop must remain on AC'
request=urllib.request.Request('http://127.0.0.1:8766/api/control/stop',method='POST')
with urllib.request.urlopen(request,timeout=5) as response:response.read()
stopped=until(lambda s:s.get('pid')==old_pid and s.get('state')=='stopped')
try:psutil.Process(old_pid).wait(timeout=15)
except psutil.NoSuchProcess:pass
point=Path(stopped['checkpoint']);manifest=json.loads(point.read_text());directory=point.parent/str(manifest['generation'])
for name,key in [('brain.npz','brain_sha256'),('policy.pt','policy_sha256')]:
    assert file_hash(directory/name)==manifest[key],name+' checkpoint mismatch'
cfg=json.loads((ROOT/'config/learning.json').read_text())
backup=ROOT/'work/learning_config_before_performance.json'
if not backup.exists():atomic_json(backup,cfg)
changes={'max_cpu_threads':2,'worker_cpu_affinity':[2,4,6,8],
         'performance_profile':'performance_cpu_v1','worker_priority':'below_normal',
         'minimum_rest_after_step_seconds':.05,'minimum_available_ram_gb':2}
cfg.update(changes);atomic_json(ROOT/'config/learning.json',cfg)
remaining=(deadline-time.time())/3600
assert remaining>0,'Original pilot deadline already elapsed; do not restart'
audit.update(checkpoint=manifest,checkpoint_path=str(point),original_deadline=deadline,requested_remaining_hours=remaining,resource_changes=changes)
atomic_json(ROOT/'work/resource_profile_change.json',audit)
with (ROOT/'work/performance_launch.log').open('ab') as log:
    subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/start_learning.ps1'),'-Hours',str(remaining)],
                   stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=30,check=True)
restored=until(lambda s:s.get('pid')!=old_pid and s.get('state')=='running' and s.get('steps',-1)>manifest['counters']['steps'],seconds=150)
assert restored['restored_steps']==manifest['counters']['steps']
assert restored['restored_actor_sha256']==manifest['actor_sha256']
assert restored['resumed_from']==str(point)
assert restored['cpu_affinity']==changes['worker_cpu_affinity']
assert abs(restored['pilot_deadline']-deadline)<30
audit.update(passed=True,restored_status=restored,game_session=json.loads((ROOT/'work/session.json').read_text()),
             note='Learned efficacy, actor and optimizer restored. Interrupted game episode restarts by checkpoint contract; no change to neural dynamics or learning parameters.')
atomic_json(ROOT/'outputs/performance_profile_validation.json',audit)
print(json.dumps({'passed':True,'pid':restored['pid'],'saved_steps':manifest['counters']['steps'],'steps':restored['steps'],
                  'affinity':restored['cpu_affinity'],'rest_seconds':restored['rest_after_step_seconds'],'pilot_deadline':restored['pilot_deadline']}),flush=True)
