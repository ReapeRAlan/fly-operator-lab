"""Read-only integrity and resource summary for the overnight readiness review."""
from pathlib import Path
import json,sqlite3,hashlib,time,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lab_store import atomic_json
def read(path):
    try:return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,ValueError):return {}
def main():
    cfg=read('config/learning.json');db=sqlite3.connect(ROOT/'work/learning/experiments.sqlite');run=cfg['experiment_id']
    episodes=db.execute('select condition,seed,count(*),sum(success) from episodes where run=? group by condition,seed',(run,)).fetchall()
    steps=db.execute('select count(*) from steps where run=?',(run,)).fetchone()[0]
    duplicate_count=db.execute("select count(*) from (select json_extract(payload,'$.game_session'),episode,sequence,count(*) as n from steps where run=? group by json_extract(payload,'$.game_session'),episode,sequence having n>1)",(run,)).fetchone()[0]
    result={'generated_unix':time.time(),'integrity':db.execute('pragma quick_check').fetchone()[0],'episodes':episodes,'steps':steps,'duplicate_step_keys_within_game_session':duplicate_count,'config':cfg,'checkpoints':[]}
    db.close()
    for p in sorted((ROOT/'work/learning/checkpoints').glob('common_*/current.json')):
        m=json.loads(p.read_text());d=p.parent/str(m['generation']);matches={}
        files=[('brain.npz','brain_sha256'),('policy.pt','policy_sha256')]
        if m.get('examples_sha256'):files.append(('examples.npz','examples_sha256'))
        for file,key in files:
            h=hashlib.sha256()
            with (d/file).open('rb') as f:
                for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
            matches[file]=h.hexdigest()==m[key]
        result['checkpoints'].append({'path':str(p),'hashes_match':matches,'counters':m['counters'],'actor_hash':m.get('actor_sha256')})
    result['performance']={name:read('outputs/performance_'+name+'.json').get('summary') for name in ('original_active','laboratory_closed','quiet_active','quiet_warm')}
    result['cpu_temperature']={'available':False,'reason':'Windows denied AWCC WMI instance access (0x80041003); no ACPI temperature instances exposed'}
    result['cpu_temperature_limit_reference']='https://www.intel.la/content/www/xl/es/products/sku/232101/intel-core-i713650hx-processor-24m-cache-up-to-4-90-ghz/specifications.html'
    result['sensor_interface_reference']='https://kernel.org/doc/html/next/wmi/devices/alienware-wmi.html'
    result['status']=read('work/learning/status.json');atomic_json(ROOT/'outputs/night_audit.json',result)
    print(json.dumps({k:result[k] for k in ('integrity','episodes','steps','duplicate_step_keys_within_game_session','performance')}),flush=True)
if __name__=='__main__':main()
