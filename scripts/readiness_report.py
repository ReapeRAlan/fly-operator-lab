"""Current readiness evidence; no fabricated training or validation results."""
from pathlib import Path
import json,time,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lab_store import atomic_json,file_hash
def read(path):
    p=ROOT/path
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
def main():
    cfg=read('config/learning.json');teacher=read('outputs/teacher_validation_v2.json');kits=read('outputs/native_kits_v2.json')
    status=read('work/learning/status.json');session=read('work/session.json');sources={}
    for folder in ('src','scripts','config','native','dashboard/src'):
        for p in sorted((ROOT/folder).glob('*')):
            if p.is_file() and p.suffix in ('.py','.json','.h','.cpp','.ps1','.jsx','.css'):sources[str(p.relative_to(ROOT))]=file_hash(p)
    rows=teacher.get('rows',[])
    result={'generated_unix':time.time(),'experiment_id':cfg['experiment_id'],'status':status,'session':session,
      'teacher':{'passed':sum(bool(r['success']) for r in rows),'total':len(rows),'scope':teacher.get('scope')},
      'equipment':{'passed':sum(bool(r['passed']) for r in kits),'total':len(kits)},
      'native_stability':read('outputs/stability_v2.json'),'display':read('outputs/display_validation.json'),
      'dashboard_qa':read('work/design/qa.json'),'perception_qa':read('outputs/perception_qa.json'),'planning_validation':read('outputs/planning_validation.json'),
      'night_audit':read('outputs/night_audit.json'),'pause_resume_qa':read('work/design/qa_continuity.json'),'sources_sha256':sources,
      'checkpoint_io_benchmark':read('outputs/checkpoint_io_benchmark.json'),'checkpoint_io_live_validation':read('outputs/checkpoint_io_live_validation.json'),
      'scenario_manifest_sha256':file_hash(ROOT/'data/learning/scenarios.json'),
      'development_checkpoint_archive':'work/archive_development_20260915',
      'windows_snapshot_fix':'Unique temporary files and bounded retry on PermissionError; two concurrent-access tests passed',
      'heldout_learning_acceptance_met':False,'remaining_actions':read('outputs/native_capabilities_v2.json').get('pending',[])}
    atomic_json(ROOT/'outputs/readiness_v2.json',result)
    print(json.dumps({k:result[k] for k in ('experiment_id','teacher','equipment','heldout_learning_acceptance_met')}))
if __name__=='__main__':main()
