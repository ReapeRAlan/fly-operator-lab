"""One-time, audited migration from artifact-byte hashes to capability contracts.

This never changes brain, policy, optimizer, examples or counters. It is
deliberately pinned to the one checkpoint and artifact set audited on
2026-09-16; it is not a general bypass for semantic compatibility checks.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from lab_store import Store,atomic_json,file_hash
from learning_brain import LearningBrain
from learning_env import FlyOperatorEnv
from learning_policy import make_model,actor_hash
from learning_checkpoint import (SEMANTIC_HASH_VERSION,_assert_finite,capability_contract,
  semantic_config,semantic_hash)
from learning_runtime import WorkerLock

EXPECTED_ACTIVE={
  'generation':0,'condition':'common_teaching','seed':7,'run':'pilot-v3.1-curriculum',
  'brain_sha256':'0e0f880e794c6695bc251486de77d76e53a9210ffe16eecc002e19b3813189a2',
  'policy_sha256':'7ff4c875b0bf45ad8a34d4477019e8ac7358f99013eb3fa06d75e5d0faa47e10',
  'examples_sha256':'0d4202ea98b77ca47bbb5415e32de10d4581ac4b4cedf81587531b94b65e1423',
  'actor_sha256':'e76db0b0f7be699ab00dd8eed2d6321cf1c9b6c2b36fa5449bd81c12ffcf7832',
  'graph_hash':'7f77d769d2fcba02f0581da8f9b3e913dd7d7d2abf067ddc71686a5b7545b161',
}
EXPECTED_SCENARIOS='fa9a1c289d1ea343973ff2f6fef8d43a3dcdaff19b01ea32f088882dab34a7db'
EXPECTED_EQUIPMENT='8dff11def49138321be87a82dbde854bfa00e5dcf016d87f288a8bde160a1673'
EXPECTED_CAPABILITY_CONTRACT='bdcb9a708e7d5dd42065c87bf8797970908050714bc742c03094c1ee65affc56'


class OfflineBridge:
    def __getattr__(self,name):raise RuntimeError(f'Offline semantic migration attempted game I/O: {name}')


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_artifact_set():
    contract=capability_contract()
    if file_hash(ROOT/'data/learning/scenarios.json')!=EXPECTED_SCENARIOS:raise ValueError('Scenario artifact differs from audited migration input')
    if file_hash(ROOT/'data/learning/equipment_catalog.json')!=EXPECTED_EQUIPMENT:raise ValueError('Equipment artifact differs from audited migration input')
    if contract_hash(contract)!=EXPECTED_CAPABILITY_CONTRACT:raise ValueError('Capability meaning differs from the authorized legacy contract')
    return contract


def verify_generation(path,expected_semantic,semantic_version,current_config,brain,model):
    manifest=json.loads(path.read_text(encoding='utf-8-sig'));directory=path.parent
    if manifest.get('semantic_hash_version',1)!=semantic_version or manifest.get('semantic_sha256')!=expected_semantic:
        raise ValueError(f'Unexpected semantic manifest: {path}')
    for name,key in [('brain.npz','brain_sha256'),('policy.pt','policy_sha256'),('examples.npz','examples_sha256')]:
        if key in manifest and file_hash(directory/name)!=manifest[key]:raise ValueError(f'Integrity mismatch: {directory/name}')
    state=torch.load(directory/'policy.pt',map_location='cpu',weights_only=False)
    if semantic_config(state.get('config',{}))!=semantic_config(current_config):raise ValueError(f'Saved learning configuration differs: {path}')
    _assert_finite(state['policy'],'policy');_assert_finite(state['optimizer'],'optimizer')
    model.policy.load_state_dict(state['policy'])
    if manifest.get('actor_sha256')!=actor_hash(model):raise ValueError(f'Actor hash mismatch: {path}')
    brain.load(directory/'brain.npz')
    if manifest.get('graph_hash')!=brain.graph_hash:raise ValueError(f'Graph hash mismatch: {path}')
    return manifest


def enrich_existing_record(record_path,current):
    record=json.loads(record_path.read_text(encoding='utf-8-sig'))
    if 'legacy_manifest' in record:return record
    legacy=copy.deepcopy(current);legacy.pop('semantic_hash_version',None);legacy.pop('semantic_migration',None);legacy['semantic_sha256']=record['legacy_semantic_sha256']
    raw=json.dumps(legacy,ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest()!=record['pointer_sha256_before']:raise ValueError('Could not reconstruct exact legacy pointer')
    record['legacy_manifest']=legacy;record['legacy_manifest_sha256']=record['pointer_sha256_before']
    record['authorized_capability_contract_sha256']=EXPECTED_CAPABILITY_CONTRACT
    record['scenarios_sha256']=EXPECTED_SCENARIOS;record['equipment_sha256']=EXPECTED_EQUIPMENT
    atomic_json(record_path,record);return record


def migrate(args):
    pointer=Path(args.pointer).resolve();cfg=json.loads((ROOT/'config/learning.json').read_text(encoding='utf-8-sig'))
    current=json.loads(pointer.read_text(encoding='utf-8-sig'));root=pointer.parent
    active=root/str(current['generation'])/'manifest.json';active_manifest=json.loads(active.read_text(encoding='utf-8-sig'))
    if current!=active_manifest:raise ValueError('Pointer and active generation manifest differ')
    contract=verify_artifact_set()
    brain=LearningBrain();env=FlyOperatorEnv(brain,bridge=OfflineBridge(),condition=current['condition'],seed=int(current['seed']));env.config=cfg
    model=make_model(env,cfg,int(current['seed']));new_semantic=semantic_hash(env,model)
    record_path=ROOT/'outputs/checkpoint_semantic_migration_v2.json'
    if current.get('semantic_hash_version')==SEMANTIC_HASH_VERSION:
        if current.get('semantic_sha256')!=new_semantic:raise ValueError('Migrated checkpoint semantic hash no longer matches')
        verify_generation(active,new_semantic,SEMANTIC_HASH_VERSION,cfg,brain,model);enrich_existing_record(record_path,current)
        print(json.dumps({'already_migrated':True,'verified':True,'pointer':str(pointer),'semantic_sha256':new_semantic},indent=2));return
    if current.get('semantic_sha256')!=args.expected_old_semantic:raise ValueError('Pointer does not match the authorized legacy semantic hash')
    for key,value in EXPECTED_ACTIVE.items():
        if current.get(key)!=value:raise ValueError(f'Active checkpoint differs from audited migration input: {key}')
    legacy_manifest=copy.deepcopy(current);manifest=verify_generation(active,args.expected_old_semantic,1,cfg,brain,model)
    if manifest!=current:raise ValueError('Legacy pointer and generation changed during verification')
    manifest['semantic_hash_version']=SEMANTIC_HASH_VERSION;manifest['semantic_sha256']=new_semantic
    manifest['semantic_migration']='outputs/checkpoint_semantic_migration_v2.json'
    pointer_before=file_hash(pointer);atomic_json(active,manifest);atomic_json(pointer,manifest);pointer_after=file_hash(pointer)
    record={'version':2,'passed':True,'created':time.time(),'pointer':str(pointer),'pointer_sha256_before':pointer_before,
      'pointer_sha256_after':pointer_after,'legacy_semantic_sha256':args.expected_old_semantic,'legacy_manifest':legacy_manifest,
      'legacy_manifest_sha256':pointer_before,'semantic_sha256':new_semantic,'migrated_generations':[int(current['generation'])],
      'reason':args.reason,'unchanged_payload_hashes':{key:manifest.get(key) for key in ('brain_sha256','policy_sha256','examples_sha256','actor_sha256','graph_hash')},
      'authorized_capability_contract_sha256':EXPECTED_CAPABILITY_CONTRACT,'capability_contract':contract,
      'scenarios_sha256':EXPECTED_SCENARIOS,'equipment_sha256':EXPECTED_EQUIPMENT,
      'native_evidence_sha256':file_hash(ROOT/'outputs/native_capabilities_v2.json'),
      'teacher_validation_sha256':file_hash(ROOT/'outputs/teacher_validation_v3.json')}
    atomic_json(record_path,record)
    store=Store(cfg['experiment_id'])
    try:store.checkpoint(manifest['condition'],manifest['seed'],pointer,pointer_after,{**manifest,'migration_record':str(record_path)})
    finally:store.close()
    print(json.dumps(record,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pointer',required=True);parser.add_argument('--expected-old-semantic',required=True);parser.add_argument('--reason',required=True)
    with WorkerLock():migrate(parser.parse_args())


if __name__=='__main__':main()
