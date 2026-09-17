"""Atomic two-generation checkpoints. Restore learning, then restart interrupted game episode."""
from pathlib import Path
import json,os,random,time
import numpy as np,torch
from lab_store import ROOT,RUNTIME,atomic_json,file_hash
from learning_policy import actor_hash
from lossless_archive import atomic_npz

SEMANTIC_HASH_VERSION=2
SEMANTIC_KEYS=('version','semantic_protocol_version','dataset','classified_neurons','directed_edges','decision_ms','filters_ms',
  'ports_per_channel','sensory_baseline_hz','sensory_gain_hz','ppo','plasticity_rates','guided_episodes_per_stage',
  'dagger_episodes_per_stage','bootstrap_evaluation_episodes','max_extra_dagger_episodes','dagger_teacher_beta',
  'stage_transfer_imitation_epochs','promotion_evaluation_episodes','acceptance_evaluation_episodes_per_seed','imitation','motor_plasticity_learning_rate',
  'imitation_epochs','promotion_success','retention_fraction','evaluation_interval_steps','policy_collapse_free_wait_steps',
  'policy_no_progress_decisions','free_wait_penalty','plasticity_reward_gain','plasticity_rule','stage_order',
  'ppo_v33','tracks','condition_settings','promotion','control_imitation_epochs')


def _assert_finite(value,label):
    if torch.is_tensor(value):
        if value.is_floating_point() and not torch.isfinite(value).all().item():raise FloatingPointError(f'Non-finite checkpoint tensor: {label}')
    elif isinstance(value,np.ndarray):
        if np.issubdtype(value.dtype,np.floating):
            flat=value.reshape(-1)
            for start in range(0,len(flat),1_000_000):
                if not np.isfinite(flat[start:start+1_000_000]).all():raise FloatingPointError(f'Non-finite checkpoint array: {label}')
    elif isinstance(value,dict):
        for key,item in value.items():_assert_finite(item,f'{label}.{key}')
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):_assert_finite(item,f'{label}[{index}]')


def semantic_config(config):
    return {key:config[key] for key in SEMANTIC_KEYS if key in config}


def capability_contract(path=None):
    """Return capability meaning without build/session provenance.

    Recompiling an equivalent native bridge changes DLL, PID and evidence-file
    hashes, but does not change which actions the learner can issue.  The
    contract still changes if an action, validation check, stage or kit changes.
    """
    target=Path(path) if path is not None else ROOT/'outputs/native_capabilities_v2.json'
    if not target.exists():return None
    raw=json.loads(target.read_text(encoding='utf-8-sig'))
    required={'version','validated_actions','pending','evidence'}
    if not required.issubset(raw):raise ValueError(f'Capability contract missing fields: {sorted(required-set(raw))}')
    if not isinstance(raw['validated_actions'],list) or not isinstance(raw['pending'],list) or not isinstance(raw['evidence'],dict):
        raise ValueError('Capability contract has invalid field types')
    evidence={}
    for action,details in sorted(raw.get('evidence',{}).items()):
        item={key:details[key] for key in ('source','stage') if key in details}
        for key in ('checks','kits'):
            if key in details:item[key]=sorted(details[key])
        evidence[action]=item
    return {'version':raw.get('version'),'validated_actions':sorted(raw.get('validated_actions',[])),
      'pending':sorted(raw.get('pending',[])),'evidence':evidence}


def semantic_hash(env,model,version=SEMANTIC_HASH_VERSION):
    if version not in (1,SEMANTIC_HASH_VERSION):raise ValueError(f'Unsupported semantic hash version: {version}')
    config=env.config if hasattr(env,'config') else {}
    artifacts={'scenarios':file_hash(ROOT/'data/learning/scenarios.json') if (ROOT/'data/learning/scenarios.json').exists() else None,
      'capabilities':file_hash(ROOT/'outputs/native_capabilities_v2.json') if version==1 and (ROOT/'outputs/native_capabilities_v2.json').exists() else capability_contract(),
      'equipment':file_hash(ROOT/'data/learning/equipment_catalog.json') if (ROOT/'data/learning/equipment_catalog.json').exists() else None}
    payload={'config':semantic_config(config),
      'sensory_schema':getattr(getattr(env,'encoder',None),'config',{}),
      'action_catalog':getattr(getattr(env,'catalog',None),'entries',[]),
      'brain_dynamics':vars(getattr(getattr(env,'brain',None),'dynamics',object())) if getattr(getattr(env,'brain',None),'dynamics',None) is not None else {},
      'artifacts':artifacts,
      'observation_shape':list(model.observation_space.shape),'action_count':int(model.action_space.n)}
    if version>=2:payload['semantic_hash_version']=version
    import hashlib
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def save_checkpoint(name,brain,model,env,counters,store,teaching_examples=None,checkpoint_root=None):
    policy_state=model.policy.state_dict();optimizer_state=model.policy.optimizer.state_dict()
    _assert_finite(policy_state,'policy');_assert_finite(optimizer_state,'optimizer')
    _assert_finite(env.readout.filters,'readout.filters');_assert_finite(env.last_features,'last_features')
    for field in ('v','g','gains','pretrace','posttrace','eligibility'):_assert_finite(getattr(brain,field),f'brain.{field}')
    root=(Path(checkpoint_root) if checkpoint_root is not None else RUNTIME/'checkpoints')/name;root.mkdir(parents=True,exist_ok=True)
    pointer=root/'current.json';previous=json.loads(pointer.read_text()) if pointer.exists() else {}
    generation=1-int(previous.get('generation',1));directory=root/str(generation);directory.mkdir(exist_ok=True)
    brain_hash=brain.save(directory/'brain.npz',{'condition':env.condition,'seed':env.seed_base})
    path=directory/'policy.pt';temporary=path.with_suffix('.tmp')
    state={'version':2,'policy':policy_state,'optimizer':optimizer_state,
      'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
      'filters':torch.from_numpy(env.readout.filters.copy()),'features':torch.from_numpy(env.last_features.copy()),
      **({'sensor_filters':torch.from_numpy(env.sensor_readout.filters.copy())} if hasattr(env,'sensor_readout') else {}),
      'brain_mode':getattr(env,'brain_mode','connectome'),
      'counters':counters,'episode_count':env.episode_count,'total_steps':env.total_steps,'total_simulated_ms':env.total_simulated_ms,'num_timesteps':model.num_timesteps,
      'config':env.config if hasattr(env,'config') else {},'game_episode_resume':'restart interrupted episode; game heap is not serialized'}
    torch.save(state,temporary);os.replace(temporary,path)
    examples_metadata={}
    if teaching_examples is not None:
        example_path=directory/'examples.npz'
        atomic_npz(example_path,{'features':np.stack([e[0] for e in teaching_examples]) if teaching_examples else np.empty((0,len(env.last_features)),np.float32),
          'actions':np.array([e[1] for e in teaching_examples],np.int64),
          'masks':np.stack([e[2] for e in teaching_examples]) if teaching_examples else np.empty((0,model.action_space.n),bool),
          'stages':np.array([e[3] if len(e)>3 else 'unknown' for e in teaching_examples],dtype='<U32') if teaching_examples else np.empty(0,dtype='<U32'),
          **({'episodes':np.array([e[4] for e in teaching_examples],np.int64)} if teaching_examples and all(len(e)>4 for e in teaching_examples) else {}),
          **({'sensor_features':np.stack([e[5] for e in teaching_examples])} if teaching_examples and all(len(e)>5 for e in teaching_examples) else {})})
        examples_metadata={'examples_sha256':file_hash(example_path),'examples_count':len(teaching_examples)}
    manifest={'version':2,'generation':generation,'brain_sha256':brain_hash,'policy_sha256':file_hash(path),
      'condition':env.condition,'seed':env.seed_base,'saved':time.time(),'counters':counters,'graph_hash':brain.graph_hash,
      'run':store.run,'actor_sha256':actor_hash(model),'semantic_hash_version':SEMANTIC_HASH_VERSION,
      'semantic_sha256':semantic_hash(env,model),**examples_metadata}
    atomic_json(directory/'manifest.json',manifest);atomic_json(pointer,manifest)
    store.checkpoint(env.condition,env.seed_base,pointer,file_hash(pointer),manifest)
    return pointer

def load_checkpoint(pointer,brain,model,env):
    pointer=Path(pointer);manifest=json.loads(pointer.read_text());directory=pointer.parent/str(manifest['generation'])
    semantic_version=int(manifest.get('semantic_hash_version',1))
    if semantic_version not in (1,SEMANTIC_HASH_VERSION):raise ValueError('Unsupported checkpoint semantic hash version')
    expected_semantic=semantic_hash(env,model,semantic_version)
    if manifest.get('semantic_sha256')!=expected_semantic:raise ValueError('Checkpoint semantic protocol mismatch')
    for file,key in [('brain.npz','brain_sha256'),('policy.pt','policy_sha256')]:
        if file_hash(directory/file)!=manifest[key]:raise ValueError('Checkpoint integrity mismatch: '+file)
    brain.load(directory/'brain.npz')
    # Only our local hash-verified checkpoint is deserialized; never accept uploaded pickle data.
    state=torch.load(directory/'policy.pt',map_location='cpu',weights_only=False)
    _assert_finite(state['policy'],'policy');_assert_finite(state['optimizer'],'optimizer')
    _assert_finite(state['filters'],'readout.filters');_assert_finite(state['features'],'last_features')
    model.policy.load_state_dict(state['policy']);model.policy.optimizer.load_state_dict(state['optimizer'])
    if manifest.get('actor_sha256') and actor_hash(model)!=manifest['actor_sha256']:raise ValueError('Restored actor differs from checkpoint')
    torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    if state.get('brain_mode','connectome')!=getattr(env,'brain_mode','connectome'):raise ValueError('Checkpoint brain mode mismatch')
    env.readout.filters[:]=state['filters'].numpy();env.last_features=state['features'].numpy().copy()
    if 'sensor_filters' in state and hasattr(env,'sensor_readout'):env.sensor_readout.filters[:]=state['sensor_filters'].numpy()
    env.episode_count=state['episode_count'];env.total_steps=state['total_steps'];model.num_timesteps=state['num_timesteps']
    env.total_simulated_ms=state.get('total_simulated_ms',state['total_steps']*50)
    return state['counters']


def load_teaching_examples(pointer,legacy_path):
    """Prefer examples published with the same weights; old checkpoints remain readable."""
    pointer=Path(pointer);manifest=json.loads(pointer.read_text()) if pointer.exists() else {}
    path=pointer.parent/str(manifest.get('generation',0))/'examples.npz' if manifest.get('examples_sha256') else Path(legacy_path)
    if manifest.get('examples_sha256'):
        if not path.exists() or file_hash(path)!=manifest['examples_sha256']:raise ValueError('Checkpoint integrity mismatch: examples.npz')
    elif not path.exists():return []
    with np.load(path,allow_pickle=False) as archive:
        x=archive['features'];y=archive['actions'];masks=archive['masks'];stages=archive['stages'] if 'stages' in archive else None
        if len(x)!=len(y) or len(x)!=len(masks) or manifest.get('examples_count',len(x))!=len(x):raise ValueError('Teaching example count mismatch')
        if stages is not None and 'episodes' in archive and 'sensor_features' in archive:
            return list(zip(x,y.tolist(),masks,stages.tolist(),archive['episodes'].tolist(),archive['sensor_features']))
        if stages is not None and 'episodes' in archive:return list(zip(x,y.tolist(),masks,stages.tolist(),archive['episodes'].tolist()))
        return list(zip(x,y.tolist(),masks,stages.tolist())) if stages is not None else list(zip(x,y.tolist(),masks))
