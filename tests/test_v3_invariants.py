import ast
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learning_brain import LearningBrain
import learning_checkpoint
from learning_checkpoint import load_checkpoint, save_checkpoint, semantic_hash
from learning_curriculum import exact_minibatch_size, run_checkpointed_evaluation, train_adapter_episode
from learning_policy import actor_hash, make_model
from learning_adapter import ActionCatalog, decision_required
from learning_env import FlyOperatorEnv
from learning_runtime import StopTraining


class SmallEnv(gym.Env):
    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Box(-5, 5, shape=(4,), dtype=np.float32)


class CheckpointEnv(SmallEnv):
    def __init__(self):
        self.condition = "adapter"
        self.seed_base = 7
        self.readout = type("Readout", (), {})()
        self.readout.filters = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.last_features = np.arange(4, dtype=np.float32)
        self.episode_count = 11
        self.total_steps = 23
        self.total_simulated_ms = 1150
        self.config = {"version": 3}


class RecordingStore:
    def __init__(self, run="v3-test"):
        self.run = run
        self.saved = []

    def checkpoint(self, *args):
        self.saved.append(args)


def tiny_brain(seed=7):
    graph = (
        np.array([10, 20, 30], np.int64),
        np.array([0, 1, 2, 2], np.int64),
        np.array([1, 2], np.int32),
        np.array([1, 1], np.int32),
        np.ones(3, np.int8),
    )
    return LearningBrain(graph=graph, seed=seed)


def learning_config():
    return json.loads((ROOT / "config" / "learning.json").read_text())


def test_campaign_roots_isolate_identically_named_checkpoints(tmp_path):
    """A new experiment cannot silently resume weights from an older campaign."""
    cfg = learning_config()
    env = CheckpointEnv()
    brain = tiny_brain()
    model = make_model(env, cfg, 7)
    store = RecordingStore()

    brain.gains[0] = 1.25
    first_actor = actor_hash(model)
    first = save_checkpoint(
        "adapter_7", brain, model, env, {"steps": 101}, store,
        checkpoint_root=tmp_path / "campaign-a" / "checkpoints",
    )

    brain.gains[0] = 2.5
    with torch.no_grad():
        model.policy.action_net.bias.add_(0.75)
    second_actor = actor_hash(model)
    second = save_checkpoint(
        "adapter_7", brain, model, env, {"steps": 202}, store,
        checkpoint_root=tmp_path / "campaign-b" / "checkpoints",
    )

    assert first != second
    assert first.parent.parent == tmp_path / "campaign-a" / "checkpoints"
    assert second.parent.parent == tmp_path / "campaign-b" / "checkpoints"
    assert first_actor != second_actor
    assert json.loads(first.read_text())["counters"]["steps"] == 101
    assert json.loads(second.read_text())["counters"]["steps"] == 202

    restored = load_checkpoint(first, brain, model, env)
    assert restored == {"steps": 101}
    assert brain.gains[0] == np.float32(1.25)
    assert actor_hash(model) == first_actor


def test_checkpoint_restore_removes_evaluation_side_effects(tmp_path):
    """The save/evaluate/load pattern restores neural, policy, RNG and counters."""
    cfg = learning_config()
    env = CheckpointEnv()
    brain = tiny_brain()
    model = make_model(env, cfg, 7)
    store = RecordingStore()
    counters = {"steps": 77, "episodes": 4, "evaluated_at": 32}

    random.seed(1234)
    np.random.seed(2345)
    torch.manual_seed(3456)
    brain.plasticity = True
    brain.step([0, 1], [80, 90], 3.0)
    brain.gains[0] = 1.375
    pointer = save_checkpoint(
        "adapter_7", brain, model, env, counters, store,
        checkpoint_root=tmp_path / "campaign" / "checkpoints",
    )
    saved_actor = actor_hash(model)
    saved_brain = {name: getattr(brain, name).copy() for name in brain.array_state}
    saved_cursor = brain.cursor
    saved_filters = env.readout.filters.copy()
    saved_features = env.last_features.copy()
    expected_rng = (random.random(), np.random.random(), torch.rand(3))

    # Stand in for several validation episodes mutating every persisted subsystem.
    random.random()
    np.random.random(100)
    torch.rand(100)
    brain.reset(999)
    brain.gains[:] = 3.0
    with torch.no_grad():
        model.policy.action_net.weight.add_(1.0)
    env.readout.filters.fill(-9)
    env.last_features.fill(-8)
    env.episode_count = 99
    env.total_steps = 999
    env.total_simulated_ms = 9999
    model.num_timesteps = 999

    restored = load_checkpoint(pointer, brain, model, env)

    assert restored == counters
    assert actor_hash(model) == saved_actor
    assert brain.cursor == saved_cursor
    for name, expected in saved_brain.items():
        assert np.array_equal(getattr(brain, name), expected), name
    assert np.array_equal(env.readout.filters, saved_filters)
    assert np.array_equal(env.last_features, saved_features)
    assert (env.episode_count, env.total_steps, env.total_simulated_ms, model.num_timesteps) == (11, 23, 1150, 0)
    actual_rng = (random.random(), np.random.random(), torch.rand(3))
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    assert torch.equal(actual_rng[2], expected_rng[2])


def semantic_environment(config):
    env = CheckpointEnv()
    env.config = copy.deepcopy(config)
    env.encoder = SimpleNamespace(config={"version": 31, "ports": "fixed"})
    env.catalog = SimpleNamespace(entries=[{"action": "wait"}, {"action": "move", "distance": 0.75}])
    env.brain = tiny_brain()
    return env


def test_semantic_hash_ignores_resources_but_tracks_protocol_and_artifacts(monkeypatch):
    cfg = learning_config()
    env = semantic_environment(cfg)
    model = make_model(SmallEnv(), cfg, 7)
    digests = {
        "scenarios.json": "scenario-a",
        "equipment_catalog.json": "equipment-a",
    }
    capabilities = {"version": 2, "validated_actions": ["move"], "pending": ["fire"], "evidence": {"move": {"stage": "move"}}}
    monkeypatch.setattr(learning_checkpoint, "file_hash", lambda path: digests[Path(path).name])
    monkeypatch.setattr(learning_checkpoint, "capability_contract", lambda path=None: copy.deepcopy(capabilities))
    baseline = semantic_hash(env, model)

    operational_changes = {
        "max_cpu_threads": 99,
        "minimum_available_ram_gb": 99,
        "pilot_hours": 999,
        "worker_priority": "idle",
    }
    for field, value in operational_changes.items():
        changed = semantic_environment(cfg)
        changed.config[field] = value
        assert semantic_hash(changed, model) == baseline, field

    changed = semantic_environment(cfg)
    changed.config["free_wait_penalty"] = cfg["free_wait_penalty"] + 0.125
    assert semantic_hash(changed, model) != baseline

    digests["scenarios.json"] = "scenario-b"
    scenario_hash = semantic_hash(env, model)
    assert scenario_hash != baseline
    digests["scenarios.json"] = "scenario-a"
    capabilities["validated_actions"].append("fire")
    capabilities["pending"].remove("fire")
    capability_hash = semantic_hash(env, model)
    assert capability_hash != baseline
    assert capability_hash != scenario_hash


def test_capability_contract_ignores_provenance_but_tracks_action_meaning(tmp_path):
    path=tmp_path/'capabilities.json'
    base={"version":2,"validated_actions":["move"],"pending":["fire"],
      "evidence":{"move":{"source":"control.json","checks":["native_movement"]}},
      "sources":[{"file":"control.json","sha256":"old","session":{"pid":1,"dll_sha256":"old"}}],
      "generator":{"sha256":"old"}}
    path.write_text(json.dumps(base),encoding='utf-8');first=learning_checkpoint.capability_contract(path)
    base["sources"]=[{"file":"control.json","sha256":"new","session":{"pid":99,"dll_sha256":"new"}}]
    base["generator"]={"sha256":"new"};path.write_text(json.dumps(base),encoding='utf-8')
    assert learning_checkpoint.capability_contract(path)==first
    base["validated_actions"].append("fire");base["pending"].remove("fire");path.write_text(json.dumps(base),encoding='utf-8')
    assert learning_checkpoint.capability_contract(path)!=first


def test_capability_contract_rejects_partial_evidence(tmp_path):
    path=tmp_path/'capabilities.json';path.write_text(json.dumps({'version':2,'validated_actions':['move']}),encoding='utf-8')
    with pytest.raises(ValueError,match='missing fields'):
        learning_checkpoint.capability_contract(path)


def test_semantic_hash_rejects_unknown_version(monkeypatch):
    cfg=learning_config();env=semantic_environment(cfg);model=make_model(SmallEnv(),cfg,7)
    with pytest.raises(ValueError,match='Unsupported semantic hash version'):
        semantic_hash(env,model,version=99)


@pytest.mark.parametrize("field", ["filters", "features"])
def test_save_rejects_nonfinite_observation_state_without_publishing_pointer(tmp_path, field):
    cfg = learning_config()
    env = CheckpointEnv()
    brain = tiny_brain()
    model = make_model(env, cfg, 7)
    store = RecordingStore()
    if field == "filters":
        env.readout.filters[0, 0] = np.nan
    else:
        env.last_features[0] = np.nan

    root = tmp_path / "campaign" / "checkpoints"
    with pytest.raises(FloatingPointError, match="Non-finite checkpoint"):
        save_checkpoint("adapter_7", brain, model, env, {"steps": 1}, store, checkpoint_root=root)

    assert not (root / "adapter_7" / "current.json").exists()
    assert store.saved == []


@pytest.mark.parametrize("field", ["filters", "features"])
def test_load_rejects_nonfinite_observation_state_before_applying_it(tmp_path, field):
    cfg = learning_config()
    env = CheckpointEnv()
    brain = tiny_brain()
    model = make_model(env, cfg, 7)
    store = RecordingStore()
    pointer = save_checkpoint(
        "adapter_7", brain, model, env, {"steps": 1}, store,
        checkpoint_root=tmp_path / "campaign" / "checkpoints",
    )
    manifest = json.loads(pointer.read_text())
    policy_path = pointer.parent / str(manifest["generation"]) / "policy.pt"
    state = torch.load(policy_path, map_location="cpu", weights_only=False)
    state[field].reshape(-1)[0] = np.nan
    torch.save(state, policy_path)
    manifest["policy_sha256"] = learning_checkpoint.file_hash(policy_path)
    pointer.write_text(json.dumps(manifest), encoding="utf-8")
    poisoned_pointer = pointer.read_bytes()

    env.readout.filters.fill(17)
    env.last_features.fill(19)
    expected_filters = env.readout.filters.copy()
    expected_features = env.last_features.copy()
    with pytest.raises(FloatingPointError, match="Non-finite checkpoint"):
        load_checkpoint(pointer, brain, model, env)

    assert np.array_equal(env.readout.filters, expected_filters)
    assert np.array_equal(env.last_features, expected_features)
    assert pointer.read_bytes() == poisoned_pointer
    assert len(store.saved) == 1


class CapturingModel:
    observation_space = gym.spaces.Box(-5, 5, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)
    device = "cpu"

    def __init__(self, bootstrap):
        self.rollout_buffer = object()
        self.batch_size = 64
        self.policy = type("Policy", (), {})()
        self.policy.predict_values = lambda features: torch.tensor([[bootstrap]], dtype=torch.float32)
        self.captured_advantages = None
        self.captured_returns = None

    def train(self):
        self.captured_advantages = self.rollout_buffer.advantages[:, 0].copy()
        self.captured_returns = self.rollout_buffer.returns[:, 0].copy()


def transition(reward, value, macro_steps, episode_start=False):
    return {
        "observation": np.array([0.25, -0.25], np.float32),
        "action": 1,
        "reward": reward,
        "episode_start": episode_start,
        "value": torch.tensor([value], dtype=torch.float32),
        "logprob": torch.tensor([0.0], dtype=torch.float32),
        "mask": np.ones(2, bool),
        "macro_steps": macro_steps,
    }


def test_macro_action_duration_controls_discount_and_time_limit_bootstrap():
    """A three-tick native command discounts by gamma**3, not one decision tick."""
    cfg = {"ppo": {"gamma": 0.9, "gae_lambda": 0.8}}
    model = CapturingModel(bootstrap=4.0)
    transitions = [
        transition(1.0, 0.5, 2, episode_start=True),
        transition(2.0, 1.0, 3),
    ]

    result = train_adapter_episode(
        model, transitions, np.zeros(2, np.float32), terminated=False, truncated=True, config=cfg
    )

    gamma = cfg["ppo"]["gamma"]
    gae = cfg["ppo"]["gae_lambda"]
    last_advantage = 2.0 + gamma**3 * 4.0 - 1.0
    first_delta = 1.0 + gamma**2 * 1.0 - 0.5
    first_advantage = first_delta + gamma**2 * gae * last_advantage
    assert np.allclose(model.captured_advantages, [first_advantage, last_advantage])
    assert np.allclose(model.captured_returns, [first_advantage + 0.5, last_advantage + 1.0])
    assert result["bootstrap_applied"] is True
    assert result["environment_ticks"] == 5


def test_promotion_gate_is_built_from_the_configured_main_track():
    """Protocol 3.3: the gate comes from config (main track x seeds), never from a hardcoded list."""
    from learning_curriculum import promotion_ready

    cfg = learning_config()
    main = cfg["tracks"]["main"]
    seeds = cfg["seeds"]
    assert main and len(seeds) >= 3
    assert not set(main) & set(cfg["tracks"]["science"])
    assert "train_curriculum" not in str(promotion_ready.__module__)  # the gate is a library function

    blocks = {"move": {f"{condition}_{seed}": [{"passed": True}] for condition in main for seed in seeds}}
    assert promotion_ready(blocks, "move", main, seeds)
    # A science control passing its block cannot promote anything on its own.
    blocks["move"]["sensor_only_7"] = [{"passed": True}]
    del blocks["move"][f"{main[0]}_{seeds[0]}"]
    assert not promotion_ready(blocks, "move", main, seeds)
    assert cfg["promotion"]["block_episodes"] >= 25 and cfg["promotion"]["required_successes"] >= 24
    assert cfg["acceptance_evaluation_episodes_per_seed"] >= 30


def test_each_seed_gets_a_disjoint_30_mission_validation_block():
    scenarios = json.loads((ROOT / "data" / "learning" / "scenarios.json").read_text())
    cfg = learning_config()
    for stage in cfg["stage_order"]:
        names = [row["name"] for row in scenarios if row["stage"] == stage and row["split"] == "validation"]
        blocks = [set(names[slot * 30 : slot * 30 + 30]) for slot in range(len(cfg["seeds"]))]
        assert all(len(block) == 30 for block in blocks), stage
        assert not (blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2]), stage


@pytest.mark.parametrize(
    ("sample_count", "expected"),
    [(65, 13), (129, 43), (67, 67)],
)
def test_exact_minibatch_size_has_no_single_sample_tail(sample_count, expected):
    size = exact_minibatch_size(sample_count, 64)
    assert size == expected
    assert sample_count % size == 0
    assert size >= 2


def test_real_ppo_with_65_samples_keeps_every_parameter_finite():
    """Regression for SB3's undefined advantage normalization on a final batch of one."""
    cfg = learning_config()
    model = make_model(SmallEnv(), cfg, 19)
    transitions = []
    for i in range(65):
        obs = np.array([i / 65, np.sin(i), np.cos(i), -i / 65], np.float32)
        mask = np.ones(3, bool)
        features = torch.as_tensor(obs[None, :])
        action = torch.tensor([i % 3])
        with torch.no_grad():
            distribution = model.policy.get_distribution(features, action_masks=mask[None, :])
            value = model.policy.predict_values(features)
            logprob = distribution.log_prob(action)
        transitions.append(
            {
                "observation": obs,
                "action": int(action[0]),
                "reward": 1.0 if i == 64 else -0.001,
                "episode_start": i == 0,
                "value": value,
                "logprob": logprob,
                "mask": mask,
                "macro_steps": 1,
            }
        )

    result = train_adapter_episode(
        model, transitions, np.zeros(4, np.float32), terminated=True, truncated=False, config=cfg
    )

    assert result["updated"] and result["ticks"] == 65
    assert model.batch_size == cfg["ppo"]["batch_size"]
    assert all(torch.isfinite(parameter).all().item() for parameter in model.policy.parameters())


@pytest.mark.parametrize("failure", [StopTraining("stop requested"), RuntimeError("evaluation failed")])
def test_checkpointed_evaluation_restores_and_reenables_after_failure(failure):
    events = []
    token = object()

    def save():
        events.append("save")
        return token

    def restore(received):
        assert received is token
        events.append("restore")

    def evaluate():
        events.append("evaluate")
        raise failure

    def set_checkpointing(enabled):
        events.append(("checkpointing", enabled))

    with pytest.raises(type(failure), match=str(failure)):
        run_checkpointed_evaluation(save, restore, evaluate, set_checkpointing)

    assert events == [
        "save",
        ("checkpointing", False),
        "evaluate",
        "restore",
        ("checkpointing", True),
    ]


def test_reward_motor_decays_snapshot_eligibility_during_credit_delay():
    snapshot = {
        "version": 1,
        "cursor": 0,
        "edges": np.array([0], np.int32),
        "posts": np.array([1], np.int32),
        "values": np.array([1.0], np.float32),
    }
    descending = np.array([1], np.int32)
    credit = np.array([1.0], np.float32)

    fresh = tiny_brain()
    fresh.plasticity = True
    fresh_result = fresh.reward_motor(copy.deepcopy(snapshot), 1.0, descending, credit)

    aged = tiny_brain()
    aged.plasticity = True
    delay_ms = aged.dynamics.eligibility_ms
    aged.cursor = round(delay_ms / aged.dynamics.dt_ms)
    aged_result = aged.reward_motor(copy.deepcopy(snapshot), 1.0, descending, credit)

    assert fresh_result["eligibility_decay"] == pytest.approx(1.0)
    assert aged_result["credit_delay_ms"] == pytest.approx(delay_ms)
    assert aged_result["eligibility_decay"] == pytest.approx(np.exp(-1.0))
    assert 0 < aged_result["absolute_change"] < fresh_result["absolute_change"]
    assert aged_result["absolute_change"] / fresh_result["absolute_change"] == pytest.approx(
        np.exp(-1.0), rel=2e-3
    )


def busy_observation(*, commands=0, pending=False):
    return {
        "operator": {
            "health": 100,
            "commands": commands,
            "weapon_state": 5,
            "ammo": 10,
            "capacity": 30,
            "position": [0, 0, 0],
            "aim": [1, 0, 0],
        },
        "inventory": [],
        "visible_enemies": [],
        "visible_friendlies": [],
        "objects": [],
        "action_receipt": {"status": "in_progress" if pending else "completed"},
        "sim_time_ms": 50,
        "stage": "move",
    }


@pytest.mark.parametrize("observation", [busy_observation(commands=1), busy_observation(pending=True)])
def test_busy_native_action_exposes_the_interventions_the_engine_allows(observation):
    """Protocol 3.3: while a native command runs, continuing, stopping and cancelling are all the
    agent's choice; the mask no longer encodes the distance at which stopping is the right answer."""
    catalog = ActionCatalog()
    enabled = lambda mask: {entry["action"] for entry, allowed in zip(catalog.entries, mask) if allowed}
    observation['goal']=[4,0,0]
    far = catalog.mask(observation, {"wait", "move", "stop", "cancel"})
    assert enabled(far) == {"wait", "stop", "cancel"}
    assert decision_required(observation, far)
    observation['goal']=[.1,0,0]
    near=catalog.mask(observation,{"wait","move","stop","cancel"})
    assert np.array_equal(near, far)
    # The curriculum decides which families exist at all; that never depends on the situation.
    only_stop=catalog.mask(observation,{"wait","move","stop"},{"wait","move","stop"})
    assert enabled(only_stop)=={'wait','stop'}
    observation['stage']='cancel';observation['action_receipt']={'status':'in_progress','action':'door_breach'}
    assert enabled(catalog.mask(observation,{"wait","cancel"}))=={'wait','cancel'}


def test_move_curriculum_exposes_stop_and_cancel_interventions():
    source = (ROOT / "scripts" / "train_curriculum.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    stage_actions_node = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "STAGE_ACTIONS" for target in node.targets)
    )
    stage_actions = ast.literal_eval(stage_actions_node)
    assert {"wait", "move", "stop", "cancel"} <= stage_actions["move"]


def test_stance_stage_only_allows_crouching_and_requires_physical_state():
    observation=busy_observation()
    observation['stage']='stance'
    catalog=ActionCatalog()
    mask=catalog.mask(observation,{'wait','crouch'})
    crouch_values=[entry.get('value') for entry,enabled in zip(catalog.entries,mask)
                   if enabled and entry['action']=='crouch']
    # Protocol 3.3: both crouch values are legal; choosing the right one is the agent's job.
    assert crouch_values==[True,False]

    env=FlyOperatorEnv.__new__(FlyOperatorEnv)
    env.mission={'stage':'stance'}
    receipt={'status':'completed','action':'crouch'}
    assert not env.skill_success({'operator':{'health':100,'crouched':False},'action_receipt':receipt})
    assert env.skill_success({'operator':{'health':100,'crouched':True},'action_receipt':receipt})


def test_dashboard_read_retries_transient_windows_replace_lock(tmp_path,monkeypatch):
    import lab_api
    target=tmp_path/'state.json';target.write_text('{"state":"running"}',encoding='utf-8')
    original=Path.read_text;calls={'count':0}
    def flaky(path,*args,**kwargs):
        if path==target and calls['count']<2:
            calls['count']+=1
            raise PermissionError('atomic replace in progress')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',flaky)
    assert lab_api.read(target)=={'state':'running'}
    assert calls['count']==2
