import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import learning_runtime
from learning_runtime import ResourceGuard
from runtime_config import DEFAULTS, load_runtime_config


def test_runtime_config_defaults_are_conservative_and_single_monitor_is_explicit(tmp_path):
    missing = tmp_path / "missing.json"
    assert load_runtime_config(missing) == DEFAULTS

    single = tmp_path / "single.json"
    single.write_text(
        json.dumps(
            {
                "version": 1,
                "display_mode": "single_monitor",
                "require_secondary_monitor": False,
                "keep_display_awake": True,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_runtime_config(single)
    assert loaded["display_mode"] == "single_monitor"
    assert loaded["require_secondary_monitor"] is False
    assert loaded["keep_display_awake"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"display_mode": "single_monitor", "require_secondary_monitor": True},
        {"display_mode": "secondary_monitor", "require_secondary_monitor": False},
        {"display_mode": "unknown", "require_secondary_monitor": False},
        {"display_mode": "single_monitor", "require_secondary_monitor": 0},
    ],
)
def test_runtime_config_rejects_ambiguous_display_policies(tmp_path, payload):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_runtime_config(path)


def resource_guard(tmp_path, require_secondary_monitor):
    """Build only the state used by reasons(), without changing process priority."""
    for directory in (tmp_path / "runtime", tmp_path / "work" / "frames", tmp_path / "outputs"):
        directory.mkdir(parents=True, exist_ok=True)
    guard = ResourceGuard.__new__(ResourceGuard)
    guard.config = {
        "minimum_available_ram_gb": 0,
        "maximum_worker_ram_gb": 100,
        "minimum_free_disk_gb": 0,
        "artifact_quota_gb": 100,
    }
    guard.process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))
    guard.last_disk_check = 0
    guard.disk_reason = ""
    guard.display_missing = False
    guard.require_secondary_monitor = require_secondary_monitor
    return guard


def test_resource_guard_does_not_probe_or_pause_for_secondary_display_in_single_monitor_mode(tmp_path, monkeypatch):
    guard = resource_guard(tmp_path, require_secondary_monitor=False)
    monkeypatch.setattr(learning_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(learning_runtime, "RUNTIME", tmp_path / "runtime")

    def unexpected_probe():
        raise AssertionError("single-monitor mode must not require a secondary display")

    monkeypatch.setattr(learning_runtime, "secondary_monitors", unexpected_probe)
    assert guard.reasons() == []
    assert guard.display_missing is False


def test_resource_guard_retains_fail_closed_secondary_monitor_policy(tmp_path, monkeypatch):
    guard = resource_guard(tmp_path, require_secondary_monitor=True)
    monkeypatch.setattr(learning_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(learning_runtime, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(learning_runtime, "secondary_monitors", lambda: [])
    reasons = guard.reasons()
    assert reasons == ["Monitor secundario desconectado"]
    assert guard.display_missing is True


def test_resource_guard_ram_pause_uses_hysteresis(tmp_path, monkeypatch):
    guard = resource_guard(tmp_path, require_secondary_monitor=False)
    guard.config.update(minimum_available_ram_gb=1.5, resume_available_ram_gb=2.0)
    monkeypatch.setattr(learning_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(learning_runtime, "RUNTIME", tmp_path / "runtime")
    available = {"bytes": 1.4e9}
    monkeypatch.setattr(learning_runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=available["bytes"]))
    assert guard.reasons() and guard.ram_low
    available["bytes"] = 1.8e9
    assert guard.reasons(), "must stay paused between the pause and resume thresholds"
    available["bytes"] = 2.1e9
    assert guard.reasons() == [] and not guard.ram_low
    available["bytes"] = 1.8e9
    assert guard.reasons() == [], "must keep running above the pause threshold"


def test_resource_guard_reloads_operational_limits_from_config(tmp_path, monkeypatch):
    guard = resource_guard(tmp_path, require_secondary_monitor=False)
    guard.config.update(minimum_available_ram_gb=1.5, resume_available_ram_gb=2.0)
    monkeypatch.setattr(learning_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(learning_runtime, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(learning_runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=1.2e9))
    assert guard.reasons()
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "learning.json").write_text('{"minimum_available_ram_gb": 1.0, "resume_available_ram_gb": 1.1, "dataset": "ignored"}')
    guard.last_limits_check = 0
    assert guard.reasons() == [] and guard.limits == {"minimum_available_ram_gb": 1.0, "resume_available_ram_gb": 1.1}
