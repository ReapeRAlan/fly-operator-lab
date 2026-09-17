"""Operational settings that do not change the scientific campaign identity."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'version': 1,
    'display_mode': 'secondary_monitor',
    'require_secondary_monitor': True,
    'keep_display_awake': True,
}


def load_runtime_config(path=None):
    """Load and validate the local display/session policy.

    Missing files retain the conservative two-monitor behavior used by the
    original laboratory launcher.  This file is deliberately separate from
    learning.json so changing a monitor does not invalidate checkpoints or
    scientific evidence hashes.
    """
    target = Path(path) if path is not None else ROOT/'config/runtime.json'
    try:
        raw = json.loads(target.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        raw = {}
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid runtime configuration: {target}') from exc
    if not isinstance(raw, dict):
        raise ValueError('Runtime configuration must be a JSON object')
    result = {**DEFAULTS, **raw}
    if type(result['require_secondary_monitor']) is not bool:
        raise ValueError('require_secondary_monitor must be boolean')
    if type(result['keep_display_awake']) is not bool:
        raise ValueError('keep_display_awake must be boolean')
    allowed = {'single_monitor', 'secondary_monitor'}
    if result['display_mode'] not in allowed:
        raise ValueError(f'display_mode must be one of {sorted(allowed)}')
    expected = 'secondary_monitor' if result['require_secondary_monitor'] else 'single_monitor'
    if result['display_mode'] != expected:
        raise ValueError('display_mode and require_secondary_monitor disagree')
    return result
