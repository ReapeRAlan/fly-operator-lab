import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import lab_api
import lab_store


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def stored_step(run, recorded_at):
    return {
        'run': run, 'recorded_at': recorded_at, 'episode': 3, 'sequence': 9,
        'condition': 'adapter_half_updates', 'seed': 11, 'stage': 'move',
        'action_label': 'esperar', 'reward': 0.25, 'decision_required': False,
        'forced_wait': False, 'command': {'action': 'wait'},
        'decision': {'controller': 'actor', 'contributions': [{'body_id': 10001, 'logit_contribution': 0.12}]},
        'neurons_readout': [{'body_id': 10001, 'spikes': 3}],
        'activity': {'spikes': 17, 'active_neurons': 4, 'wall_seconds': 0.02},
        'observation': {'operator': {'health': 100}},
    }


@pytest.fixture
def isolated_dashboard(tmp_path, monkeypatch):
    runtime = tmp_path / 'work' / 'learning'
    runtime.mkdir(parents=True)
    (tmp_path / 'config').mkdir()
    write_json(tmp_path / 'config' / 'learning.json', {
        'experiment_id': 'v4.1-night1', 'version': 4,
        'semantic_protocol_version': '3.3',
    })
    monkeypatch.setattr(lab_store, 'RUNTIME', runtime)
    monkeypatch.setattr(lab_api, 'ROOT', tmp_path)
    monkeypatch.setattr(lab_api, 'RUNTIME', runtime)
    now = time.time()
    db = lab_store.connect()
    smoke = stored_step('smoke-v4.1-science', now - 120)
    db.execute('INSERT INTO steps(run,episode,sequence,payload) VALUES(?,?,?,?)',
               (smoke['run'], smoke['episode'], smoke['sequence'], json.dumps(smoke)))
    db.execute('INSERT INTO episodes(run,condition,seed,stage,split,success,native_victory,reward,seconds,payload) VALUES(?,?,?,?,?,?,?,?,?,?)',
               (smoke['run'], smoke['condition'], smoke['seed'], smoke['stage'], 'validation', 1, 1, .25, .3, json.dumps(smoke)))
    db.commit()
    db.close()
    write_json(runtime / 'status.json', {'run': smoke['run'], 'state': 'running'})
    write_json(runtime / 'latest.json', smoke)
    write_json(runtime / 'live_version.json', {'run': smoke['run'], 'revision': 77, 'updated': now, 'writing': False})
    write_json(runtime / 'gain_audit.jsonl', {'run': smoke['run'], 'revision': 77, 'mean': 1.0})
    return runtime, smoke


def test_selected_campaign_rejects_another_runs_live_snapshot(isolated_dashboard):
    runtime, smoke = isolated_dashboard
    with TestClient(lab_api.app) as client:
        current = client.get('/api/state').json()
        assert current['context']['selected_run'] == 'v4.1-night1'
        assert current['context']['freshness'] == 'obsoleto'
        assert current['context']['source_run'] == smoke['run']
        assert current['latest'] == {}
        circuit = client.get('/api/circuit/live').json()
        assert circuit['families'] is None
        assert circuit['coherent'] is False
        assert client.get('/api/plasticity').json()['history'] == []


def test_historical_selection_has_own_records_and_no_live_activity(isolated_dashboard):
    _, smoke = isolated_dashboard
    with TestClient(lab_api.app) as client:
        response = client.get('/api/state', params={'run': smoke['run']})
        assert response.status_code == 200
        state = response.json()
        assert state['context']['freshness'] == 'histórico'
        assert state['status']['state'] == 'historical'
        assert state['latest']['run'] == smoke['run']
        history = client.get('/api/history', params={'run': smoke['run']}).json()
        assert history['run'] == smoke['run']
        assert history['items'][0]['action_label'] == 'esperar'
        circuit = client.get('/api/circuit/live', params={'run': smoke['run']}).json()
        assert circuit['families'] is None
        assert circuit['context']['freshness'] == 'histórico'
        causal = client.get('/api/causal/current', params={'run': smoke['run']}).json()
        assert causal['observed']['active_readout'] == []
        assert causal['inferred']['contributions'][0]['body_id'] == 10001


def test_campaign_list_unknown_run_and_unarchived_audit(isolated_dashboard):
    runtime, smoke = isolated_dashboard
    with TestClient(lab_api.app) as client:
        campaigns = client.get('/api/campaigns').json()['campaigns']
        by_run = {item['run']: item for item in campaigns}
        assert by_run['v4.1-night1']['configured']
        assert by_run[smoke['run']]['episodes'] == 1
        assert client.get('/api/state', params={'run': 'does-not-exist'}).status_code == 404
        audit = client.get('/api/plasticity', params={'run': smoke['run']}).json()
        assert audit['archive_available'] is False
        assert audit['history'] == []
        write_json(runtime / 'campaigns' / smoke['run'] / 'gain_audit.jsonl',
                   {'run': smoke['run'], 'revision': 10, 'mean': 1.02, 'median': 1.0, 'p05': .9, 'p95': 1.1, 'std': .05})
        audit = client.get('/api/plasticity', params={'run': smoke['run']}).json()
        assert audit['archive_available'] is True
        assert [row['revision'] for row in audit['history']] == [10]
