"""API tests. Run from the repo root: pytest tests/ -q   (needs fastapi, httpx, pytest)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as api


class FakeModel:
    """Scores candidates by closeness (feature 0 = distance in metres)."""
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        p = np.exp(-x[:, 0] / 20.0)
        return np.column_stack((1 - p, p))


@pytest.fixture()
def client():
    api.model = FakeModel()
    with TestClient(api.app) as c:
        api.model = FakeModel()  # startup may reset to None if no model file
        yield c
    api.model = None


def grid_trip():
    # 4 nodes along a street (east-west), 3 arcs, GPS points just north of it
    nodes = [[-93.350 + 0.001 * i, 44.860] for i in range(4)]
    arcs = [[0, 1], [1, 2], [2, 3]]
    track = [[-93.3495 + 0.0005 * i, 44.86003, float(i)] for i in range(6)]
    return {'trip_id': 'demo', 'track': track, 'nodes': nodes, 'arcs': arcs, 'candidate_count': 3}


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200 and r.json()['model_loaded'] is True


def test_predict_ok(client):
    r = client.post('/predict', json=grid_trip())
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body['matched_arc_indices']) == 6
    assert len(body['matched_coordinates']) == 6
    assert body['route'] == sorted(body['route'])  # drives west to east
    assert set(body['route']) <= {0, 1, 2}


def test_predict_without_timestamps(client):
    t = grid_trip()
    t['track'] = [row[:2] for row in t['track']]
    assert client.post('/predict', json=t).status_code == 200


def test_no_model_503(client):
    api.model = None
    assert client.post('/predict', json=grid_trip()).status_code == 503


@pytest.mark.parametrize('mutate', [
    lambda t: t.pop('track'),
    lambda t: t.update(track=[]),
    lambda t: t.update(track=[[1.0]]),
    lambda t: t.update(track=[[500.0, 44.0]]),                 # bad longitude
    lambda t: t.update(nodes=[[-93.35, 44.86, 9.0]]),          # wrong width
    lambda t: t.update(arcs=[[0, 99]]),                        # node index out of range
    lambda t: t.update(arcs=[[-1, 0]]),                        # negative index
    lambda t: t.update(arcs=[[0, 1.5]]),                       # non-integer
    lambda t: t.update(candidate_count=0),
    lambda t: t.update(candidate_count=1000),
    lambda t: t.update(transition_weight=-1),
    lambda t: t.update(track=[[-93.35, 44.86, 0], [-93.35, 44.86]]),  # ragged
])
def test_bad_input_is_4xx_not_500(client, mutate):
    t = grid_trip()
    mutate(t)
    r = client.post('/predict', json=t)
    assert r.status_code in (400, 422), (r.status_code, r.text)


def test_non_json_body(client):
    r = client.post('/predict', content='not json', headers={'content-type': 'application/json'})
    assert r.status_code == 422


def test_dataset_trip_validation(client):
    assert client.post('/trips/abc/predict').status_code == 400
    assert client.post('/trips/12345678/predict').status_code == 404
    assert client.post('/trips/00000001/predict?candidate_count=0').status_code == 422


def test_cors_preflight(client):
    r = client.options('/predict', headers={'Origin': 'http://x', 'Access-Control-Request-Method': 'POST'})
    assert r.status_code == 200
