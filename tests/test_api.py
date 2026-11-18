import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as api


class FakeModel:
    classes_ = np.array([0, 1])
    n_features_in_ = 9

    def predict_proba(self, x):
        positive = np.exp(-x[:, 0] / 20.0)
        return np.column_stack((1 - positive, positive))


def sample_trip():
    return {
        'trip_id': 'demo',
        'track': [[-93.3495 + .0005 * i, 44.86003, float(i)] for i in range(6)],
        'nodes': [[-93.350 + .001 * i, 44.860] for i in range(4)],
        'arcs': [[0, 1], [1, 2], [2, 3]],
        'candidate_count': 3,
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, '_load_model', lambda: setattr(api, 'model', FakeModel()))
    with TestClient(api.app) as test_client:
        yield test_client


def write_trip(root, trip_id='00000001', with_route=True):
    folder = root / trip_id
    folder.mkdir(parents=True)
    trip = sample_trip()
    for suffix, rows in (('track', trip['track']), ('nodes', trip['nodes']), ('arcs', trip['arcs'])):
        (folder / f'{trip_id}.{suffix}').write_text(
            '\n'.join(' '.join(map(str, row)) for row in rows))
    if with_route:
        (folder / f'{trip_id}.route').write_text('0\n1\n2\n')
    return folder


def test_health_and_prediction(client):
    assert client.get('/health').json()['model_loaded'] is True
    response = client.post('/predict', json=sample_trip())
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body['matched_arc_indices']) == 6
    assert len(body['matched_coordinates']) == 6
    assert body['candidate_count'] == 3


def test_dataset_listing_prediction_and_map_never_render_reference_route(
        client, tmp_path, monkeypatch):
    root = tmp_path / 'evaluator_test'
    root.mkdir()
    write_trip(root, with_route=True)
    monkeypatch.setattr(api, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(api, 'EVALUATOR_SPLIT', 'evaluator_test')
    assert client.get('/trips').json()['trips'] == ['00000001']
    assert client.post('/trips/00000001/predict').status_code == 200
    response = client.get('/trips/00000001/map')
    assert response.status_code == 200
    assert 'leaflet' in response.text.lower()
    assert 'Reference route' not in response.text


def test_uploaded_map_and_ground_truth_field_is_ignored(client):
    trip = sample_trip()
    trip['route'] = [2, 2, 2]
    response = client.post('/map', json=trip)
    assert response.status_code == 200, response.text
    assert 'Original GPS observations' in response.text
    assert 'Reference route' not in response.text


@pytest.mark.parametrize('mutation', [
    lambda item: item.update(track=[]),
    lambda item: item.update(track=[[500, 44]]),
    lambda item: item.update(nodes=[[-93.35, 44.86, 1]]),
    lambda item: item.update(arcs=[[0, 99]]),
    lambda item: item.update(candidate_count=0),
    lambda item: item.update(transition_weight=-1),
])
def test_invalid_input_returns_client_error(client, mutation):
    trip = sample_trip()
    mutation(trip)
    response = client.post('/predict', json=trip)
    assert response.status_code in (400, 422), response.text


def test_dataset_identifier_validation(client):
    assert client.post('/trips/abc/predict').status_code == 400
    assert client.post('/trips/12345678/predict').status_code == 404


def test_missing_model_returns_503(client):
    api.model = None
    assert client.post('/predict', json=sample_trip()).status_code == 503
