"""FastAPI service for PathSnap map matching."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import math
import os
from pathlib import Path
import pickle
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pathsnap import Trip, decode


log = logging.getLogger('pathsnap.api')
MODEL_PATH = Path(os.environ.get('PATHSNAP_MODEL', 'artifacts/models/histgb-20261118T070402Z.pkl'))
DATA_ROOT = Path(os.environ.get('PATHSNAP_DATA', '.'))
EVALUATOR_SPLIT = os.environ.get('PATHSNAP_EVALUATOR_SPLIT', 'evaluator_test')
_origins = os.environ.get('PATHSNAP_CORS_ORIGINS', '*')
CORS_ORIGINS = [origin.strip() for origin in _origins.split(',') if origin.strip()]
CANDIDATE_COUNT = int(os.environ.get('PATHSNAP_CANDIDATE_COUNT', '24'))
TRANSITION_WEIGHT = float(os.environ.get('PATHSNAP_TRANSITION_WEIGHT', '0'))
CONTEXT_RADIUS = int(os.environ.get('PATHSNAP_CONTEXT_RADIUS', '0'))
MAX_TRACK_POINTS = int(os.environ.get('PATHSNAP_MAX_TRACK_POINTS', '50000'))
MAX_NODES = int(os.environ.get('PATHSNAP_MAX_NODES', '500000'))
MAX_ARCS = int(os.environ.get('PATHSNAP_MAX_ARCS', '1000000'))
model: Any | None = None
model_error: str | None = None


def _load_model() -> None:
    global model, model_error, CANDIDATE_COUNT, CONTEXT_RADIUS
    model = None
    model_error = None
    if not MODEL_PATH.is_file():
        model_error = f'Model file not found: {MODEL_PATH}'
        log.error(model_error)
        return
    try:
        with MODEL_PATH.open('rb') as stream:
            model = pickle.load(stream)
        metadata_path = MODEL_PATH.with_suffix('.json')
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if 'PATHSNAP_CANDIDATE_COUNT' not in os.environ:
                CANDIDATE_COUNT = int(metadata.get('candidate_count', CANDIDATE_COUNT))
            if 'PATHSNAP_CONTEXT_RADIUS' not in os.environ:
                CONTEXT_RADIUS = int(metadata.get('context_radius', CONTEXT_RADIUS))
        expected = getattr(model, 'n_features_in_', None)
        expected_by_context = 9 if CONTEXT_RADIUS == 0 else 9 + (4 if CONTEXT_RADIUS == 1 else 8)
        if expected is not None and expected != expected_by_context:
            raise ValueError(f'Model has {expected} input features; configured pipeline creates {expected_by_context}')
        log.info('Loaded model from %s (candidates=%s, context_radius=%s)',
                 MODEL_PATH, CANDIDATE_COUNT, CONTEXT_RADIUS)
    except Exception as exc:
        model = None
        model_error = f'{type(exc).__name__}: {exc}'
        log.exception('Could not load PathSnap model')


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title='PathSnap Route Matcher', version='2.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=['GET', 'POST'], allow_headers=['*'])


def _validate_rows(name: str, rows: list[list[float]], min_cols: int,
                   max_cols: int, max_rows: int) -> None:
    if not rows:
        raise ValueError(f'{name} must not be empty')
    if len(rows) > max_rows:
        raise ValueError(f'{name} exceeds the limit of {max_rows} rows')
    for index, row in enumerate(rows):
        if not min_cols <= len(row) <= max_cols:
            raise ValueError(f'{name}[{index}] must have {min_cols}-{max_cols} values')
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f'{name}[{index}] contains a non-finite value')


class TripRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    trip_id: str | None = None
    track: list[list[float]] = Field(..., description='[longitude, latitude, optional timestamp]')
    nodes: list[list[float]] = Field(..., description='[longitude, latitude] rows')
    arcs: list[list[int]] = Field(..., description='[start_node_row, end_node_row] rows')
    candidate_count: int | None = Field(None, ge=1, le=64)
    transition_weight: float | None = Field(None, ge=0, le=10)

    @field_validator('track')
    @classmethod
    def validate_track(cls, value):
        _validate_rows('track', value, 2, 3, MAX_TRACK_POINTS)
        if len({len(row) for row in value}) != 1:
            raise ValueError('track rows must have the same number of columns')
        for index, (longitude, latitude, *_) in enumerate(value):
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError(f'track[{index}] has invalid longitude/latitude')
        return value

    @field_validator('nodes')
    @classmethod
    def validate_nodes(cls, value):
        _validate_rows('nodes', value, 2, 2, MAX_NODES)
        for index, (longitude, latitude) in enumerate(value):
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError(f'nodes[{index}] has invalid longitude/latitude')
        return value

    @field_validator('arcs')
    @classmethod
    def validate_arcs(cls, value):
        _validate_rows('arcs', value, 2, 2, MAX_ARCS)
        return value


@app.exception_handler(Exception)
async def unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
    log.exception('Unhandled API error', exc_info=exc)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error while matching the trip'})


def _require_model() -> None:
    if model is None:
        raise HTTPException(status_code=503, detail=f'Model unavailable: {model_error or MODEL_PATH}')


def _decode(trip: Trip, candidate_count: int | None, transition_weight: float | None) -> dict[str, Any]:
    count = CANDIDATE_COUNT if candidate_count is None else candidate_count
    weight = TRANSITION_WEIGHT if transition_weight is None else transition_weight
    try:
        return decode(trip, model, count, weight, context_radius=CONTEXT_RADIUS)
    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get('/health')
def health() -> dict[str, Any]:
    return {'status': 'ok', 'model_loaded': model is not None,
            'model_path': str(MODEL_PATH), 'model_error': model_error,
            'candidate_count': CANDIDATE_COUNT, 'transition_weight': TRANSITION_WEIGHT,
            'context_radius': CONTEXT_RADIUS, 'evaluator_split': EVALUATOR_SPLIT}


@app.get('/trips')
def list_evaluation_trips() -> dict[str, Any]:
    root = DATA_ROOT / EVALUATOR_SPLIT
    ids = sorted(path.name for path in root.iterdir() if path.is_dir()
                 and path.name.isdigit() and len(path.name) == 8) if root.is_dir() else []
    return {'split': EVALUATOR_SPLIT, 'trips': ids}


@app.post('/predict')
def predict(request: TripRequest) -> dict[str, Any]:
    _require_model()
    try:
        trip = Trip.from_payload(request.model_dump(exclude={'candidate_count', 'transition_weight'}))
    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _decode(trip, request.candidate_count, request.transition_weight)


@app.post('/trips/{trip_id}/predict')
def predict_dataset_trip(trip_id: str, candidate_count: int | None = None,
                         transition_weight: float | None = None) -> dict[str, Any]:
    _require_model()
    if not trip_id.isdigit() or len(trip_id) != 8:
        raise HTTPException(status_code=400, detail='trip_id must be an 8-digit dataset ID')
    folder = DATA_ROOT / EVALUATOR_SPLIT / trip_id
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f'Trip {trip_id} was not found')
    try:
        trip = Trip.from_directory(folder, include_route=False)
    except (ValueError, FileNotFoundError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _decode(trip, candidate_count, transition_weight)


@app.get('/trips/{trip_id}/map', response_class=HTMLResponse)
def dataset_trip_map(trip_id: str, candidate_count: int | None = None,
                     transition_weight: float | None = None) -> HTMLResponse:
    _require_model()
    if not trip_id.isdigit() or len(trip_id) != 8:
        raise HTTPException(status_code=400, detail='trip_id must be an 8-digit dataset ID')
    folder = DATA_ROOT / EVALUATOR_SPLIT / trip_id
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f'Trip {trip_id} was not found')
    try:
        trip = Trip.from_directory(folder, include_route=False)
    except (ValueError, FileNotFoundError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = _decode(trip, candidate_count, transition_weight)
    from viz import build_map
    return HTMLResponse(build_map(trip, result, reference=None))


@app.post('/map', response_class=HTMLResponse)
def uploaded_trip_map(request: TripRequest) -> HTMLResponse:
    _require_model()
    try:
        trip = Trip.from_payload(request.model_dump(exclude={'candidate_count', 'transition_weight'}))
    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = _decode(trip, request.candidate_count, request.transition_weight)
    from viz import build_map
    return HTMLResponse(build_map(trip, result, reference=None))
