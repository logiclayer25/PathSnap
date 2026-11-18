"""FastAPI service for PathSnap route reconstruction."""
from __future__ import annotations

import logging
import math
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from pathsnap import Trip, decode
from viz import build_map

log = logging.getLogger('pathsnap.api')

MODEL_PATH = Path(os.environ.get('PATHSNAP_MODEL', 'artifacts/models/histgb-20261118T070402Z.pkl'))
DATA_ROOT = Path(os.environ.get('PATHSNAP_DATA', 'Student folder'))
EVALUATOR_SPLIT = os.environ.get('PATHSNAP_EVALUATOR_SPLIT', 'evaluator_test')
CORS_ORIGINS = [o.strip() for o in os.environ.get('PATHSNAP_CORS_ORIGINS', '*').split(',') if o.strip()]
DEFAULT_CANDIDATE_COUNT = int(os.environ.get('PATHSNAP_CANDIDATE_COUNT', '24'))
DEFAULT_TRANSITION_WEIGHT = float(os.environ.get('PATHSNAP_TRANSITION_WEIGHT', '0'))
MAX_TRACK_POINTS = int(os.environ.get('PATHSNAP_MAX_TRACK_POINTS', '50000'))
MAX_NODES = int(os.environ.get('PATHSNAP_MAX_NODES', '500000'))
MAX_ARCS = int(os.environ.get('PATHSNAP_MAX_ARCS', '1000000'))

model: Any | None = None
model_error: str | None = None


def _load_model() -> None:
    global model, model_error
    if not MODEL_PATH.is_file():
        log.warning('Model file not found: %s', MODEL_PATH)
        return
    try:
        with MODEL_PATH.open('rb') as stream:
            model = pickle.load(stream)  # trusted local artifact only
        log.info('Loaded model from %s', MODEL_PATH)
    except Exception as exc:  # e.g. scikit-learn version mismatch; keep the server up and report it
        model_error = f'{type(exc).__name__}: {exc}'
        log.error('Could not load model %s: %s (expected scikit-learn==1.7.2)', MODEL_PATH, model_error)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title='PathSnap Route Matcher', version='1.1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=['GET', 'POST'], allow_headers=['*'])


def _check_rows(name: str, rows: list[list[float]], min_cols: int, max_cols: int, max_rows: int) -> None:
    if not rows:
        raise ValueError(f'{name} must not be empty')
    if len(rows) > max_rows:
        raise ValueError(f'{name} has {len(rows)} rows; maximum is {max_rows}')
    for i, row in enumerate(rows):
        if not (min_cols <= len(row) <= max_cols):
            raise ValueError(f'{name}[{i}] must have {min_cols}-{max_cols} values, got {len(row)}')
        if not all(math.isfinite(v) for v in row):
            raise ValueError(f'{name}[{i}] contains a non-finite value')


class TripRequest(BaseModel):
    trip_id: str | None = None
    track: list[list[float]] = Field(..., description='Rows: [longitude, latitude, optional timestamp]')
    nodes: list[list[float]] = Field(..., description='Rows: [longitude, latitude]')
    arcs: list[list[int]] = Field(..., description='Rows: [start_node_row, end_node_row]')
    candidate_count: int = Field(DEFAULT_CANDIDATE_COUNT, ge=1, le=64)
    transition_weight: float = Field(DEFAULT_TRANSITION_WEIGHT, ge=0, le=10)

    @field_validator('track')
    @classmethod
    def _track(cls, v):
        _check_rows('track', v, 2, 3, MAX_TRACK_POINTS)
        if len({len(r) for r in v}) != 1:
            raise ValueError('track rows must all have the same number of columns')
        for i, (lon, lat, *_) in enumerate(v):
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f'track[{i}] is outside valid lon/lat range (order is [longitude, latitude])')
        return v

    @field_validator('nodes')
    @classmethod
    def _nodes(cls, v):
        _check_rows('nodes', v, 2, 2, MAX_NODES)
        for i, (lon, lat) in enumerate(v):
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f'nodes[{i}] is outside valid lon/lat range (order is [longitude, latitude])')
        return v

    @field_validator('arcs')
    @classmethod
    def _arcs(cls, v):
        _check_rows('arcs', v, 2, 2, MAX_ARCS)
        return v


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception('Unhandled error')
    return JSONResponse(status_code=500, content={'detail': 'Internal server error while matching the trip'})


@app.get('/health')
def health() -> dict[str, Any]:
    return {'status': 'ok', 'model_loaded': model is not None, 'model_path': str(MODEL_PATH), 'model_error': model_error,
            'candidate_count': DEFAULT_CANDIDATE_COUNT,
            'transition_weight': DEFAULT_TRANSITION_WEIGHT,
            'evaluator_split': EVALUATOR_SPLIT}


def _require_model() -> None:
    if model is None:
        raise HTTPException(status_code=503, detail=f'Model unavailable ({model_error or "file not found"}): {MODEL_PATH}')


@app.post('/predict')
def predict(request: TripRequest) -> dict[str, Any]:
    _require_model()
    try:
        payload = request.model_dump()
        payload.pop('route', None)  # labels are never accepted from clients
        trip = Trip.from_payload(payload)
        return decode(trip, model, request.candidate_count, request.transition_weight)
    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/trips/{trip_id}/predict')
def predict_dataset_trip(trip_id: str, candidate_count: int = DEFAULT_CANDIDATE_COUNT,
                         transition_weight: float = DEFAULT_TRANSITION_WEIGHT) -> dict[str, Any]:
    _require_model()
    if not trip_id.isdigit() or len(trip_id) != 8:
        raise HTTPException(status_code=400, detail='trip_id must be an 8-digit dataset ID')
    if not 1 <= candidate_count <= 64:
        raise HTTPException(status_code=422, detail='candidate_count must be between 1 and 64')
    if not 0 <= transition_weight <= 10:
        raise HTTPException(status_code=422, detail='transition_weight must be between 0 and 10')
    folder = DATA_ROOT / EVALUATOR_SPLIT / trip_id
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f'Trip {trip_id} was not found')
    try:
        trip = Trip.from_directory(folder, include_route=False)
        return decode(trip, model, candidate_count, transition_weight)
    except (ValueError, FileNotFoundError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get('/trips')
def list_trips() -> dict[str, Any]:
    """IDs of the evaluation trips available on this server."""
    root = DATA_ROOT / EVALUATOR_SPLIT
    ids = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8) \
        if root.is_dir() else []
    return {'split': EVALUATOR_SPLIT, 'trips': ids,
            **({} if ids else {'note': f'No trip folders found in {root}'})}


@app.get('/trips/{trip_id}/map', response_class=HTMLResponse)
def trip_map(trip_id: str, candidate_count: int = DEFAULT_CANDIDATE_COUNT,
             transition_weight: float = DEFAULT_TRANSITION_WEIGHT) -> HTMLResponse:
    """Folium map of GPS, nearby roads and predicted route (+ reference route if the files include it)."""
    _require_model()
    if not trip_id.isdigit() or len(trip_id) != 8:
        raise HTTPException(status_code=400, detail='trip_id must be an 8-digit dataset ID')
    if not 1 <= candidate_count <= 64 or not 0 <= transition_weight <= 10:
        raise HTTPException(status_code=422, detail='candidate_count or transition_weight out of range')
    folder = DATA_ROOT / EVALUATOR_SPLIT / trip_id
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f'Trip {trip_id} was not found')
    try:
        trip = Trip.from_directory(folder, include_route=False)
        result = decode(trip, model, candidate_count, transition_weight)
        reference = Trip.from_directory(folder, include_route=True).route
        return HTMLResponse(build_map(trip, result, reference))
    except (ValueError, FileNotFoundError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/map', response_class=HTMLResponse)
def uploaded_map(request: TripRequest) -> HTMLResponse:
    """Folium map for an uploaded trip (no reference route)."""
    _require_model()
    try:
        trip = Trip.from_payload(request.model_dump())
        result = decode(trip, model, request.candidate_count, request.transition_weight)
        return HTMLResponse(build_map(trip, result, None))
    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
