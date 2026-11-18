"""FastAPI service for PathSnap route reconstruction."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pathsnap import Trip, decode

MODEL_PATH = Path(os.environ.get('PATHSNAP_MODEL', 'artifacts/models/histgb-20261118T070402Z.pkl'))
DATA_ROOT = Path(os.environ.get('PATHSNAP_DATA', 'Student folder'))
EVALUATOR_SPLIT = os.environ.get('PATHSNAP_EVALUATOR_SPLIT', 'evaluator_test')
CORS_ORIGINS = [origin.strip() for origin in os.environ.get('PATHSNAP_CORS_ORIGINS', '*').split(',') if origin.strip()]
DEFAULT_CANDIDATE_COUNT = int(os.environ.get('PATHSNAP_CANDIDATE_COUNT', '24'))
DEFAULT_TRANSITION_WEIGHT = float(os.environ.get('PATHSNAP_TRANSITION_WEIGHT', '0'))

app = FastAPI(title='PathSnap Route Matcher', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=['GET', 'POST'], allow_headers=['*'])
model: Any | None = None


class TripRequest(BaseModel):
    trip_id: str | None = None
    track: list[list[float]] = Field(..., description='Rows: [longitude, latitude, optional timestamp]')
    nodes: list[list[float]] = Field(..., description='Rows: [longitude, latitude]')
    arcs: list[list[int]] = Field(..., description='Rows: [start_node_row, end_node_row]')
    candidate_count: int = Field(DEFAULT_CANDIDATE_COUNT, ge=1, le=64)
    transition_weight: float = Field(DEFAULT_TRANSITION_WEIGHT, ge=0, le=10)


@app.on_event('startup')
def load_model() -> None:
    global model
    if MODEL_PATH.is_file():
        with MODEL_PATH.open('rb') as stream:
            model = pickle.load(stream)


@app.get('/health')
def health() -> dict[str, Any]:
    return {'status': 'ok', 'model_loaded': model is not None, 'model_path': str(MODEL_PATH),
            'candidate_count': DEFAULT_CANDIDATE_COUNT,
            'transition_weight': DEFAULT_TRANSITION_WEIGHT,
            'evaluator_split': EVALUATOR_SPLIT}


@app.post('/predict')
def predict(request: TripRequest) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail=f'Model unavailable. Train it first: {MODEL_PATH}')
    try:
        trip = Trip.from_payload(request.model_dump())
        result = decode(trip, model, request.candidate_count, request.transition_weight)
        return result
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post('/trips/{trip_id}/predict')
def predict_dataset_trip(trip_id: str, candidate_count: int = DEFAULT_CANDIDATE_COUNT,
                         transition_weight: float = DEFAULT_TRANSITION_WEIGHT) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail=f'Model unavailable. Train it first: {MODEL_PATH}')

    if not trip_id.isdigit() or len(trip_id) != 8:
        raise HTTPException(status_code=400, detail='trip_id must be an 8-digit dataset ID')
    folder = DATA_ROOT / EVALUATOR_SPLIT / trip_id
    if folder.is_dir():
        try:

            trip = Trip.from_directory(folder, include_route=False)
            return decode(trip, model, candidate_count, transition_weight)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=404, detail=f'Trip {trip_id} was not found')
