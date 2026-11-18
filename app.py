"""FastAPI service for PathSnap route reconstruction."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pathsnap import Trip, decode

MODEL_PATH = Path(os.environ.get('PATHSNAP_MODEL', 'artifacts/candidate_model.pkl'))
DATA_ROOT = Path(os.environ.get('PATHSNAP_DATA', 'Student folder'))
EVALUATOR_SPLIT = os.environ.get('PATHSNAP_EVALUATOR_SPLIT', 'evaluator_test')

app = FastAPI(title='PathSnap Route Matcher', version='1.0.0')
model: Any | None = None


class TripRequest(BaseModel):
    trip_id: str | None = None
    track: list[list[float]] = Field(..., description='Rows: [longitude, latitude, optional timestamp]')
    nodes: list[list[float]] = Field(..., description='Rows: [longitude, latitude]')
    arcs: list[list[int]] = Field(..., description='Rows: [start_node_row, end_node_row]')
    candidate_count: int = Field(12, ge=1, le=64)
    transition_weight: float = Field(1.5, ge=0, le=10)


@app.on_event('startup')
def load_model() -> None:
    global model
    if MODEL_PATH.is_file():
        with MODEL_PATH.open('rb') as stream:
            model = pickle.load(stream)


@app.get('/health')
def health() -> dict[str, Any]:
    return {'status': 'ok', 'model_loaded': model is not None, 'model_path': str(MODEL_PATH)}


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
def predict_dataset_trip(trip_id: str, candidate_count: int = 12) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail=f'Model unavailable. Train it first: {MODEL_PATH}')

    if not trip_id.isdigit() or len(trip_id) != 8:
        raise HTTPException(status_code=400, detail='trip_id must be an 8-digit dataset ID')
    folder = DATA_ROOT / EVALUATOR_SPLIT / trip_id
    if folder.is_dir():
        try:

            trip = Trip.from_directory(folder, include_route=False)
            return decode(trip, model, candidate_count)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=404, detail=f'Trip {trip_id} was not found')
