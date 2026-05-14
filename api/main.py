"""FastAPI-сервис предсказания целевого действия.

Запуск:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.preprocessing import build_features, fill_nans
from api.schemas import HealthResponse, PredictionResponse, VisitRequest

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"

app = FastAPI(
    title="СберАвтоподписка - предсказание целевого действия",
    description="Принимает данные визита и возвращает вероятность того, "
    "что в визите будет совершено целевое действие.",
    version="1.0.0",
)


_state: dict = {"model": None, "threshold": 0.5}


@app.on_event("startup")
def _load_model() -> None:
    if not MODEL_PATH.exists():
        return
    artifact = joblib.load(MODEL_PATH)
    _state["model"] = artifact["pipeline"]
    _state["threshold"] = float(artifact.get("threshold", 0.5))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_state["model"] is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(visit: VisitRequest) -> PredictionResponse:
    if _state["model"] is None:
        raise HTTPException(
            status_code=503, detail=f"model not loaded; expected at {MODEL_PATH}"
        )

    raw = pd.DataFrame([visit.model_dump()])
    cleaned = fill_nans(raw)
    features = build_features(cleaned)

    proba = float(_state["model"].predict_proba(features)[0, 1])
    threshold = _state["threshold"]
    prediction = int(proba >= threshold)

    return PredictionResponse(
        session_id=visit.session_id,
        prediction=prediction,
        probability=proba,
        threshold=threshold,
    )
