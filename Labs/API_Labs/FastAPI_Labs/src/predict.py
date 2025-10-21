import json
import logging
import os
from functools import lru_cache
from typing import Tuple

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException

from data import WineData, PredictResponse, HealthResponse

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

router = APIRouter()


def _metadata_path() -> str:
    return os.path.join(MODEL_DIR, "metadata.json")


def _load_metadata() -> dict:
    meta_file = _metadata_path()
    if not os.path.exists(meta_file):
        raise FileNotFoundError("metadata.json not found. Train a model first.")
    with open(meta_file) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_model_and_meta():
    meta = _load_metadata()
    model_path = meta.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing at {model_path}. Train a model first.")
    model = joblib.load(model_path)
    class_names = meta.get("class_names", [str(i) for i in range(3)])
    version = meta.get("version", "unknown")
    return model, class_names, version


def _vectorize(payload: WineData, feature_order: list) -> np.ndarray:
    # Ensure input order matches training
    x = np.array(
        [
            payload.alcohol,
            payload.malic_acid,
            payload.ash,
            payload.alcalinity_of_ash,
            payload.magnesium,
            payload.total_phenols,
            payload.flavanoids,
            payload.nonflavanoid_phenols,
            payload.proanthocyanins,
            payload.color_intensity,
            payload.hue,
            payload.od280_od315_of_diluted_wines,
            payload.proline,
        ],
        dtype=float,
    ).reshape(1, -1)
    return x


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        _, _, version = _load_model_and_meta()
        return HealthResponse(status="ok", model_loaded=True, model_version=version)
    except Exception as e:
        logging.exception("Health check failed.")
        return HealthResponse(status=f"error: {e}", model_loaded=False, model_version=None)


@router.post("/predict", response_model=PredictResponse)
def predict(payload: WineData) -> PredictResponse:
    try:
        model, class_names, version = _load_model_and_meta()
        # Infer feature order from metadata if present
        meta = _load_metadata()
        feature_order = meta.get("features")
        x = _vectorize(payload, feature_order)
        pred = model.predict(x)[0]
        proba = float(np.max(model.predict_proba(x)))
        class_name = class_names[int(pred)]
        logging.info(f"/predict -> class={class_name}, confidence={proba:.4f}")
        return PredictResponse(
            predicted_class_id=int(pred),
            predicted_class_name=class_name,
            confidence=proba,
            model_version=version,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logging.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
