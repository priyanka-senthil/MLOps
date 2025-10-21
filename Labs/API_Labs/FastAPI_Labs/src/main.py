from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predict import router as predict_router
from train import train_model
from data import TrainResponse

app = FastAPI(
    title="Wine Classifier API",
    description="FastAPI service for sklearn Wine dataset with model versioning, metrics, and probabilities.",
    version="1.0.0",
)

# Allow local dev tools / notebooks / Postman
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(predict_router)


@app.get("/", tags=["meta"])
def root():
    return {
        "message": "Welcome to Wine Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "train": "/train",
    }


@app.post("/train", response_model=TrainResponse, tags=["training"])
def train():
    """
    Retrain the model and update artifacts (model/*.pkl, metadata.json).
    """
    result = train_model()
    return {
        "status": "ok",
        "model_version": result["version"],
        "metrics": result["metrics"],
        "saved_artifacts": result["artifacts"],
    }
