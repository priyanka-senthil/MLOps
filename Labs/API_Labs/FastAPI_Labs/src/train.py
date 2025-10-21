import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

CLASS_NAMES = ["class_0", "class_1", "class_2"]  # wine target has 3 classes


def _current_version() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")



def train_model(random_state: int = 42, test_size: float = 0.2) -> Dict[str, Any]:
    logging.info("Loading wine dataset...")
    data = load_wine()
    X, y = data.data, data.target

    feature_names = list(data.feature_names)

    logging.info("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline: scale + RandomForest
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=250, random_state=random_state)),
        ]
    )

    logging.info("Training model...")
    pipe.fit(X_train, y_train)

    logging.info("Evaluating model...")
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    version = _current_version()

    # Save artifacts
    model_path = os.path.join(MODEL_DIR, f"wine_model_{version}.pkl")
    meta_path = os.path.join(MODEL_DIR, "metadata.json")

    logging.info(f"Saving model to {model_path} ...")
    joblib.dump(pipe, model_path)

    metadata = {
        "version": version,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "model_path": model_path,
        "dataset": "sklearn.load_wine",
        "features": feature_names,
        "class_names": CLASS_NAMES,
        "metrics": {
            "accuracy": acc,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "confusion_matrix": cm,
        },
    }

    logging.info(f"Writing metadata to {meta_path} ...")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Training complete.")
    return {
        "version": version,
        "metrics": metadata["metrics"],
        "artifacts": {"model_path": model_path, "metadata_path": meta_path},
    }


if __name__ == "__main__":
    result = train_model()
    print(
        json.dumps(
            {
                "status": "ok",
                "model_version": result["version"],
                "metrics": result["metrics"],
                "saved_artifacts": result["artifacts"],
            },
            indent=2,
        )
    )
