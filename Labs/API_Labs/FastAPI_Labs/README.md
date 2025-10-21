# 🍷 Wine Classifier API – FastAPI Lab

## 📘 Overview
This project exposes a **Machine Learning model as an API** using **FastAPI** and **Uvicorn**.  
The API serves predictions from a trained **RandomForestClassifier** on the **Wine dataset** (from `scikit-learn`) and provides additional endpoints for retraining, health checking, and model versioning.

Unlike the original Iris dataset lab, this version introduces enhancements such as:
- Confidence scores for predictions
- Model retraining via API
- Metadata and version tracking
- Logging and error handling
- Clean modular structure (`data.py`, `train.py`, `predict.py`, `main.py`)


## 🧠 Objective
To demonstrate how an ML model can be:
1. **Trained and serialized** as an artifact (`.pkl` file)
2. **Served through FastAPI** as a prediction service
3. **Extended** with endpoints for retraining, confidence scoring, and health checks


## 🧩 Key Features
| Feature | Description |
|----------|--------------|
| **Dataset** | Uses `sklearn.datasets.load_wine` (13 numeric features, 3 classes) |
| **Model** | `RandomForestClassifier` with `StandardScaler` in a `Pipeline` |
| **Confidence Score** | Returns probability of the predicted class (`0.0–1.0`) |
| **Versioning** | Each model has a UTC-based version string (e.g., `2025-10-20T23-09-30`) |
| **Endpoints** | `/predict`, `/train`, `/health`, `/` (root metadata) |
| **Logging** | Structured logs for both training and predictions saved under `logs/` |
| **Error Handling** | Returns clear HTTP status codes and messages |
| **Pydantic Models** | Used for request validation and response formatting |


## 🗂️ Project Structure
```

mlops_labs/
└── fastapi_lab1/
├── model/
│   ├── wine_model_2025-10-20T23-27-44.pkl
│   └── metadata.json
├── logs/
│   ├── train.log
│   └── app.log
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   ├── predict.py
│   └── main.py
├── requirements.txt
└── README.md

````


## ⚙️ Setup Instructions

### 1️⃣ Create and activate a virtual environment
```bash
python -m venv fastapi_lab1_env
source fastapi_lab1_env/bin/activate   # Mac/Linux
fastapi_lab1_env\Scripts\activate      # Windows
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model

```bash
cd src
python train.py
```

This will:

* Train a `RandomForestClassifier`
* Save the model under `model/`
* Generate `metadata.json` with version, metrics, and class names
* Log training details in `logs/train.log`

## 🚀 Running the API Server

Run from the project root (recommended):

```bash
cd src
uvicorn main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```


## 🧪 API Endpoints

### **1️⃣ Root Endpoint**

**`GET /`**

> Displays available routes and API metadata.

Example:

```json
{
  "message": "Welcome to Wine Classifier API",
  "docs": "/docs",
  "health": "/health",
  "predict": "/predict",
  "train": "/train"
}
```


### **2️⃣ Health Check**

**`GET /health`**

> Confirms whether the model is loaded and provides version info.

Response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "2025-10-20T23-09-30"
}
```


### **3️⃣ Prediction Endpoint**

**`POST /predict`**

> Predicts wine class from 13 numerical features and returns confidence.

#### Request Body

```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.4,
  "proline": 1050
}
```

#### Example Response

```json
{
  "predicted_class_id": 0,
  "predicted_class_name": "class_0",
  "confidence": 0.98,
  "model_version": "2025-10-20T23-09-30"
}
```

> **Confidence Score:**
> Represents the model’s probability for the predicted class (range 0–1).
> A higher value = stronger certainty.


### **4️⃣ Retraining Endpoint**

**`POST /train`**

> Triggers model retraining and updates stored artifacts.

Response:

```json
{
  "status": "ok",
  "model_version": "2025-10-20T23-10-15",
  "metrics": {
    "accuracy": 1.0,
    "precision_weighted": 1.0,
    "recall_weighted": 1.0,
    "f1_weighted": 1.0
  },
  "saved_artifacts": {
    "model_path": "model/wine_model_2025-10-20T23-10-15.pkl",
    "metadata_path": "model/metadata.json"
  }
}
```


## 🧰 Pydantic Data Models

| Model             | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `WineData`        | Defines structure & validation for incoming prediction data |
| `PredictResponse` | Formats API prediction output                               |
| `TrainResponse`   | Formats retraining results                                  |
| `HealthResponse`  | Used for `/health` endpoint                                 |


## 📊 Metrics & Logging

| Log File         | Description                                   |
| ---------------- | --------------------------------------------- |
| `logs/train.log` | Model training progress, metrics, saved files |
| `logs/app.log`   | API-level logs for predictions and errors     |

All logs include timestamps and severity levels for traceability.


## 🧠 How Confidence Score Works

The **confidence score** represents the **maximum predicted probability** from the model’s `predict_proba()` output:

```python
proba = model.predict_proba(X).max()
```

This value indicates how certain the model is about its classification:

* `>0.9` → Very confident
* `0.6–0.9` → Moderately confident
* `<0.6` → Low confidence (consider review)

Including confidence scores makes the API more **interpretable**, **transparent**, and **ready for real-world integration**.


## 🧾 Example curl Command

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.2,
    "malic_acid": 1.78,
    "ash": 2.14,
    "alcalinity_of_ash": 11.2,
    "magnesium": 100,
    "total_phenols": 2.65,
    "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26,
    "proanthocyanins": 1.28,
    "color_intensity": 4.38,
    "hue": 1.05,
    "od280_od315_of_diluted_wines": 3.4,
    "proline": 1050
  }'
```

## 🧰 Requirements

```text
fastapi
uvicorn
scikit-learn
joblib
numpy
pydantic
```

## 🏁 Summary

This FastAPI lab demonstrates how to build a **production-like ML inference service** that not only predicts but also provides:

* Versioning
* Confidence scores
* Retraining and health monitoring
* Logging and modular architecture
