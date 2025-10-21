from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


# Input schema for Wine dataset (13 features)
# Names match sklearn's load_wine().feature_names
class WineData(BaseModel):
    alcohol: float = Field(..., ge=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315_of_diluted_wines: float = Field(..., ge=0)
    proline: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class TrainResponse(BaseModel):
    status: str
    model_version: str
    metrics: Dict[str, Any]
    saved_artifacts: Dict[str, str]
