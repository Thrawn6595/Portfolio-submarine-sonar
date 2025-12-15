from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import joblib
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Sonar Classification API",
    description="Mine vs Rock classification using sonar signals",
    version="1.0.0"
)

model = None
MODEL_PATH = Path(__file__).parent.parent / "trained_model.pkl"

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_items=60, max_items=60)
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.02] * 60
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str

@app.on_event("startup")
async def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/")
def root():
    return {"message": "Sonar Classification API", "status": "running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(features)[0][prediction])
        else:
            proba = 1.0
        
        label = "Mine" if prediction == 1 else "Rock"
        
        return PredictionResponse(
            prediction=prediction,
            probability=proba,
            label=label
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "input_features": 60,
        "output_classes": 2,
        "labels": {0: "Rock", 1: "Mine"}
    }
