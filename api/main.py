"""
FastAPI REST API for Hotel Booking Predictions
CRISP-DM Phase 6: Deployment
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classification_model import CancellationClassifier
from time_series_model import DemandForecaster

app = FastAPI(
    title="Hotel Booking Prediction API",
    description="ML API for cancellation prediction and demand forecasting",
    version="1.0.0",
)

# Load models on startup
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
classifier = None
forecaster = None


@app.on_event("startup")
async def load_models():
    """Load trained models on startup."""
    global classifier, forecaster

    clf_path = os.path.join(MODELS_DIR, "cancellation_model.pkl")
    ts_path = os.path.join(MODELS_DIR, "demand_model.pkl")

    if os.path.exists(clf_path):
        try:
            classifier = CancellationClassifier.load(clf_path)
            print("Classification model loaded")
        except Exception as e:
            print(f"Warning: Failed to load classification model: {e}")
    else:
        print(f"Warning: Classification model not found at {clf_path}")

    if os.path.exists(ts_path):
        try:
            forecaster = DemandForecaster.load(ts_path)
            print("Demand forecaster loaded")
        except Exception as e:
            print(f"Warning: Failed to load demand model: {e}")
            print("Time series forecasting will be unavailable")
    else:
        print(f"Warning: Demand model not found at {ts_path}")


# Request/Response Models
class BookingFeatures(BaseModel):
    """Input features for cancellation prediction."""

    lead_time: int
    arrival_date_week_number: int
    arrival_month_num: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int = 0
    babies: int = 0
    is_repeated_guest: int = 0
    previous_cancellations: int = 0
    previous_bookings_not_canceled: int = 0
    booking_changes: int = 0
    days_in_waiting_list: int = 0
    adr: float
    required_car_parking_spaces: int = 0
    total_of_special_requests: int = 0
    hotel: int = 0  # 0=Resort, 1=City
    meal: int = 0
    market_segment: int = 0
    distribution_channel: int = 0
    deposit_type: int = 0
    customer_type: int = 0
    season: int = 0


class CancellationPrediction(BaseModel):
    """Cancellation prediction response."""

    will_cancel: bool
    cancellation_probability: float
    confidence: str


class DemandForecast(BaseModel):
    """Demand forecast for a single day."""

    date: str
    predicted_bookings: float
    lower_bound: float
    upper_bound: float


class DemandForecastResponse(BaseModel):
    """Demand forecast response."""

    forecasts: List[DemandForecast]
    model_type: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: dict


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return {
        "status": "healthy",
        "models_loaded": {
            "classification": classifier is not None,
            "time_series": forecaster is not None,
        },
    }


@app.post("/predict/cancellation", response_model=CancellationPrediction)
async def predict_cancellation(booking: BookingFeatures):
    """
    Predict whether a booking will be cancelled.

    Returns cancellation probability and prediction.
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classification model not loaded. Train the model first.",
        )

    # Prepare features
    features = booking.model_dump()

    # Add derived features
    features["total_nights"] = (
        features["stays_in_weekend_nights"] + features["stays_in_week_nights"]
    )
    features["total_guests"] = (
        features["adults"] + features["children"] + features["babies"]
    )
    features["is_weekend_stay"] = 1 if features["stays_in_weekend_nights"] > 0 else 0

    # Create DataFrame
    X = pd.DataFrame([features])

    # Ensure columns match training
    for col in classifier.feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[classifier.feature_names]

    # Predict
    probability = float(classifier.predict_proba(X)[0])
    will_cancel = probability > 0.5

    # Determine confidence
    if probability > 0.8 or probability < 0.2:
        confidence = "high"
    elif probability > 0.6 or probability < 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "will_cancel": will_cancel,
        "cancellation_probability": round(probability, 4),
        "confidence": confidence,
    }


@app.get("/predict/demand", response_model=DemandForecastResponse)
async def predict_demand(days: int = 30):
    """
    Forecast booking demand for the next N days.

    Args:
        days: Number of days to forecast (default: 30, max: 90)
    """
    if forecaster is None:
        raise HTTPException(
            status_code=503, detail="Demand model not loaded. Train the model first."
        )

    if days < 1 or days > 90:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 90")

    # Generate forecast
    forecast_df = forecaster.predict(periods=days)

    forecasts = []
    for _, row in forecast_df.iterrows():
        forecasts.append(
            {
                "date": row["ds"].strftime("%Y-%m-%d"),
                "predicted_bookings": round(float(row["yhat"]), 2),
                "lower_bound": round(float(row["yhat_lower"]), 2),
                "upper_bound": round(float(row["yhat_upper"]), 2),
            }
        )

    return {"forecasts": forecasts, "model_type": forecaster.model_type}


@app.post("/predict/cancellation/batch")
async def predict_cancellation_batch(bookings: List[BookingFeatures]):
    """Predict cancellation for multiple bookings."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")

    results = []
    for booking in bookings:
        # Reuse single prediction logic
        pred = await predict_cancellation(booking)
        results.append(pred)

    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
