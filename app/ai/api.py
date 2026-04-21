"""
DecisionForge REST API
----------------------
FastAPI application exposing decision engine as a real-time REST service.

Endpoints:
  GET  /health             — service health check
  POST /api/v1/decide      — make a single decision
  POST /api/v1/decide/batch — make decisions for multiple users
  GET  /api/v1/metrics     — model evaluation metrics + latency profile

Run locally:
    uvicorn app.ai.api:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import time

from app.config_loader import load_config
from app.core.decision_engine import DecisionEngine
from app.ml.churn_model import ChurnModel
from app.ml.anomaly_model import AnomalyModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DecisionForge API",
    description=(
        "Real-time ML decision engine for churn intervention and anomaly detection. "
        "Converts ML predictions into ROI-justified business actions."
    ),
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Global model state (initialized at startup)
# ---------------------------------------------------------------------------

_engine: Optional[DecisionEngine] = None
_churn_model: Optional[ChurnModel] = None
_anomaly_model: Optional[AnomalyModel] = None


@app.on_event("startup")
def startup_event():
    """Train models on startup using default simulation data."""
    global _engine, _churn_model, _anomaly_model

    config = load_config("configs/ecommerce.yaml")

    _churn_model = ChurnModel()
    _anomaly_model = AnomalyModel()

    # Simulation training data
    churn_data = pd.DataFrame({
        "tenure": [1, 5, 10, 2, 7, 3, 8, 1, 6, 4],
        "monthly_charges": [200, 150, 100, 220, 130, 210, 120, 195, 145, 175],
        "churn": [1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    })
    anomaly_data = pd.DataFrame({
        "request_count_today": [1, 2, 1, 10, 2, 1, 15, 1, 3, 1],
        "login_attempts": [1, 1, 1, 7, 1, 1, 10, 1, 2, 1],
    })

    _churn_model.train(churn_data, target="churn")
    _anomaly_model.train(anomaly_data)
    _engine = DecisionEngine(config)

    print("[DecisionForge] Models trained and engine ready.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class UserFeatures(BaseModel):
    tenure: float = Field(..., description="Months the user has been a customer", example=3)
    monthly_charges: float = Field(..., description="Monthly billing amount in USD", example=210)
    request_count_today: int = Field(..., description="Number of API/page requests today", example=2)
    login_attempts: int = Field(default=1, description="Login attempts today", example=1)


class DecisionRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user identifier for audit logging")
    features: UserFeatures


class DecisionResponse(BaseModel):
    user_id: Optional[str]
    decision: str = Field(..., description="INTERVENE | DO_NOTHING | FLAG")
    reason: str
    expected_value: float
    churn_probability: float
    anomaly_score: float
    latency_ms: float


class BatchDecisionRequest(BaseModel):
    users: list[DecisionRequest]


class MetricsResponse(BaseModel):
    churn_model_metrics: dict
    anomaly_model_metrics: dict
    latency_profile: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """
    Health check endpoint.

    Returns service status and whether models are loaded and ready.
    """
    return {
        "status": "healthy",
        "models_loaded": _engine is not None,
        "version": "2.0.0",
    }


@app.post("/api/v1/decide", response_model=DecisionResponse, tags=["Decision"])
def make_decision(request: DecisionRequest):
    """
    Make a single real-time decision for one user.

    The engine evaluates:
    1. Security/anomaly risk — blocks or flags suspicious activity
    2. ROI justification — only intervenes if economically positive
    3. Returns decision with reasoning and latency metrics
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Models not yet initialized.")

    f = request.features

    # Run ML models
    churn_prob = _churn_model.predict_proba({
        "tenure": f.tenure,
        "monthly_charges": f.monthly_charges,
    })
    expected_lift = churn_prob * 0.3
    anomaly_score = _anomaly_model.score({
        "request_count_today": f.request_count_today,
        "login_attempts": f.login_attempts,
    })

    # Run decision engine
    result = _engine.decide({
        "churn_probability": churn_prob,
        "expected_lift": expected_lift,
        "anomaly_score": anomaly_score,
        "request_count_today": f.request_count_today,
    })

    return DecisionResponse(
        user_id=request.user_id,
        decision=result["decision"],
        reason=result["reason"],
        expected_value=result.get("expected_value", 0.0),
        churn_probability=round(churn_prob, 4),
        anomaly_score=round(anomaly_score, 4),
        latency_ms=result.get("latency_ms", 0.0),
    )


@app.post("/api/v1/decide/batch", tags=["Decision"])
def make_batch_decisions(request: BatchDecisionRequest):
    """
    Process multiple users in one request.

    Useful for batch scoring pipelines or A/B test evaluation.
    Returns a list of decisions in the same order as the input users.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Models not yet initialized.")

    results = []
    for user_req in request.users:
        result = make_decision(user_req)
        results.append(result)
    return {"decisions": results, "count": len(results)}


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Evaluation"])
def get_metrics():
    """
    Return model evaluation metrics (Precision, Recall, F1, AUC) and latency profile.

    Churn model metrics come from offline validation at startup.
    Anomaly model metrics include contamination and score distribution.
    Latency profile documents system design trade-offs.
    """
    if _churn_model is None:
        raise HTTPException(status_code=503, detail="Models not yet initialized.")

    return MetricsResponse(
        churn_model_metrics=_churn_model.get_metrics(),
        anomaly_model_metrics=_anomaly_model.get_metrics(),
        latency_profile=_engine.get_latency_profile(),
    )