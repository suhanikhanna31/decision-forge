"""
FastAPI endpoints for AI-enhanced DecisionForge (Ollama version)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from app.config_loader import load_config
from app.ai.ai_enhanced_engine import AIEnhancedDecisionEngine
from app.ai.nl_query_interface import NLQueryInterface

# Global instances
ai_engine = None
nl_interface = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize AI components on startup."""
    global ai_engine, nl_interface
    
    config = load_config("configs/ecommerce.yaml")
    
    ai_engine = AIEnhancedDecisionEngine(
        config=config,
        enable_ai=True,
        model="llama3.2"
    )
    
    nl_interface = NLQueryInterface(ai_engine)
    
    print("✓ AI-enhanced DecisionForge initialized (Ollama)")
    
    yield
    
    ai_engine.clear_history()
    print("✓ Cleanup complete")


app = FastAPI(
    title="DecisionForge AI API",
    description="AI-enhanced decision engine with Ollama (FREE)",
    version="2.0.0",
    lifespan=lifespan
)


class UserInput(BaseModel):
    anomaly_score: float = Field(..., ge=0, le=1)
    request_count_today: int = Field(..., ge=0)
    expected_lift: float = Field(..., ge=0, le=1)
    revenue: float = Field(..., gt=0)
    incentive_cost: float = Field(..., ge=0)
    churn: Optional[float] = Field(None, ge=0, le=1)


class DecisionRequest(BaseModel):
    inputs: UserInput
    return_explanation: bool = True
    return_recommendations: bool = False


@app.get("/")
async def root():
    return {
        "name": "DecisionForge AI API",
        "version": "2.0.0",
        "ai_backend": "Ollama (FREE)",
        "model": "llama3.2"
    }


@app.post("/api/v1/decide")
async def make_decision(request: DecisionRequest):
    try:
        result = ai_engine.decide_with_explanation(
            inputs=request.inputs.model_dump(),
            return_explanation=request.return_explanation,
            return_recommendations=request.return_recommendations
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_statistics():
    return ai_engine.get_decision_history_summary()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ai_enabled": ai_engine.enable_ai,
        "backend": "Ollama",
        "model": "llama3.2"
    }
