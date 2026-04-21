# DecisionForge – Revenue & Risk-Aware Decision Engine with AI Explainability

DecisionForge is a production-oriented ML decision system that converts machine learning predictions into accountable business actions. Instead of acting directly on churn probability, it evaluates return on investment (ROI), cost, and security risk before deciding whether to intervene, do nothing, or flag behavior — and exposes everything through a well-documented REST API.

---

## The Problem

Most retention systems act directly on churn probability, overspend on incentives, ignore anomalous behavior, and lack explainability or auditability. This results in wasted retention spend and poor business decisions.

## The Solution

DecisionForge introduces a decision layer between prediction and action. Every action must be:
- **Economically justified** — ROI-positive before any intervention
- **Security-cleared** — anomaly and abuse checks run first
- **Explainable and auditable** — every decision is logged with inputs and rationale
- **Accessible via API** — REST endpoints with interactive Swagger documentation

---

## Decision Flow

```
User Features
    ↓
Feature Engineering & Preprocessing
• Derived features (charge_per_tenure, request_login_ratio, etc.)
• StandardScaler normalization
• Median imputation for missing values
    ↓
ML Scoring Layer
• Churn Prediction (Logistic Regression)
• Anomaly Detection (Isolation Forest)
    ↓
Model Evaluation (Offline Validation)
• Precision, Recall, F1-Score, ROC-AUC
• Exposed via /api/v1/metrics endpoint
    ↓
Decision Engine
• Security Gate (anomaly threshold check)
• ROI / Expected Value Calculation
• Latency tracking per decision (~1–5ms P99)
    ↓
Decision Output
• INTERVENE / DO_NOTHING / FLAG
• Reason + Expected Value + Latency (ms)
    ↓
Audit Log
• Timestamp, inputs, decision, rationale
```

---

## System Components

### ML Pipeline
- **FeaturePreprocessor** (`app/ml/preprocessor.py`) — structured feature engineering and scaling pipeline applied consistently at training and inference time
- **ChurnModel** (`app/ml/churn_model.py`) — Logistic Regression with Precision, Recall, F1, and ROC-AUC evaluation on a held-out validation split
- **AnomalyModel** (`app/ml/anomaly_model.py`) — Isolation Forest with supervised or unsupervised evaluation depending on label availability

### Core Decision Engine
- **DecisionEngine** (`app/core/decision_engine.py`) — routes predictions to business actions with per-request latency tracking and a documented latency vs accuracy trade-off profile
- **ROICalculator** (`app/core/roi.py`) — computes expected monetary value before any intervention is approved
- **SecurityGate** (`app/core/security.py`) — blocks or flags anomalous requests before spending resources
- **AuditLogger** (`app/core/audit.py`) — records every decision with inputs, output, and timestamp

### REST API
- **FastAPI app** (`app/ai/api.py`) — production-ready REST API with Pydantic request/response schemas, startup model training, and interactive Swagger UI

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check and model load status |
| POST | `/api/v1/decide` | Real-time decision for a single user |
| POST | `/api/v1/decide/batch` | Batch decisions for multiple users |
| GET | `/api/v1/metrics` | Model evaluation metrics + latency profile |

### Start the API
```bash
uvicorn app.ai.api:app --reload --port 8000
```

Interactive docs: **http://localhost:8000/docs**

### Example Request
```bash
curl -X POST "http://localhost:8000/api/v1/decide" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "features": {
      "tenure": 2,
      "monthly_charges": 210,
      "request_count_today": 1,
      "login_attempts": 1
    }
  }'
```

### Example Response
```json
{
  "user_id": "user_001",
  "decision": "INTERVENE",
  "reason": "Expected value $18.40 — ROI positive",
  "expected_value": 18.40,
  "churn_probability": 0.847,
  "anomaly_score": 0.12,
  "latency_ms": 1.3
}
```

### Metrics Endpoint Response
```json
{
  "churn_model_metrics": {
    "precision": 0.85,
    "recall": 0.80,
    "f1_score": 0.82,
    "roc_auc": 0.91
  },
  "anomaly_model_metrics": {
    "contamination": 0.2,
    "flagged_fraction": 0.2
  },
  "latency_profile": {
    "total_p99_ms": "~5ms",
    "trade_off_note": "Logistic Regression chosen over XGBoost: ~5-10% lower F1 but sub-2ms inference latency."
  }
}
```

---

## Feature Engineering

Raw features are transformed before training and inference by `FeaturePreprocessor`:

**Churn features:**
| Feature | Description |
|---------|-------------|
| `tenure` | Months as a customer (raw) |
| `monthly_charges` | Monthly billing amount (raw) |
| `charge_per_tenure` | `monthly_charges / (tenure + 1)` — cost relative to loyalty |
| `high_charge_flag` | Binary flag if `monthly_charges > 180` |

**Anomaly features:**
| Feature | Description |
|---------|-------------|
| `request_count_today` | Raw request volume |
| `login_attempts` | Raw login count |
| `request_login_ratio` | `requests / (logins + 1)` — bot-like behavior signal |
| `high_request_flag` | Binary flag if `requests > 5` |

All features are normalized with `StandardScaler`. Missing values are imputed with training-set medians at inference time, keeping latency consistent.

---

## Model Evaluation

Run offline evaluation to print Precision, Recall, F1, and ROC-AUC:

```bash
python evaluate_models.py
```

The same metrics are also available live via the `/api/v1/metrics` endpoint once the API is running.

---

## System Design Trade-offs

### Latency vs Accuracy
| Model | Inference Latency | F1 vs XGBoost |
|-------|------------------|----------------|
| Logistic Regression (current) | ~0.1–2ms | ~5–10% lower |
| XGBoost | ~10–20ms | Higher |
| Neural Network | ~50–200ms | Highest |

Logistic Regression was chosen to meet a <5ms P99 SLA for real-time customer-facing decisions. XGBoost or neural networks would be preferred for batch use cases where latency is not constrained.

### Synchronous vs Async
The current implementation is synchronous — simple to debug and sufficient for moderate traffic. For >1,000 req/sec, this should move to async FastAPI workers or a message queue (e.g. Kafka → worker pool).

### Thresholds in YAML vs Code
ROI thresholds, anomaly cutoffs, and incentive costs live in `configs/ecommerce.yaml`, not hardcoded. Business teams can retune without redeploying.

---

## Business Impact (Simulated Evaluation)

Run the business simulation:
```bash
python evaluate.py
```

DecisionForge was evaluated against a baseline that intervenes purely on churn probability with no cost or risk consideration:

- **~13% simulated net revenue uplift** by avoiding unprofitable incentives
- **~20% reduction** in unnecessary retention spend
- **10–20% blocking** of risky or abusive interactions

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements-ai.txt
```

### 2. Create config
```bash
mkdir -p configs
```

Add `configs/ecommerce.yaml`:
```yaml
revenue_per_user: 100
incentive_cost: 20
anomaly_threshold: 0.7
request_limit: 10
```

### 3. Run model evaluation
```bash
python evaluate_models.py
```

### 4. Run business simulation
```bash
python evaluate.py
```

### 5. Start REST API
```bash
uvicorn app.ai.api:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for interactive API documentation.

---

## File Structure

```
decision-forge/
├── app/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── ai/
│   │   ├── __init__.py
│   │   └── api.py                  # REST API (FastAPI)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── decision_engine.py      # Decision logic + latency tracking
│   │   ├── roi.py                  # Expected value calculator
│   │   ├── security.py             # Anomaly/abuse gate
│   │   └── audit.py                # Decision logger
│   └── ml/
│       ├── __init__.py
│       ├── preprocessor.py         # Feature engineering + scaling
│       ├── churn_model.py          # Logistic Regression + P/R/F1/AUC
│       └── anomaly_model.py        # Isolation Forest + evaluation
├── configs/
│   └── ecommerce.yaml              # Client-specific thresholds
├── evaluate.py                     # Business simulation (revenue uplift)
├── evaluate_models.py              # ML model evaluation (P/R/F1)
├── run_decision.py                 # Single decision example
├── requirements-ai.txt
└── README.md
```

---

## Technology Stack

- **Python 3.13+**
- **scikit-learn** — Logistic Regression, Isolation Forest, StandardScaler
- **FastAPI + Pydantic** — REST API with typed request/response schemas
- **pandas + numpy** — feature engineering and preprocessing
- **PyYAML** — client-specific configuration
- **uvicorn** — ASGI server

---

## Project Philosophy

1. **Predictions do not equal decisions** — ML scores must be converted to business actions
2. **Not acting is sometimes optimal** — ROI must justify every intervention
3. **Every decision must justify its cost** — expected value calculation is mandatory
4. **Models must be evaluated, not just trained** — Precision, Recall, F1 are tracked and exposed
5. **Decisions must be auditable** — complete logging of inputs, outputs, and rationale
6. **Configuration over code** — YAML-based client-specific tuning
7. **Latency is a design constraint** — model choices are explicitly traded off against speed

---

## Author

Built by Suhani Khanna as a demonstration of production-grade ML decision systems with REST API integration, structured feature engineering, and model evaluation.

---

**DecisionForge: Where ML predictions meet business accountability.** 🚀