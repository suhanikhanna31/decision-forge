# DecisionForge – Revenue & Risk-Aware Decision Engine with AI Explainability

DecisionForge is a backend decision system that converts machine learning predictions into accountable business actions. Instead of acting directly on churn probability, it evaluates return on investment (ROI), cost, and security risk before deciding whether to intervene, do nothing, or flag behavior.

**NEW: Now enhanced with AI-powered explanations** using free local AI (Ollama) to provide natural language reasoning for every decision, personalized customer messages, and strategic insights.

## The Problem

Most retention systems act directly on churn probability, overspend on incentives, ignore abuse and anomalous behavior, and lack explainability and auditability. This results in wasted retention spend and poor business decisions.

## The Solution

DecisionForge introduces a decision layer between prediction and action. Every action must be economically justified (ROI-positive), pass security and risk checks, and be explainable and auditable.

**With AI Integration:** Every decision now comes with natural language explanations, personalized customer retention messages, and actionable recommendations—all generated locally for free.

## Decision Flow

```
User Features
    ↓
ML Scoring Layer
• Churn Prediction (Logistic Regression)
• Anomaly Detection (Isolation Forest)
    ↓
Decision Engine
• Expected Value Calculation
• ROI Threshold Check
• Security Gating
    ↓
AI Enhancement Layer (NEW)
• Natural Language Explanations
• Personalized Messages
• Strategic Insights
    ↓
Decision Output
• INTERVENE / DO_NOTHING / FLAG
• AI Explanation
• Recommendations
    ↓
Audit Log
```

## System Components

### Core Components
* **Churn Model**: Predicts churn probability using historical user behavior
* **Anomaly Model**: Detects suspicious or abusive behavior using isolation-based anomaly detection
* **ROI Calculator**: Computes expected monetary value of an intervention
* **Decision Engine**: Converts ML scores and business constraints into actions
* **Security Gate**: Blocks or flags risky interactions before spending resources
* **Audit Logger**: Records every decision with inputs, rationale, and timestamp
* **YAML Configuration**: Enables client-specific thresholds, costs, and risk tolerance without code changes

### AI Components (NEW)
* **AI Explainer**: Generates natural language explanations for decisions using Ollama
* **AI-Enhanced Decision Engine**: Wraps core engine with AI capabilities
* **Message Generator**: Creates personalized customer retention messages
* **Insights Generator**: Analyzes decision patterns and provides strategic recommendations
* **REST API**: FastAPI endpoints with interactive Swagger UI documentation
* **Natural Language Query Interface**: Ask questions in plain English about system decisions

## Business Impact

### Core System (Simulated Evaluation)
DecisionForge was evaluated against a baseline strategy that intervenes purely on churn probability without considering cost or risk. In offline simulation, DecisionForge achieved:

- **~13% simulated net revenue uplift** by avoiding unprofitable incentives
- **~20% reduction** in unnecessary retention spend
- **10–20% blocking** of risky or abusive interactions
- **Maintained or improved retention** through selective, ROI-driven intervention

### AI Enhancement
The AI integration adds:
- **100% explainability** for every decision with natural language reasoning
- **Automated customer communication** with personalized retention messages
- **Strategic insights** from decision pattern analysis
- **Zero additional cost** - runs entirely locally with Ollama

## Technology Stack

**Core System:**
- Python 3.13+
- scikit-learn (Logistic Regression, Isolation Forest)
- Modular backend architecture
- YAML-based configuration
- Lightweight evaluation scripts

**AI Enhancement:**
- Ollama (free local AI)
- FastAPI with Swagger UI
- Natural language processing
- RESTful API architecture

No external paid services or dashboards required.

## Quick Start

### 1. Install Core Dependencies
```bash
pip install -r requirements-ai.txt
```

### 2. Set Up Ollama (Free Local AI)
```bash
# Install Ollama
brew install ollama

# Start Ollama server
ollama serve

# In another terminal, download the model
ollama pull llama3.2
```

### 3. Run Examples

**Test Basic Setup:**
```bash
python test_ai_setup.py
```

**Make a Single Decision with AI:**
```bash
python test_simple_decision.py
```

**Run Core Evaluation:**
```bash
python evaluate.py
```

**Start REST API Server:**
```bash
uvicorn app.ai.api:app --reload --port 8000
```

Then visit: http://localhost:8000/docs for interactive API documentation

## Example Usage

### Python API
```python
from app.config_loader import load_config
from app.ai.ai_enhanced_engine import AIEnhancedDecisionEngine

# Initialize
config = load_config("configs/ecommerce.yaml")
engine = AIEnhancedDecisionEngine(config, enable_ai=True)

# Make decision with AI explanation
result = engine.decide_with_explanation(
    inputs={
        "anomaly_score": 0.82,
        "request_count_today": 15,
        "expected_lift": 0.15,
        "revenue": 1200,
        "incentive_cost": 150,
        "churn": 0.75
    },
    return_explanation=True,
    return_recommendations=True
)

print(f"Decision: {result['decision']['decision']}")
print(f"Explanation: {result['explanation']}")
print(f"Recommendations: {result['recommendations']}")
```

### REST API
```bash
curl -X POST "http://localhost:8000/api/v1/decide" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "anomaly_score": 0.85,
      "request_count_today": 15,
      "expected_lift": 0.15,
      "revenue": 1200,
      "incentive_cost": 150,
      "churn": 0.75
    },
    "return_explanation": true
  }'
```

## AI Features

### 1. Decision Explanations
Every decision comes with natural language reasoning:
```
"Recommend immediate intervention for this high-value customer. 
With 75% churn probability and anomaly score of 0.82, the customer 
shows clear signs of leaving. The expected value of $856 demonstrates 
strong ROI justification."
```

### 2. Personalized Messages
Automatically generate retention messages:
```
"We greatly value your 18 months with us as a Premium customer. 
To show our appreciation, we'd like to offer you an exclusive 20% 
discount for the next 3 months. Please reply to activate this 
special offer."
```

### 3. Strategic Insights
Analyze decision patterns:
```
KEY TRENDS: 67% of decisions were interventions in high-value segment
OPPORTUNITIES: Focus on customers with anomaly scores 0.7-0.85
RECOMMENDATIONS: Implement proactive outreach for early warning signs
```

### 4. REST API
- Interactive Swagger UI documentation
- Health monitoring endpoints
- Batch processing support
- Natural language queries

## How Evaluation Works

A baseline policy intervenes whenever churn probability exceeds a fixed threshold. DecisionForge intervenes only when expected value is positive and the action passes security checks. 

For each strategy, net revenue is computed as revenue generated minus incentive cost. Aggregate net revenue is compared across users to measure uplift.

The AI enhancement maintains the same decision logic (preserving the 13% uplift) while adding explainability and automation.

## Project Philosophy

1. **Predictions do not equal decisions** - ML scores must be converted to business actions
2. **Not acting is sometimes optimal** - ROI must justify every intervention
3. **Every decision must justify its cost** - Expected value calculation is mandatory
4. **Business logic must be explainable** - AI provides natural language reasoning
5. **Decisions must be auditable** - Complete logging of inputs, outputs, and rationale
6. **Configuration over code** - YAML-based client-specific tuning

## What This Is

- A **decision infrastructure system** mirroring enterprise-grade backend platforms
- A **complete MLOps pipeline** from prediction to action to audit
- An **AI-enhanced decision engine** with free local explainability
- A **production-ready REST API** with interactive documentation

## What This Is Not

- ❌ Not a dashboard
- ❌ Not a Kaggle notebook
- ❌ Not a pure ML demo
- ❌ Not a paid cloud service (runs 100% locally for free)

## File Structure

```
decision-forge/
├── app/
│   ├── ai/                          # AI Enhancement Layer (NEW)
│   │   ├── __init__.py
│   │   ├── ai_explainer.py         # Natural language explanations
│   │   ├── ai_enhanced_engine.py   # AI-wrapped decision engine
│   │   ├── nl_query_interface.py   # Natural language queries
│   │   └── api.py                  # REST API endpoints
│   ├── core/                        # Core Decision Engine
│   │   ├── decision_engine.py
│   │   ├── roi.py
│   │   ├── security.py
│   │   └── audit.py
│   ├── ml/                          # ML Models
│   │   ├── churn_model.py
│   │   └── anomaly_model.py
│   └── config_loader.py
├── configs/
│   └── ecommerce.yaml               # Configuration
├── test_ai_setup.py                 # AI setup validation (NEW)
├── test_simple_decision.py          # AI decision test (NEW)
├── evaluate.py                      # Core evaluation
├── run_decision.py                  # Single decision example
└── requirements-ai.txt              # All dependencies (NEW)
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional ML models (XGBoost, Neural Networks)
- More sophisticated ROI models
- Real-time streaming decision pipelines
- A/B testing framework
- Multi-language support for AI explanations
- Additional API endpoints

## License

MIT License

## Author

Built by Suhani Khanna as a demonstration of production-grade decision systems with AI explainability.

---

**DecisionForge: Where ML predictions meet business accountability, now with AI-powered explainability—all running locally for free.** 🚀