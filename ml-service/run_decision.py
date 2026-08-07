from app.config_loader import load_config
from app.core.decision_engine import DecisionEngine
from app.ml.churn_model import ChurnModel
from app.ml.anomaly_model import AnomalyModel

import pandas as pd

# 1️ Load client config
config = load_config("configs/ecommerce.yaml")

# 2️ Initialize models
churn_model = ChurnModel()
anomaly_model = AnomalyModel()

# --- Dummy training data (simulated) ---
churn_data = pd.DataFrame({
    "tenure": [1, 5, 10, 2, 7],
    "monthly_charges": [200, 150, 100, 220, 130],
    "churn": [1, 0, 0, 1, 0]
})

anomaly_data = pd.DataFrame({
    "request_count_today": [1, 2, 1, 10, 2],
    "login_attempts": [1, 1, 1, 7, 1]
})

# 3️ Train models
churn_model.train(churn_data, target="churn")
anomaly_model.train(anomaly_data)

# 4️ Incoming user features
user_features = {
    "tenure": 2,
    "monthly_charges": 210
}

security_features = {
    "request_count_today": 1,
    "login_attempts": 1
}

# 5️ Get ML scores
churn_prob = churn_model.predict_proba(user_features)
expected_lift = churn_prob * 0.3  # conservative uplift assumption
anomaly_score = anomaly_model.score(security_features)

# 6️ Build decision input
decision_input = {
    "churn_probability": churn_prob,
    "expected_lift": expected_lift,
    "anomaly_score": anomaly_score,
    "request_count_today": security_features["request_count_today"]
}

# 7️ Run decision engine
engine = DecisionEngine(config)
decision = engine.decide(decision_input)

print("\nFINAL DECISION:")
print(decision)