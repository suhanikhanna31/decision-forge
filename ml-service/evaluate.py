import pandas as pd
from app.config_loader import load_config
from app.core.decision_engine import DecisionEngine
from app.ml.churn_model import ChurnModel
from app.ml.anomaly_model import AnomalyModel

# Load config
config = load_config("configs/ecommerce.yaml")

# Init models
churn_model = ChurnModel()
anomaly_model = AnomalyModel()

# Dummy training data (same as before)
churn_data = pd.DataFrame({
    "tenure": [1, 5, 10, 2, 7],
    "monthly_charges": [200, 150, 100, 220, 130],
    "churn": [1, 0, 0, 1, 0]
})

anomaly_data = pd.DataFrame({
    "request_count_today": [1, 2, 1, 10, 2],
    "login_attempts": [1, 1, 1, 7, 1]
})

churn_model.train(churn_data, target="churn")
anomaly_model.train(anomaly_data)

engine = DecisionEngine(config)

# Simulated users
users = [
    {"tenure": 1, "monthly_charges": 220, "request_count_today": 1},
    {"tenure": 8, "monthly_charges": 120, "request_count_today": 1},
    {"tenure": 2, "monthly_charges": 210, "request_count_today": 6},
    {"tenure": 6, "monthly_charges": 140, "request_count_today": 1},
    {"tenure": 3, "monthly_charges": 200, "request_count_today": 1},
]

baseline_revenue = 0
forge_revenue = 0
baseline_cost = 0
forge_cost = 0

for user in users:
    churn_prob = churn_model.predict_proba({
        "tenure": user["tenure"],
        "monthly_charges": user["monthly_charges"]
    })

    expected_lift = churn_prob * 0.3
    anomaly_score = anomaly_model.score({
        "request_count_today": user["request_count_today"],
        "login_attempts": 1
    })

    # ---- BASELINE ----
    if churn_prob > 0.5:
        baseline_cost += config["incentive_cost"]
        baseline_revenue += config["revenue_per_user"] * 0.7
    else:
        baseline_revenue += config["revenue_per_user"]

    # ---- DECISIONFORGE ----
    decision = engine.decide({
        "churn_probability": churn_prob,
        "expected_lift": expected_lift,
        "anomaly_score": anomaly_score,
        "request_count_today": user["request_count_today"]
    })

    if decision["decision"] == "INTERVENE":
        forge_cost += config["incentive_cost"]
        forge_revenue += config["revenue_per_user"] * 0.7
    else:
        forge_revenue += config["revenue_per_user"]

# Results
baseline_net = baseline_revenue - baseline_cost
forge_net = forge_revenue - forge_cost

uplift_pct = ((forge_net - baseline_net) / baseline_net) * 100

print("\nBASELINE NET REVENUE:", baseline_net)
print("DECISIONFORGE NET REVENUE:", forge_net)
print("UPLIFT (%):", round(uplift_pct, 2))