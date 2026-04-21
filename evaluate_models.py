"""
evaluate_models.py
------------------
Offline model evaluation script.

Prints Precision, Recall, F1-score for the churn model and
evaluation summary for the anomaly model.

Run:
    python evaluate_models.py
"""

import pandas as pd
from app.ml.churn_model import ChurnModel
from app.ml.anomaly_model import AnomalyModel

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

churn_data = pd.DataFrame({
    "tenure":           [1, 5, 10, 2, 7, 3, 8,  1,  6,  4],
    "monthly_charges":  [200, 150, 100, 220, 130, 210, 120, 195, 145, 175],
    "churn":            [1,   0,   0,   1,   0,   1,   0,   1,   0,   0],
})

anomaly_data = pd.DataFrame({
    "request_count_today": [1, 2, 1, 10, 2, 1, 15, 1, 3, 1],
    "login_attempts":      [1, 1, 1,  7, 1, 1, 10, 1, 2, 1],
})

# Optional: provide pseudo ground-truth labels for anomaly eval
# -1 = anomaly, 1 = normal
anomaly_labels = pd.Series([1, 1, 1, -1, 1, 1, -1, 1, 1, 1])

# ---------------------------------------------------------------------------
# Train & evaluate churn model
# ---------------------------------------------------------------------------

print("=" * 60)
print("CHURN MODEL EVALUATION")
print("=" * 60)

churn_model = ChurnModel()
churn_metrics = churn_model.train(churn_data, target="churn")

print(f"\nPrecision : {churn_metrics['precision']}")
print(f"Recall    : {churn_metrics['recall']}")
print(f"F1-Score  : {churn_metrics['f1_score']}")
print(f"ROC-AUC   : {churn_metrics['roc_auc']}")
print("\nFull Classification Report:")
print(churn_metrics["classification_report"])

# ---------------------------------------------------------------------------
# Train & evaluate anomaly model
# ---------------------------------------------------------------------------

print("=" * 60)
print("ANOMALY MODEL EVALUATION")
print("=" * 60)

anomaly_model = AnomalyModel()
anomaly_metrics = anomaly_model.train(anomaly_data, labels=anomaly_labels)

for key, value in anomaly_metrics.items():
    print(f"{key:25s}: {value}")

# ---------------------------------------------------------------------------
# Inference sanity check
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("INFERENCE SANITY CHECK")
print("=" * 60)

test_users = [
    {"tenure": 1, "monthly_charges": 220, "label": "High-risk churner"},
    {"tenure": 8, "monthly_charges": 120, "label": "Loyal customer"},
]

for user in test_users:
    prob = churn_model.predict_proba({"tenure": user["tenure"], "monthly_charges": user["monthly_charges"]})
    print(f"\n{user['label']}")
    print(f"  Churn probability : {prob:.4f}")

anomaly_tests = [
    {"request_count_today": 1,  "login_attempts": 1, "label": "Normal user"},
    {"request_count_today": 15, "login_attempts": 8, "label": "Bot-like behavior"},
]

for user in anomaly_tests:
    score = anomaly_model.score({"request_count_today": user["request_count_today"], "login_attempts": user["login_attempts"]})
    print(f"\n{user['label']}")
    print(f"  Anomaly score : {score:.4f} {'⚠️  HIGH RISK' if score > 0.5 else '✓ Normal'}")

print("\n" + "=" * 60)
print("Evaluation complete.")
print("=" * 60)