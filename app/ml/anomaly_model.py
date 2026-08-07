"""
Anomaly Detection Model
-----------------------
Isolation Forest with structured feature engineering and evaluation metrics.

Design Trade-offs:
  - Isolation Forest chosen for unsupervised anomaly detection:
      ✓ No labeled anomaly data required
      ✓ Efficient O(n log n) training, O(log n) inference
      ✗ Scores are relative, not calibrated probabilities
  - contamination=0.2 means we expect ~20% of traffic to be anomalous.
    This is tunable per client via YAML config.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from app.ml.preprocessor import FeaturePreprocessor


class AnomalyModel:
    """
    Detects anomalous/abusive user behavior.

    Pipeline:
        raw features → feature engineering → scaling → isolation forest
    """

    FEATURE_COLS = [
        "request_count_today",
        "login_attempts",
        "request_login_ratio",
        "high_request_flag",
    ]

    def __init__(self, contamination: float = 0.2):
        """
        Args:
            contamination : expected fraction of anomalies in training data.
                            Controls the decision threshold of Isolation Forest.
        """
        self.model = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        self.preprocessor = FeaturePreprocessor()
        self.is_trained = False
        self.evaluation_report: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, labels: pd.Series = None) -> dict:
        """
        Train the anomaly model.

        Args:
            df     : DataFrame with raw behavior features
            labels : optional Series of ground-truth labels (1=normal, -1=anomaly)
                     If provided, supervised metrics (P/R/F1) are computed.

        Returns:
            dict with evaluation metrics (or empty dict if no labels)
        """
        # 1. Feature engineering
        df = self.preprocessor.engineer_anomaly_features(df)

        # Ensure all feature columns exist
        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0

        X = df[self.FEATURE_COLS]

        # 2. Fit preprocessor
        X_scaled_df = self.preprocessor.fit_transform(X)
        X_scaled = X_scaled_df[self.FEATURE_COLS].values

        # 3. Train Isolation Forest
        self.model.fit(X_scaled)
        self.is_trained = True

        # 4. Evaluate if labels provided
        if labels is not None:
            preds = self.model.predict(X_scaled)  # 1 = normal, -1 = anomaly
            # Convert to binary: anomaly=1, normal=0
            y_pred_bin = (preds == -1).astype(int)
            y_true_bin = (labels == -1).astype(int)

            self.evaluation_report = {
                "precision": round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 4),
                "recall": round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 4),
                "f1_score": round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 4),
                "note": "Supervised eval using provided labels.",
            }
        else:
            # Unsupervised: report contamination assumption and score distribution
            scores = self.model.decision_function(X_scaled)
            self.evaluation_report = {
                "note": "Unsupervised model — no ground-truth labels provided.",
                "contamination": self.model.contamination,
                "score_mean": round(float(np.mean(scores)), 4),
                "score_std": round(float(np.std(scores)), 4),
                "flagged_fraction": round(
                    float(np.mean(self.model.predict(X_scaled) == -1)), 4
                ),
            }

        return self.evaluation_report

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, features: dict) -> float:
        """
        Return anomaly score for a single request.

        Score interpretation:
          > 0.5  → likely anomalous (high risk)
          0–0.5  → borderline
          < 0    → normal behavior

        Args:
            features : dict with keys 'request_count_today', 'login_attempts'

        Returns:
            float anomaly score (higher = more anomalous)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        row = pd.DataFrame([features])
        row = self.preprocessor.engineer_anomaly_features(row)

        for col in self.FEATURE_COLS:
            if col not in row.columns:
                row[col] = 0

        row_scaled = self.preprocessor.transform(row)
        X = row_scaled[self.FEATURE_COLS].values

        # Isolation Forest decision_function: lower = more anomalous
        raw_score = self.model.decision_function(X)[0]
        # Invert and normalize to [0, 1] range for intuitive interpretation
        anomaly_score = float(1 / (1 + np.exp(raw_score * 5)))
        return round(anomaly_score, 4)

    # ------------------------------------------------------------------
    # Metrics access
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return last evaluation metrics dict."""
        return self.evaluation_report