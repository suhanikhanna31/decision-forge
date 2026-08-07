"""
Churn Prediction Model
----------------------
Logistic Regression with structured feature engineering and evaluation metrics.

Design Trade-offs:
  - Logistic Regression chosen over tree-based models:
      ✓ Low inference latency (~0.1ms per prediction)
      ✓ Probability outputs are well-calibrated
      ✗ Lower accuracy on non-linear patterns vs XGBoost
  - StandardScaler is applied before training for numerical stability.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.ml.preprocessor import FeaturePreprocessor


class ChurnModel:
    """
    Predicts the probability that a user will churn.

    Pipeline:
        raw features → feature engineering → scaling → logistic regression
    """

    # Features used for training and inference
    FEATURE_COLS = ["tenure", "monthly_charges", "charge_per_tenure", "high_charge_flag"]

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.preprocessor = FeaturePreprocessor()
        self.is_trained = False
        self.evaluation_report: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, target: str = "churn") -> dict:
        """
        Train the churn model and evaluate with Precision, Recall, F1, AUC.

        Args:
            df     : DataFrame with raw features + target column
            target : name of the binary target column (0 = retained, 1 = churned)

        Returns:
            dict with evaluation metrics
        """
        # 1. Feature engineering
        df = self.preprocessor.engineer_churn_features(df)

        X = df[self.FEATURE_COLS]
        y = df[target]

        # 2. Fit preprocessor on training data
        X_scaled_df = self.preprocessor.fit_transform(X)
        X_scaled = X_scaled_df[self.FEATURE_COLS].values

        # 3. Train/val split for offline evaluation (if enough data)
        if len(df) >= 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # Small dataset: train = val (demo/simulation mode)
            X_train, X_val, y_train, y_val = X_scaled, X_scaled, y, y

        # 4. Fit model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # 5. Evaluate
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]

        self.evaluation_report = {
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_val, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_val, y_proba), 4) if len(set(y_val)) > 1 else "N/A",
            "classification_report": classification_report(
                y_val, y_pred, zero_division=0
            ),
        }

        return self.evaluation_report

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: dict) -> float:
        """
        Return churn probability for a single user.

        Args:
            features : dict with keys 'tenure', 'monthly_charges'

        Returns:
            float in [0, 1] — probability of churn
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Derive engineered features for inference
        row = pd.DataFrame([features])
        row = self.preprocessor.engineer_churn_features(row)

        # Fill any missing engineered columns with 0
        for col in self.FEATURE_COLS:
            if col not in row.columns:
                row[col] = 0

        row_scaled = self.preprocessor.transform(row)
        X = row_scaled[self.FEATURE_COLS].values

        return float(self.model.predict_proba(X)[0][1])

    # ------------------------------------------------------------------
    # Metrics access
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return last evaluation metrics dict."""
        return self.evaluation_report