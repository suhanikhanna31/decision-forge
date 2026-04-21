"""
Feature Engineering & Preprocessing Pipeline
---------------------------------------------
Handles structured feature transformation before model training/inference.

Design Trade-offs:
  - We use StandardScaler for normalization (better accuracy) vs MinMaxScaler
    (faster, but sensitive to outliers in production traffic spikes).
  - Feature engineering is done at preprocessing time, not inference time,
    to keep latency low during real-time decisions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """
    Transforms raw user features into model-ready inputs.

    Responsibilities:
      1. Derive engineered features (e.g., charge_per_tenure)
      2. Normalize numerical features with StandardScaler
      3. Handle missing values with median imputation
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_medians: dict = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Churn feature engineering
    # ------------------------------------------------------------------

    def engineer_churn_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to churn data.

        New features:
          - charge_per_tenure : monthly_charges / (tenure + 1)
              Captures how expensive the service is relative to loyalty.
          - high_charge_flag  : 1 if monthly_charges > 180, else 0
              Binary flag for high-spend customers needing priority handling.
        """
        df = df.copy()
        df["charge_per_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)
        df["high_charge_flag"] = (df["monthly_charges"] > 180).astype(int)
        return df

    # ------------------------------------------------------------------
    # Anomaly feature engineering
    # ------------------------------------------------------------------

    def engineer_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to anomaly/security data.

        New features:
          - request_login_ratio : request_count_today / (login_attempts + 1)
              High ratio indicates automated scraping or bot-like behavior.
          - high_request_flag   : 1 if request_count_today > 5, else 0
              Quick binary signal for rate-limiting decisions.
        """
        df = df.copy()
        df["request_login_ratio"] = df["request_count_today"] / (
            df["login_attempts"] + 1
        )
        df["high_request_flag"] = (df["request_count_today"] > 5).astype(int)
        return df

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        """
        Compute scaling parameters and median values from training data.
        Must be called before transform().
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Store medians for missing-value imputation at inference time
        self.feature_medians = df[numeric_cols].median().to_dict()
        self.scaler.fit(df[numeric_cols])
        self.fitted_columns = numeric_cols
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply median imputation + standard scaling to a DataFrame.

        Latency note: This runs in O(n*f) time where n = rows, f = features.
        For single-row real-time inference this is sub-millisecond.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy()
        # Impute missing values with training medians
        for col, median_val in self.feature_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)

        # Scale only the columns seen during fit
        cols_present = [c for c in self.fitted_columns if c in df.columns]
        df[cols_present] = self.scaler.transform(df[cols_present])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Single-row inference helper
    # ------------------------------------------------------------------

    def transform_single(self, feature_dict: dict, feature_keys: list) -> np.ndarray:
        """
        Transform a single inference row (dict) into a numpy array.

        Args:
            feature_dict : raw feature values, e.g. {"tenure": 2, ...}
            feature_keys : ordered list of feature names the model expects

        Returns:
            Scaled 1-D numpy array ready for model.predict()
        """
        row = {k: feature_dict.get(k, self.feature_medians.get(k, 0)) for k in feature_keys}
        df = pd.DataFrame([row])
        scaled = self.transform(df)
        return scaled[feature_keys].values[0]