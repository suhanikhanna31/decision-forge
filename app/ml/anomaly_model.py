import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyModel:
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42
        )

    def train(self, df: pd.DataFrame):
        self.model.fit(df)

    def score(self, features: dict) -> float:
        df = pd.DataFrame([features])
        score = -self.model.decision_function(df)[0]
        return min(max(score, 0.0), 1.0)