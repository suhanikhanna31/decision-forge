import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ChurnModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, df: pd.DataFrame, target: str):
        X = df.drop(columns=[target])
        y = df[target]
        self.model.fit(X, y)

    def predict_proba(self, features: dict) -> float:
        df = pd.DataFrame([features])
        return self.model.predict_proba(df)[0][1]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)