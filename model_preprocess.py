import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pt, down_clip, up_clip):
        self.pt = pt
        self.down_clip = down_clip
        self.up_clip = up_clip

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        required_cols = ["Oldpeak", "Cholesterol"]
        missing = [col for col in required_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns in CustomTransformer: {missing}")

        X["Oldpeak"] = pd.to_numeric(X["Oldpeak"], errors="coerce")
        X["Cholesterol"] = pd.to_numeric(X["Cholesterol"], errors="coerce")

        if X["Oldpeak"].isna().any():
            raise ValueError("Oldpeak contains missing or non-numeric values.")
        if X["Cholesterol"].isna().any():
            raise ValueError("Cholesterol contains missing or non-numeric values.")

        transformed_oldpeak = self.pt.transform(X[["Oldpeak"]])
        X["Oldpeak"] = np.clip(transformed_oldpeak, self.down_clip, self.up_clip)

        X["Cholesterol"] = np.sqrt(np.clip(X["Cholesterol"], 0, None))

        return X


class HeartModel:


    def __init__(self, pipeline, threshold, columns, cat_columns):
        self.pipeline = pipeline
        self.threshold = threshold
        self.columns = columns
        self.cat_columns = cat_columns

    def _prepare(self, X):
        X = X.copy()

        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        for col in self.cat_columns:
            X[col] = X[col].astype("category")

        return X[self.columns]

    def predict(self, X):
        X = self._prepare(X)
        p = self.pipeline.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)

    def predict_proba(self, X):
        X = self._prepare(X)
        return self.pipeline.predict_proba(X)
