"""Classic baselines, from "guess randomly" up to gradient boosting on bag-of-words.

Baselines matter: an LLM that is off by $60 sounds bad until you see that a
constant guess is off by $106. Every model here follows the same pattern:
``fit(train_items)`` then ``price(text)``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

from price_estimate.data.items import Item
from price_estimate.models.base import Pricer


class RandomPricer(Pricer):
    name = "random"

    def __init__(self, low: int = 1, high: int = 1000, seed: int = 42):
        self.rng = random.Random(seed)
        self.low, self.high = low, high

    def price(self, text: str) -> float:
        return float(self.rng.randrange(self.low, self.high))


class ConstantPricer(Pricer):
    """Always predict the training-set mean."""

    name = "constant"

    def __init__(self):
        self.value = 0.0

    def fit(self, items: list[Item]) -> ConstantPricer:
        self.value = float(np.mean([i.price for i in items]))
        return self

    def price(self, text: str) -> float:
        return self.value


class FeatureLinearPricer(Pricer):
    """Linear regression on three hand-made features: weight, weight-unknown, text length."""

    name = "linear_regression"
    columns = ["weight", "weight_unknown", "text_length"]

    def __init__(self):
        self.model = LinearRegression()

    @staticmethod
    def features(weight: float | None, text: str) -> dict[str, float]:
        weight = weight or 0.0
        return {"weight": weight, "weight_unknown": float(weight == 0), "text_length": len(text)}

    def fit(self, items: list[Item]) -> FeatureLinearPricer:
        frame = pd.DataFrame([self.features(i.weight, i.text) for i in items])
        self.model.fit(frame[self.columns], [i.price for i in items])
        return self

    def _predict(self, weight: float | None, text: str) -> float:
        frame = pd.DataFrame([self.features(weight, text)])[self.columns]
        return max(float(self.model.predict(frame)[0]), 0.0)

    def price(self, text: str) -> float:
        return self._predict(None, text)

    def __call__(self, item: Item) -> float:
        return self._predict(item.weight, item.text)


class _BagOfWordsPricer(Pricer):
    """Shared plumbing: CountVectorizer over the description + any sklearn regressor."""

    def __init__(self, max_features: int = 2000):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = None

    def make_model(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, items: list[Item], limit: int | None = None):
        items = items[:limit] if limit else items
        x = self.vectorizer.fit_transform([i.text for i in items])
        self.model = self.make_model()
        self.model.fit(x, np.array([i.price for i in items]))
        return self

    def price(self, text: str) -> float:
        x = self.vectorizer.transform([text])
        return max(float(self.model.predict(x)[0]), 0.0)

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path):
        import joblib

        data = joblib.load(path)
        pricer = cls()
        pricer.vectorizer, pricer.model = data["vectorizer"], data["model"]
        return pricer


class BagOfWordsLinearPricer(_BagOfWordsPricer):
    name = "nlp_linear_regression"

    def make_model(self):
        return LinearRegression()


class RandomForestPricer(_BagOfWordsPricer):
    name = "random_forest"

    def __init__(self, n_estimators: int = 100, n_jobs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators, self.n_jobs = n_estimators, n_jobs

    def make_model(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=42, n_jobs=self.n_jobs
        )


class XGBoostPricer(_BagOfWordsPricer):
    name = "xgboost"

    def __init__(
        self, n_estimators: int = 1000, learning_rate: float = 0.1, n_jobs: int = 4, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_estimators, self.learning_rate, self.n_jobs = n_estimators, learning_rate, n_jobs

    def make_model(self):
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=self.n_jobs,
        )
