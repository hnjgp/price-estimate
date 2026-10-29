"""Combine several pricers into one estimate.

Two improvements over the course version, which used fixed weights
(0.8 frontier / 0.1 fine-tuned / 0.1 neural net) even though its docstring
mentioned a learned linear model:

1. ``fit_weights`` actually *learns* the weights from validation predictions,
   using non-negative least squares (weights >= 0, rescaled to sum to 1).
2. If a member fails at prediction time (an API is down, Modal is cold), the
   ensemble drops it and renormalises the remaining weights instead of crashing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from price_estimate.models.base import Pricer

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"frontier": 0.8, "specialist": 0.1, "neural_network": 0.1}


def fit_weights(
    predictions: Mapping[str, Sequence[float]], truths: Sequence[float]
) -> dict[str, float]:
    """Learn non-negative weights that sum to 1 from per-member validation predictions."""
    names = list(predictions)
    matrix = np.column_stack([np.asarray(predictions[n], dtype=float) for n in names])
    weights, _ = nnls(matrix, np.asarray(truths, dtype=float))
    total = weights.sum()
    if total <= 0:
        weights = np.full(len(names), 1 / len(names))
    else:
        weights = weights / total
    return {name: float(w) for name, w in zip(names, weights, strict=True)}


def save_weights(weights: Mapping[str, float], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(weights), indent=2), encoding="utf-8")


def load_weights(path: str | Path | None) -> dict[str, float]:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(DEFAULT_WEIGHTS)


class EnsemblePricer(Pricer):
    name = "ensemble"

    def __init__(
        self,
        members: Mapping[str, Pricer],
        weights: Mapping[str, float] | None = None,
        preprocess=None,
    ):
        self.members = dict(members)
        self.weights = {k: v for k, v in (weights or DEFAULT_WEIGHTS).items() if k in self.members}
        if not self.weights:
            self.weights = {name: 1 / len(self.members) for name in self.members}
        self.preprocess = preprocess
        self.last_breakdown: dict[str, float] = {}

    def price(self, text: str) -> float:
        if self.preprocess is not None:
            try:
                text = self.preprocess(text)
            except Exception as error:
                log.warning("Preprocessing failed, using raw text: %s", error)
        estimates: dict[str, float] = {}
        for name in self.weights:
            try:
                estimates[name] = float(self.members[name].price(text))
            except Exception as error:
                log.warning("Ensemble member %s failed: %s", name, error)
        if not estimates:
            raise RuntimeError("Every ensemble member failed")
        total = sum(self.weights[n] for n in estimates)
        if total <= 0:
            total, weights = len(estimates), {n: 1.0 for n in estimates}
        else:
            weights = self.weights
        self.last_breakdown = estimates
        return sum(estimates[n] * weights[n] for n in estimates) / total
