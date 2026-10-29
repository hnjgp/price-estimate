"""Error metrics for price prediction.

Mean absolute error (in dollars) is the headline number because it is easy to
explain: "on average we are off by $X". RMSLE is reported too because it treats a
$10 miss on a $20 item as worse than a $10 miss on a $900 item.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

GOOD, OK, BAD = "green", "orange", "red"


def color_for(error: float, truth: float) -> str:
    """Bucket a single prediction: green = close, orange = so-so, red = poor."""
    ratio = error / truth if truth > 0 else math.inf
    if error < 40 or ratio < 0.2:
        return GOOD
    if error < 80 or ratio < 0.4:
        return OK
    return BAD


def mae(truths: Sequence[float], guesses: Sequence[float]) -> float:
    return sum(abs(g - t) for t, g in zip(truths, guesses, strict=True)) / len(truths)


def rmse(truths: Sequence[float], guesses: Sequence[float]) -> float:
    return math.sqrt(sum((g - t) ** 2 for t, g in zip(truths, guesses, strict=True)) / len(truths))


def rmsle(truths: Sequence[float], guesses: Sequence[float]) -> float:
    pairs = zip(truths, guesses, strict=True)
    total = sum((math.log1p(max(g, 0)) - math.log1p(max(t, 0))) ** 2 for t, g in pairs)
    return math.sqrt(total / len(truths))


def r2(truths: Sequence[float], guesses: Sequence[float]) -> float:
    mean = sum(truths) / len(truths)
    ss_tot = sum((t - mean) ** 2 for t in truths)
    ss_res = sum((t - g) ** 2 for t, g in zip(truths, guesses, strict=True))
    return 1 - ss_res / ss_tot if ss_tot else 0.0


def hit_rate(truths: Sequence[float], guesses: Sequence[float]) -> float:
    """Share of predictions in the green bucket."""
    hits = sum(color_for(abs(g - t), t) == GOOD for t, g in zip(truths, guesses, strict=True))
    return hits / len(truths)


def confidence_interval(errors: Sequence[float], z: float = 1.96) -> float:
    """Half-width of the 95% confidence interval for the mean absolute error."""
    n = len(errors)
    if n < 2:
        return 0.0
    mean = sum(errors) / n
    var = sum((e - mean) ** 2 for e in errors) / (n - 1)
    return z * math.sqrt(var / n)


def summarize(truths: Sequence[float], guesses: Sequence[float]) -> dict[str, float]:
    errors = [abs(g - t) for t, g in zip(truths, guesses, strict=True)]
    return {
        "mae": mae(truths, guesses),
        "mae_ci95": confidence_interval(errors),
        "rmse": rmse(truths, guesses),
        "rmsle": rmsle(truths, guesses),
        "r2": r2(truths, guesses),
        "hit_rate": hit_rate(truths, guesses),
        "n": float(len(truths)),
    }
