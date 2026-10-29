"""Deduplicate, rebalance and split the item pool.

Raw Amazon data is dominated by cheap items and by the Automotive category. A
model trained on that would learn to always guess low. We therefore sample with
a weight that grows with price and down-weights over-represented categories.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from price_estimate.data.items import Item

CATEGORY_DAMPING = {"Tools_and_Home_Improvement": 0.5, "Automotive": 0.05}


def deduplicate(items: Sequence[Item], seed: int = 42) -> list[Item]:
    """Shuffle, then drop items whose title or full text has been seen before."""
    pool = list(items)
    random.Random(seed).shuffle(pool)
    seen_titles: set[str] = set()
    seen_text: set[str] = set()
    result = []
    for item in pool:
        if item.title in seen_titles or (item.full and item.full in seen_text):
            continue
        seen_titles.add(item.title)
        if item.full:
            seen_text.add(item.full)
        result.append(item)
    return result


def sampling_weights(items: Sequence[Item], power: float = 2.0) -> np.ndarray:
    prices = np.array([i.price for i in items], dtype=float)
    categories = np.array([i.category for i in items])
    span = prices.max() - prices.min()
    scaled = (prices - prices.min()) / (span + 1e-9)
    weights = scaled**power + 1e-12  # tiny floor so the cheapest item is still possible
    for category, factor in CATEGORY_DAMPING.items():
        weights[categories == category] *= factor
    return weights / weights.sum()


def balanced_sample(items: Sequence[Item], size: int, seed: int = 42) -> list[Item]:
    size = min(size, len(items))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=size, replace=False, p=sampling_weights(items))
    sample = [items[i] for i in idx]
    random.Random(seed).shuffle(sample)
    return sample


def split(
    items: Sequence[Item], val_size: int, test_size: int
) -> tuple[list[Item], list[Item], list[Item]]:
    """Return (train, validation, test); test is taken from the end."""
    if val_size + test_size >= len(items):
        raise ValueError("Not enough items for the requested validation and test sizes")
    train_end = len(items) - val_size - test_size
    return (
        list(items[:train_end]),
        list(items[train_end : train_end + val_size]),
        list(items[train_end + val_size :]),
    )
