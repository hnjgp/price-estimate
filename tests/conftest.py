import random

import pytest

from price_estimate.data.items import Item

WORDS = {
    "Electronics": ["laptop", "monitor", "4k", "ssd", "gaming", "display", "wireless", "speaker"],
    "Toys_and_Games": ["puzzle", "lego", "doll", "board", "game", "plush", "kids", "toy"],
    "Appliances": ["fridge", "washer", "dryer", "filter", "oven", "steel", "kitchen", "water"],
}
BASE_PRICE = {"Electronics": 400, "Toys_and_Games": 25, "Appliances": 150}


def make_items(n: int, seed: int = 0) -> list[Item]:
    """Synthetic items whose price really depends on their words, so models can learn."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        category = rng.choice(list(WORDS))
        words = rng.sample(WORDS[category], 4)
        price = BASE_PRICE[category] * rng.uniform(0.7, 1.3)
        summary = (
            f"Title: {' '.join(words)}\nCategory: {category}\nDescription: a {words[0]} product"
        )
        items.append(
            Item(
                title=f"{category} item {i}",
                category=category,
                price=round(price, 2),
                summary=summary,
                weight=rng.choice([0.0, rng.uniform(0.1, 30)]),
            )
        )
    return items


@pytest.fixture
def items():
    return make_items(300)
