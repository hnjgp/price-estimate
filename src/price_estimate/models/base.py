"""Common interface for every price model.

``price(text)`` is what the agents use: they only have a free-text description.
``__call__(item)`` is what the evaluator uses; by default it just prices the item's
text, but feature-based models can override it to use structured fields (weight).
"""

from __future__ import annotations

from price_estimate.data.items import Item


class Pricer:
    name: str = "pricer"

    def price(self, text: str) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, item: Item) -> float:
        return self.price(item.text)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
