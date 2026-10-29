"""Turn one raw Amazon-Reviews-2023 metadata row into a clean ``Item``.

Rows are rejected (``None``) when the price is missing or out of range, or when
there is too little text to learn from.
"""

from __future__ import annotations

import json
import re

from price_estimate.data.items import Item

MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000

# Fields that leak no useful signal, or leak identifiers the model could memorise
REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

# Long upper-case alphanumeric tokens (e.g. "B07XJ8C8F5") are part numbers
PART_NUMBER = re.compile(r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")

UNIT_TO_POUNDS = {
    "pounds": 1.0,
    "pound": 1.0,
    "lbs": 1.0,
    "ounces": 1 / 16,
    "ounce": 1 / 16,
    "oz": 1 / 16,
    "grams": 1 / 453.592,
    "gram": 1 / 453.592,
    "g": 1 / 453.592,
    "milligrams": 1 / 453_592,
    "kilograms": 1 / 0.453592,
    "kilogram": 1 / 0.453592,
    "kg": 1 / 0.453592,
}


def simplify(value) -> str:
    """Flatten a list/str field into one line, capped at MAX_TEXT_EACH characters."""
    text = str(value).replace("\n", " ").replace("\r", "").replace("\t", "")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()[:MAX_TEXT_EACH]


def scrub(title: str, description, features, details: dict) -> str:
    """Build the full text for an item with noisy fields and part numbers removed."""
    details = {k: v for k, v in details.items() if k not in REMOVALS}
    parts = [title]
    if description:
        parts.append(simplify(description))
    if features:
        parts.append(simplify(features))
    if details:
        parts.append(json.dumps(details))
    text = "\n".join(parts)
    return PART_NUMBER.sub("", text).strip()[:MAX_TEXT_TOTAL]


def get_weight(details: dict) -> float:
    """Parse "Item Weight" (e.g. "2.5 Pounds", "12 ounces") into pounds; 0 if unknown.

    Never raises: malformed values simply count as unknown.
    """
    raw = details.get("Item Weight")
    if not raw or not isinstance(raw, str):
        return 0.0
    parts = raw.lower().split()
    try:
        amount = float(parts[0].replace(",", ""))
    except (ValueError, IndexError):
        return 0.0
    if len(parts) >= 3 and parts[1] == "hundredths" and parts[2] == "pounds":
        return amount / 100
    if len(parts) < 2:
        return 0.0
    factor = UNIT_TO_POUNDS.get(parts[1])
    return amount * factor if factor else 0.0


def parse(datapoint: dict, category: str) -> Item | None:
    try:
        price = float(datapoint["price"])
    except (TypeError, ValueError, KeyError):
        return None
    if not MIN_PRICE <= price <= MAX_PRICE:
        return None
    try:
        details = json.loads(datapoint.get("details") or "{}")
    except (TypeError, json.JSONDecodeError):
        details = {}
    title = datapoint.get("title") or ""
    full = scrub(title, datapoint.get("description"), datapoint.get("features"), details)
    if len(full) < MIN_CHARS:
        return None
    return Item(title=title, category=category, price=price, full=full, weight=get_weight(details))
