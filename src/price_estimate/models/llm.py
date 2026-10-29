"""Frontier LLM pricers: zero-shot, and RAG (zero-shot plus 5 similar priced products).

Any model reachable through LiteLLM works for zero-shot (``openai/gpt-4.1-nano``,
``anthropic/claude-sonnet-4-5``, ``gemini/...``). The RAG pricer takes an
OpenAI-compatible client so it can be swapped or mocked in tests.
"""

from __future__ import annotations

from price_estimate.config import settings
from price_estimate.evaluation.evaluator import parse_price
from price_estimate.models.base import Pricer

INSTRUCTION = "Estimate the price of this product. Respond with the price, no explanation"


def make_context(similars: list[str], prices: list[float]) -> str:
    lines = [
        "For context, here are some other items that might be similar to the item you need to estimate.\n"
    ]
    for text, price in zip(similars, prices, strict=True):
        lines.append(f"Potentially related product:\n{text}\nPrice is ${price:.2f}\n")
    return "\n".join(lines)


def build_messages(
    text: str, similars: list[str] | None = None, prices: list[float] | None = None
) -> list[dict]:
    content = f"{INSTRUCTION}\n\n{text}\n\n"
    if similars:
        content += make_context(similars, prices or [])
    return [{"role": "user", "content": content}]


class ZeroShotPricer(Pricer):
    """Ask an LLM directly, with no examples."""

    def __init__(self, model: str, **completion_kwargs):
        self.model = model
        self.name = model.split("/")[-1]
        self.kwargs = completion_kwargs

    def price(self, text: str) -> float:
        from litellm import completion

        response = completion(model=self.model, messages=build_messages(text), **self.kwargs)
        return parse_price(response.choices[0].message.content)


class RAGPricer(Pricer):
    """Retrieve similar priced products, then ask a frontier model."""

    def __init__(self, similar_products, client=None, model: str | None = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.similar = similar_products
        self.client = client
        self.model = model or settings.frontier_model
        self.name = f"{self.model}_rag"

    def price(self, text: str) -> float:
        documents, prices = self.similar.find(text)
        kwargs = {"seed": 42}
        if self.model.startswith("gpt-5"):
            kwargs["reasoning_effort"] = "none" if self.model.startswith("gpt-5.1") else "minimal"
        response = self.client.chat.completions.create(
            model=self.model, messages=build_messages(text, documents, prices), **kwargs
        )
        return parse_price(response.choices[0].message.content)
