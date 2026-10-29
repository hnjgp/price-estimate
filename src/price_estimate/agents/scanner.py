from __future__ import annotations

from collections.abc import Sequence

from price_estimate.agents.base import CYAN, Agent
from price_estimate.agents.deals import DealSelection, Opportunity, ScrapedDeal
from price_estimate.config import settings

SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation. Provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal.
A thorough description of the product matters more than the terms of the deal.
Be careful with products described as "$XXX off" or "reduced by $XXX" - that isn't the actual price. Only respond with products when you are highly confident about the price."""

USER_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
Rephrase each description as a summary of the product itself, not the terms of the deal.

Deals:

"""
USER_SUFFIX = "\n\nInclude exactly 5 deals, no more."


class ScannerAgent(Agent):
    name = "Scanner Agent"
    color = CYAN

    def __init__(self, client=None, model: str | None = None, fetcher=ScrapedDeal.fetch):
        self.log("initializing")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model or settings.scanner_model
        self.fetcher = fetcher

    def fetch_new_deals(self, memory: Sequence[Opportunity]) -> list[ScrapedDeal]:
        seen = {opp.deal.url for opp in memory}
        scraped = [deal for deal in self.fetcher() if deal.url not in seen]
        self.log(f"found {len(scraped)} deals not seen before")
        return scraped

    @staticmethod
    def make_user_prompt(scraped: Sequence[ScrapedDeal]) -> str:
        return USER_PREFIX + "\n\n".join(d.describe() for d in scraped) + USER_SUFFIX

    def scan(self, memory: Sequence[Opportunity] = ()) -> DealSelection | None:
        scraped = self.fetch_new_deals(memory)
        if not scraped:
            return None
        self.log(f"asking {self.model} to pick the best deals (structured output)")
        extra = {"reasoning_effort": "minimal"} if self.model.startswith("gpt-5-") else {}
        response = self.client.chat.completions.parse(
            **extra,
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.make_user_prompt(scraped)},
            ],
            response_format=DealSelection,
        )
        selection = response.choices[0].message.parsed
        if selection is None:
            return None
        selection.deals = [d for d in selection.deals if d.price > 0]
        self.log(f"selected {len(selection.deals)} deals with a valid price")
        return selection
