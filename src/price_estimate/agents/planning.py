from __future__ import annotations

from collections.abc import Sequence

from price_estimate.agents.base import GREEN, Agent
from price_estimate.agents.deals import Deal, Opportunity
from price_estimate.config import settings


class PlanningAgent(Agent):
    """Deterministic workflow: scan -> estimate every deal -> alert on the best one."""

    name = "Planning Agent"
    color = GREEN

    def __init__(self, scanner, pricer, messenger, threshold: float | None = None):
        self.scanner = scanner
        self.pricer = pricer
        self.messenger = messenger
        self.threshold = settings.deal_threshold if threshold is None else threshold
        self.log("ready")

    def evaluate(self, deal: Deal) -> Opportunity:
        estimate = self.pricer.price(deal.product_description)
        return Opportunity(deal=deal, estimate=estimate, discount=estimate - deal.price)

    def plan(self, memory: Sequence[Opportunity] = ()) -> Opportunity | None:
        self.log("starting a run")
        selection = self.scanner.scan(memory=memory)
        if not selection or not selection.deals:
            self.log("no new deals")
            return None
        opportunities = []
        for deal in selection.deals[:5]:
            try:
                opportunities.append(self.evaluate(deal))
            except Exception as error:
                self.log(f"could not price {deal.url}: {error}")
        if not opportunities:
            return None
        best = max(opportunities, key=lambda o: o.discount)
        self.log(f"best discount ${best.discount:.2f} (threshold ${self.threshold:.0f})")
        if best.discount <= self.threshold:
            return None
        self.messenger.alert(best)
        return best
