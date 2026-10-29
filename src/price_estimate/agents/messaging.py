from __future__ import annotations

import os

import requests

from price_estimate.agents.base import WHITE, Agent
from price_estimate.agents.deals import Opportunity
from price_estimate.config import settings

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    """Send push notifications (Pushover). Optionally lets Claude write the message."""

    name = "Messaging Agent"
    color = WHITE

    def __init__(self, sender=None):
        self.log("initializing")
        self.user = os.getenv("PUSHOVER_USER")
        self.token = os.getenv("PUSHOVER_TOKEN")
        self.sender = sender or self._pushover
        self.sent: list[str] = []

    def _pushover(self, text: str) -> bool:
        if not (self.user and self.token):
            self.log("Pushover is not configured; message logged only")
            return False
        try:
            response = requests.post(
                PUSHOVER_URL,
                data={
                    "user": self.user,
                    "token": self.token,
                    "message": text,
                    "sound": "cashregister",
                },
                timeout=15,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as error:
            self.log(f"push failed: {error}")
            return False

    def push(self, text: str) -> bool:
        self.sent.append(text)
        self.log(f"pushing: {text[:80]}")
        return self.sender(text)

    @staticmethod
    def format_alert(opp: Opportunity) -> str:
        return (
            f"Deal alert! ${opp.deal.price:.2f} (worth ~${opp.estimate:.2f}, "
            f"save ${opp.discount:.2f}): {opp.deal.product_description[:120]}... {opp.deal.url}"
        )

    def alert(self, opportunity: Opportunity) -> bool:
        return self.push(self.format_alert(opportunity))

    def craft_message(self, description: str, deal_price: float, estimate: float) -> str:
        from anthropic import Anthropic

        prompt = (
            "Summarize this deal in 2-3 exciting sentences for a push notification.\n"
            f"Item: {description}\nOffered price: {deal_price}\nEstimated true value: {estimate}\n"
            "Respond only with the message."
        )
        response = Anthropic().messages.create(
            model=settings.messaging_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def notify(self, description: str, deal_price: float, estimate: float, url: str) -> bool:
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                text = self.craft_message(description, deal_price, estimate)
            except Exception as error:
                self.log(f"Claude message failed, using plain text: {error}")
                text = None
        else:
            text = None
        text = text or f"Deal: {description[:150]} at ${deal_price:.2f} (worth ~${estimate:.2f})"
        return self.push(f"{text[:200]}... {url}")
