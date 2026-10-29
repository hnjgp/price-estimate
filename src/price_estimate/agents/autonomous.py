"""LLM-driven variant of the planner: the model decides which tools to call.

Same tools as the PlanningAgent (scan, estimate, notify), but orchestrated by
tool calling. A hard limit on iterations prevents an endless loop if the model
keeps calling tools.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from price_estimate.agents.base import GREEN, Agent
from price_estimate.agents.deals import Deal, Opportunity
from price_estimate.config import settings

SYSTEM = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
USER = """First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
Then just reply OK to indicate success."""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    _tool(
        "scan_the_internet_for_bargains",
        "Returns top bargains scraped from the internet with their offered prices",
        {},
        [],
    ),
    _tool(
        "estimate_true_value",
        "Given the description of an item, estimate how much it is actually worth",
        {"description": {"type": "string", "description": "The description of the item"}},
        ["description"],
    ),
    _tool(
        "notify_user_of_deal",
        "Send the user a push notification about the single most compelling deal; only call this once",
        {
            "description": {"type": "string", "description": "Description of the item"},
            "deal_price": {"type": "number", "description": "The price offered by this deal"},
            "estimated_true_value": {"type": "number", "description": "The estimated actual value"},
            "url": {"type": "string", "description": "The URL of this deal"},
        },
        ["description", "deal_price", "estimated_true_value", "url"],
    ),
]


class AutonomousPlanningAgent(Agent):
    name = "Autonomous Planning Agent"
    color = GREEN
    MAX_TURNS = 12

    def __init__(self, scanner, pricer, messenger, client=None, model: str | None = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.scanner, self.pricer, self.messenger = scanner, pricer, messenger
        self.client = client
        self.model = model or settings.frontier_model
        self.memory: Sequence[Opportunity] = ()
        self.opportunity: Opportunity | None = None

    # ------------------------------------------------------------ tools
    def scan_the_internet_for_bargains(self) -> str:
        selection = self.scanner.scan(memory=self.memory)
        return selection.model_dump_json() if selection else "No deals found"

    def estimate_true_value(self, description: str) -> str:
        return f"The estimated true value of {description} is {self.pricer.price(description):.2f}"

    def notify_user_of_deal(
        self, description: str, deal_price: float, estimated_true_value: float, url: str
    ) -> str:
        if self.opportunity:
            return "A notification was already sent; not sending another"
        self.messenger.notify(description, deal_price, estimated_true_value, url)
        self.opportunity = Opportunity(
            deal=Deal(product_description=description, price=deal_price, url=url),
            estimate=estimated_true_value,
            discount=estimated_true_value - deal_price,
        )
        return "Notification sent ok"

    def handle_tool_calls(self, message) -> list[dict]:
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for call in message.tool_calls:
            tool = mapping.get(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
                content = tool(**args) if tool else f"Unknown tool {call.function.name}"
            except Exception as error:
                content = f"Tool error: {error}"
            self.log(f"tool {call.function.name} -> {content[:80]}")
            results.append({"role": "tool", "content": content, "tool_call_id": call.id})
        return results

    # ------------------------------------------------------------- loop
    def plan(self, memory: Sequence[Opportunity] = ()) -> Opportunity | None:
        self.log("starting a run")
        self.memory, self.opportunity = memory, None
        messages: list = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
        for _ in range(self.MAX_TURNS):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOLS
            )
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls":
                self.log(f"finished: {choice.message.content}")
                break
            messages.append(choice.message)
            messages.extend(self.handle_tool_calls(choice.message))
        else:
            self.log("stopped after reaching the turn limit")
        return self.opportunity
