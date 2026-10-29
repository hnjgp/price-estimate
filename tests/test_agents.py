import json
from types import SimpleNamespace

import pytest

from price_estimate.agents.autonomous import AutonomousPlanningAgent
from price_estimate.agents.deals import Deal, DealSelection, Opportunity, ScrapedDeal, clean_html
from price_estimate.agents.framework import DealAgentFramework, Memory
from price_estimate.agents.messaging import MessagingAgent
from price_estimate.agents.planning import PlanningAgent
from price_estimate.agents.scanner import ScannerAgent

bs4 = pytest.importorskip("bs4")

DEALS = [
    Deal(product_description="55 inch 4K TV", price=200, url="https://x/tv"),
    Deal(product_description="Gaming laptop RTX", price=650, url="https://x/laptop"),
    Deal(product_description="USB cable", price=5, url="https://x/cable"),
]
ESTIMATES = {"55 inch 4K TV": 450, "Gaming laptop RTX": 700, "USB cable": 8}


class FakeScanner:
    def __init__(self, deals=DEALS):
        self.deals, self.memory_seen = deals, None

    def scan(self, memory=()):
        self.memory_seen = memory
        return DealSelection(deals=self.deals) if self.deals else None


class FakePricer:
    def price(self, text):
        return ESTIMATES[text]


def messenger():
    return MessagingAgent(sender=lambda text: True)


class TestPlanning:
    def test_picks_biggest_discount_and_alerts(self):
        m = messenger()
        best = PlanningAgent(FakeScanner(), FakePricer(), m, threshold=50).plan()
        assert best.deal.url == "https://x/tv"
        assert best.discount == 250
        assert len(m.sent) == 1 and "x/tv" in m.sent[0]

    def test_below_threshold_is_ignored(self):
        m = messenger()
        assert PlanningAgent(FakeScanner(), FakePricer(), m, threshold=1000).plan() is None
        assert m.sent == []

    def test_no_deals(self):
        assert PlanningAgent(FakeScanner([]), FakePricer(), messenger()).plan() is None


class TestScanner:
    def test_filters_seen_and_bad_prices(self):
        scraped = [
            ScrapedDeal(title="TV", summary="s", url="https://x/tv"),
            ScrapedDeal(title="New", summary="s", url="https://x/new"),
        ]
        parsed = DealSelection(
            deals=[
                Deal(product_description="new", price=10, url="https://x/new"),
                Deal(product_description="free?", price=0, url="https://x/zero"),
            ]
        )
        calls = []

        def parse(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
        memory = [Opportunity(deal=DEALS[0], estimate=1, discount=1)]
        agent = ScannerAgent(client=client, model="gpt-5-mini", fetcher=lambda: scraped)
        selection = agent.scan(memory)
        assert [d.url for d in selection.deals] == ["https://x/new"]
        prompt = calls[0]["messages"][1]["content"]
        assert "https://x/new" in prompt and "https://x/tv" not in prompt


def test_clean_html():
    html = '<div class="snippet summary">Great <b>TV</b>\n deal</div>'
    assert clean_html(html) == "Great TV deal"


class ScriptedClient:
    """Fake OpenAI client that replays a fixed sequence of responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        return self.responses.pop(0)


def tool_response(*calls):
    tool_calls = [
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(calls)
    ]
    message = SimpleNamespace(tool_calls=tool_calls, content=None)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=message)])


def final_response(text="OK"):
    message = SimpleNamespace(tool_calls=None, content=text)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)])


class TestAutonomous:
    def test_full_tool_loop(self):
        notify = {
            "description": "55 inch 4K TV",
            "deal_price": 200,
            "estimated_true_value": 450,
            "url": "https://x/tv",
        }
        client = ScriptedClient(
            [
                tool_response(("scan_the_internet_for_bargains", {})),
                tool_response(("estimate_true_value", {"description": "55 inch 4K TV"})),
                tool_response(("notify_user_of_deal", notify), ("notify_user_of_deal", notify)),
                final_response(),
            ]
        )
        m = messenger()
        agent = AutonomousPlanningAgent(FakeScanner(), FakePricer(), m, client=client)
        result = agent.plan()
        assert result.discount == 250
        assert len(m.sent) == 1  # the second notify call is refused

    def test_turn_limit(self):
        client = ScriptedClient(
            [tool_response(("estimate_true_value", {"description": "USB cable"}))] * 20
        )
        agent = AutonomousPlanningAgent(FakeScanner(), FakePricer(), messenger(), client=client)
        agent.MAX_TURNS = 3
        assert agent.plan() is None

    def test_bad_tool_arguments_do_not_crash(self):
        call = SimpleNamespace(
            id="x", function=SimpleNamespace(name="estimate_true_value", arguments="{bad")
        )
        agent = AutonomousPlanningAgent(
            FakeScanner(), FakePricer(), messenger(), client=ScriptedClient([])
        )
        [result] = agent.handle_tool_calls(SimpleNamespace(tool_calls=[call]))
        assert result["content"].startswith("Tool error")


class TestFramework:
    def test_memory_persists(self, tmp_path):
        path = tmp_path / "memory.json"
        planner = PlanningAgent(FakeScanner(), FakePricer(), messenger(), threshold=50)
        framework = DealAgentFramework(planner=planner, memory=Memory(path))
        framework.run()
        assert len(Memory(path).items) == 1

    def test_corrupt_memory(self, tmp_path):
        path = tmp_path / "memory.json"
        path.write_text("{oops", encoding="utf-8")
        assert Memory(path).items == []
