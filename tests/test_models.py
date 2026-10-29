import pytest

from price_estimate.evaluation import metrics
from price_estimate.evaluation.evaluator import evaluate, parse_price
from price_estimate.models.baselines import (
    BagOfWordsLinearPricer,
    ConstantPricer,
    FeatureLinearPricer,
    RandomForestPricer,
    RandomPricer,
)
from price_estimate.models.ensemble import EnsemblePricer, fit_weights, load_weights, save_weights
from price_estimate.models.llm import RAGPricer, build_messages
from tests.conftest import make_items


@pytest.fixture(scope="module")
def data():
    items = make_items(600, seed=1)
    return items[:500], items[500:]


class TestMetrics:
    def test_basic(self):
        assert metrics.mae([10, 20], [12, 18]) == 2
        assert metrics.rmse([0, 0], [3, 4]) == pytest.approx((12.5) ** 0.5)
        assert metrics.r2([1, 2, 3], [1, 2, 3]) == 1
        assert metrics.rmsle([10], [10]) == 0

    def test_color_handles_zero_truth(self):
        assert metrics.color_for(100, 0) == "red"
        assert metrics.color_for(5, 0) == "green"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("$1,299.99", 1299.99),
            ("about 40 dollars", 40),
            ("no idea", 0),
            (-5, 0),
            (None, 0),
            (12, 12),
        ],
    )
    def test_parse_price(self, raw, expected):
        assert parse_price(raw) == pytest.approx(expected)


class TestBaselines:
    def test_text_models_beat_constant(self, data):
        train, test = data
        constant = evaluate(ConstantPricer().fit(train), test, verbose=False).summary["mae"]
        for model in (BagOfWordsLinearPricer(), RandomForestPricer(n_estimators=30)):
            mae = evaluate(model.fit(train), test, verbose=False).summary["mae"]
            assert mae < constant * 0.5, model.name

    def test_feature_and_random_run(self, data):
        train, test = data
        assert FeatureLinearPricer().fit(train)(test[0]) >= 0
        assert 1 <= RandomPricer().price("anything") < 1000

    def test_save_load(self, data, tmp_path):
        train, test = data
        model = BagOfWordsLinearPricer().fit(train)
        model.save(tmp_path / "m.joblib")
        loaded = BagOfWordsLinearPricer.load(tmp_path / "m.joblib")
        assert loaded.price(test[0].text) == pytest.approx(model.price(test[0].text))

    def test_evaluator_survives_failures(self, data):
        _, test = data

        def flaky(item):
            if item.price > 300:
                raise TimeoutError("api down")
            return "$50"

        result = evaluate(flaky, test, size=50, verbose=False)
        assert result.failures > 0
        assert len(result.guesses) == 50


class Fixed:
    def __init__(self, value, fail=False):
        self.value, self.fail = value, fail

    def price(self, text):
        if self.fail:
            raise ConnectionError("down")
        return self.value


class TestEnsemble:
    def test_weighted_average(self):
        ens = EnsemblePricer({"a": Fixed(100), "b": Fixed(200)}, {"a": 0.75, "b": 0.25})
        assert ens.price("x") == pytest.approx(125)

    def test_failing_member_is_dropped(self):
        ens = EnsemblePricer({"a": Fixed(100), "b": Fixed(0, fail=True)}, {"a": 0.5, "b": 0.5})
        assert ens.price("x") == pytest.approx(100)
        assert ens.last_breakdown == {"a": 100}

    def test_all_fail(self):
        with pytest.raises(RuntimeError):
            EnsemblePricer({"a": Fixed(1, fail=True)}).price("x")

    def test_fit_weights_prefers_accurate_member(self):
        truths = [10, 50, 100, 200, 400]
        good = [11, 49, 102, 198, 401]
        bad = [300, 10, 250, 20, 50]
        weights = fit_weights({"good": good, "bad": bad}, truths)
        assert weights["good"] > 0.9
        assert sum(weights.values()) == pytest.approx(1)

    def test_weights_roundtrip(self, tmp_path):
        save_weights({"a": 0.3, "b": 0.7}, tmp_path / "w.json")
        assert load_weights(tmp_path / "w.json") == {"a": 0.3, "b": 0.7}
        assert "frontier" in load_weights(tmp_path / "missing.json")


class FakeSimilar:
    def find(self, text):
        return ["Widget A", "Widget B"], [10.0, 30.0]


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("M", (), {"content": self.reply})
        choice = type("C", (), {"message": message})
        return type("R", (), {"choices": [choice]})


class TestLLM:
    def test_messages_include_context(self):
        content = build_messages("A drill", ["Saw"], [99.0])[0]["content"]
        assert "A drill" in content and "Price is $99.00" in content

    def test_rag_pricer(self):
        client = FakeClient("$24.50")
        pricer = RAGPricer(FakeSimilar(), client=client, model="gpt-5.1")
        assert pricer.price("a widget") == pytest.approx(24.5)
        call = client.calls[0]
        assert call["reasoning_effort"] == "none"
        assert "Widget B" in call["messages"][0]["content"]
