import json

import pytest

from price_estimate.data.items import PREFIX, QUESTION, Item
from price_estimate.data.parser import MIN_CHARS, get_weight, parse, scrub
from price_estimate.data.sampling import balanced_sample, deduplicate, split
from price_estimate.data.summarizer import BatchJob, make_request, parse_result_line


def raw_row(price="49.99", text_len=800, details=None):
    return {
        "price": price,
        "title": "Acme Widget",
        "description": ["x" * text_len],
        "features": ["durable", "light"],
        "details": json.dumps(details or {"Item Weight": "2 Pounds", "Part Number": "ABC1234567"}),
    }


class TestWeight:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2 Pounds", 2.0),
            ("16 ounces", 1.0),
            ("453.592 Grams", 1.0),
            ("1 Kilograms", 1 / 0.453592),
            ("250 hundredths pounds", 2.5),
            ("1,200 Grams", 1200 / 453.592),
        ],
    )
    def test_units(self, raw, expected):
        assert get_weight({"Item Weight": raw}) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "", "heavy", "3", "12 stones", "hundredths"])
    def test_malformed_never_raises(self, raw):
        assert get_weight({"Item Weight": raw}) == 0.0


class TestParse:
    def test_valid_row(self):
        item = parse(raw_row(), "Electronics")
        assert item.price == pytest.approx(49.99)
        assert item.weight == pytest.approx(2.0)
        assert "Part Number" not in item.full

    @pytest.mark.parametrize("price", ["", "free", None, "0.1", "5000"])
    def test_rejects_bad_price(self, price):
        assert parse(raw_row(price=price), "Electronics") is None

    def test_rejects_short_text(self):
        assert parse(raw_row(text_len=10), "Electronics") is None

    def test_bad_details_json(self):
        row = raw_row()
        row["details"] = "{not json"
        assert parse(row, "Electronics") is not None

    def test_scrub_removes_part_numbers(self):
        text = scrub("Drill DCD771C2 kit", "great", None, {})
        assert "DCD771C2" not in text
        assert MIN_CHARS < len(scrub("t", "y" * 10_000, "z" * 10_000, {})) <= 4000


class TestItem:
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens)

    def test_prompts(self):
        item = Item(title="t", category="c", price=12.6, summary="one two three four")
        item.make_prompts(self.FakeTokenizer(), max_tokens=2, do_round=True)
        assert item.prompt == f"{QUESTION}\n\none two\n\n{PREFIX}"
        assert item.completion == "13.00"
        item.make_prompts(self.FakeTokenizer(), max_tokens=10, do_round=False)
        assert item.completion == "12.6"

    def test_text_fallback(self):
        assert Item(title="t", category="c", price=1, full="raw").text == "raw"
        assert Item(title="t", category="c", price=1).text == "t"


class TestSampling:
    def test_deduplicate(self, items):
        doubled = items + [i.model_copy() for i in items]
        assert len(deduplicate(doubled)) == len(items)

    def test_sample_prefers_expensive(self, items):
        sample = balanced_sample(items, 100)
        mean = lambda xs: sum(i.price for i in xs) / len(xs)  # noqa: E731
        assert len(sample) == 100
        assert mean(sample) > mean(items)

    def test_split(self, items):
        train, val, test = split(items, 20, 30)
        assert (len(train), len(val), len(test)) == (250, 20, 30)
        with pytest.raises(ValueError):
            split(items, 200, 200)


class TestBatch:
    def test_request_roundtrip(self, tmp_path, items):
        for idx, item in enumerate(items):
            item.id, item.full = idx, f"text {idx}"
        line = json.loads(make_request(items[3]))
        assert line["custom_id"] == "3"
        assert line["body"]["messages"][1]["content"] == "text 3"

        result = {
            "custom_id": "3",
            "response": {"body": {"choices": [{"message": {"content": "S"}}]}},
        }
        assert parse_result_line(json.dumps(result)) == (3, "S")

        job = BatchJob(0, 5, tmp_path)
        job.write_requests(items)
        assert len(job.request_file.read_text(encoding="utf-8").splitlines()) == 5
