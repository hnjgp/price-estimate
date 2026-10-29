"""Compare price models on the same test items and write a results table.

    python scripts/benchmark.py                       # classic baselines only (no API keys needed)
    python scripts/benchmark.py --dnn                 # + trained neural network
    python scripts/benchmark.py --llm openai/gpt-4.1-nano anthropic/claude-sonnet-4-5
    python scripts/benchmark.py --rag                 # + GPT with retrieved examples

Results go to reports/benchmark.csv, reports/benchmark.md and one HTML chart per model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from price_estimate.config import settings
from price_estimate.data.items import Item
from price_estimate.evaluation.evaluator import EvalResult, evaluate
from price_estimate.models.baselines import (
    BagOfWordsLinearPricer,
    ConstantPricer,
    FeatureLinearPricer,
    RandomForestPricer,
    RandomPricer,
    XGBoostPricer,
)


def to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    rows = [
        "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=200, help="number of test items")
    parser.add_argument("--train-limit", type=int, default=None, help="cap training items (speed)")
    parser.add_argument("--dnn", action="store_true")
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--llm", nargs="*", default=[], help="LiteLLM model names for zero-shot")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    train, _, test = Item.from_hub(settings.dataset_name())
    if args.train_limit:
        train = train[: args.train_limit]
    print(f"train={len(train):,} test={len(test):,}")

    out = Path(args.out)
    results: list[EvalResult] = []

    def run(predictor, title=None):
        results.append(
            evaluate(predictor, test, size=args.size, title=title, charts_dir=out / "charts")
        )

    run(RandomPricer())
    run(ConstantPricer().fit(train))
    run(FeatureLinearPricer().fit(train))
    run(BagOfWordsLinearPricer().fit(train))
    run(RandomForestPricer().fit(train, limit=15_000))
    run(XGBoostPricer().fit(train))

    if args.dnn:
        from price_estimate.models.dnn import DNNPricer

        run(DNNPricer.load(settings.dnn_path))

    for model in args.llm:
        from price_estimate.models.llm import ZeroShotPricer

        run(ZeroShotPricer(model))

    if args.rag:
        from price_estimate.models.llm import RAGPricer
        from price_estimate.rag import SimilarProducts, get_collection, get_encoder

        run(RAGPricer(SimilarProducts(get_collection(), get_encoder())))

    table = pd.DataFrame([r.to_dict() for r in results]).sort_values("mae")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "benchmark.csv", index=False)
    (out / "benchmark.md").write_text(to_markdown(table), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
