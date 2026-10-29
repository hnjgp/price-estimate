"""Learn ensemble weights from validation predictions instead of guessing them.

Each available member prices N validation items; non-negative least squares then
finds the best blend. Weights are saved to artifacts/ensemble_weights.json and
picked up automatically by the agents.

    python scripts/fit_ensemble.py --size 200 --members frontier neural_network
"""

from __future__ import annotations

import argparse

from price_estimate.config import settings
from price_estimate.data.items import Item
from price_estimate.evaluation import metrics
from price_estimate.models.ensemble import fit_weights, save_weights


def build_members(names: list[str]) -> dict:
    members = {}
    if "frontier" in names:
        from price_estimate.models.llm import RAGPricer
        from price_estimate.rag import SimilarProducts, get_collection, get_encoder

        members["frontier"] = RAGPricer(SimilarProducts(get_collection(), get_encoder()))
    if "specialist" in names:
        from price_estimate.models.specialist import SpecialistPricer

        members["specialist"] = SpecialistPricer()
    if "neural_network" in names:
        from price_estimate.models.dnn import DNNPricer

        members["neural_network"] = DNNPricer.load(settings.dnn_path)
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument(
        "--members", nargs="+", default=["frontier", "specialist", "neural_network"]
    )
    args = parser.parse_args()

    _, val, _ = Item.from_hub(settings.dataset_name())
    val = val[: args.size]
    truths = [i.price for i in val]
    members = build_members(args.members)

    predictions = {}
    for name, pricer in members.items():
        predictions[name] = [pricer.price(i.text) for i in val]
        print(f"{name}: validation MAE ${metrics.mae(truths, predictions[name]):.2f}")

    weights = fit_weights(predictions, truths)
    blended = [sum(weights[n] * predictions[n][k] for n in weights) for k in range(len(val))]
    print("weights:", {k: round(v, 3) for k, v in weights.items()})
    print(f"ensemble: validation MAE ${metrics.mae(truths, blended):.2f}")
    save_weights(weights, settings.ensemble_path)
    print(f"Saved to {settings.ensemble_path}")


if __name__ == "__main__":
    main()
