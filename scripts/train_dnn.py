"""Train the deep residual network and save it to artifacts/deep_neural_network.pth.

python scripts/train_dnn.py --epochs 5
python scripts/train_dnn.py --epochs 1 --hidden 512 --layers 4   # quick CPU run
"""

from __future__ import annotations

import argparse

from price_estimate.config import settings
from price_estimate.data.items import Item
from price_estimate.evaluation.evaluator import evaluate
from price_estimate.models.dnn import DNNConfig, DNNPricer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--val-size", type=int, default=1000)
    args = parser.parse_args()

    train, val, test = Item.from_hub(settings.dataset_name())
    pricer = DNNPricer(DNNConfig(hidden_size=args.hidden, num_layers=args.layers))
    params = sum(p.numel() for p in pricer.model.parameters())
    print(f"Training on {len(train):,} items, {params:,} parameters, device={pricer.device}")
    pricer.fit(train, val[: args.val_size], epochs=args.epochs)
    pricer.save(settings.dnn_path)
    print(f"Saved to {settings.dnn_path}")
    evaluate(pricer, test)


if __name__ == "__main__":
    main()
