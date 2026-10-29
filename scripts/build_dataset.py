"""Step 1 - build the raw item dataset from Amazon-Reviews-2023 and push it to the Hub.

    python scripts/build_dataset.py --size 820000 --push

Produces ``<MY_HF_USER>/items_raw_full`` and ``items_raw_lite``.
"""

from __future__ import annotations

import argparse

from price_estimate.config import CATEGORIES, settings
from price_estimate.data.items import Item
from price_estimate.data.loaders import ItemLoader
from price_estimate.data.sampling import balanced_sample, deduplicate, split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=820_000, help="total items after sampling")
    parser.add_argument("--val", type=int, default=10_000)
    parser.add_argument("--test", type=int, default=10_000)
    parser.add_argument("--categories", nargs="*", default=CATEGORIES)
    parser.add_argument(
        "--push", action="store_true", help="push the result to the Hugging Face Hub"
    )
    args = parser.parse_args()

    items: list[Item] = []
    for category in args.categories:
        items.extend(ItemLoader(category).load())
    print(f"Loaded {len(items):,} items")

    items = deduplicate(items)
    print(f"After deduplication: {len(items):,}")

    sample = balanced_sample(items, args.size)
    train, val, test = split(sample, args.val, args.test)
    print(f"train={len(train):,} val={len(val):,} test={len(test):,}")

    if args.push:
        if not settings.my_hf_user:
            raise SystemExit("Set MY_HF_USER in .env before pushing")
        owner = settings.my_hf_user
        Item.push_to_hub(f"{owner}/items_raw_full", train, val, test)
        Item.push_to_hub(f"{owner}/items_raw_lite", train[:20_000], val[:1_000], test[:1_000])
        print("Pushed to the Hub")


if __name__ == "__main__":
    main()
