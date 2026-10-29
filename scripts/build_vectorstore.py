"""Embed the training items into the local Chroma vector store used by the RAG pricer.

python scripts/build_vectorstore.py --limit 400000
"""

from __future__ import annotations

import argparse

from price_estimate.config import settings
from price_estimate.data.items import Item
from price_estimate.rag import build_vectorstore, get_collection, get_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    train, _, _ = Item.from_hub(settings.dataset_name())
    items = train[: args.limit] if args.limit else train
    count = build_vectorstore(items, get_collection(), get_encoder())
    print(f"Vector store at {settings.vectorstore_dir} now holds {count:,} products")


if __name__ == "__main__":
    main()
