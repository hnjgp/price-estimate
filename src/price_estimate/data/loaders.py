"""Load and parse whole Amazon categories in parallel."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from tqdm import tqdm

from price_estimate.data.items import Item
from price_estimate.data.parser import parse

CHUNK_SIZE = 1000
SOURCE_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
WORKERS = max((os.cpu_count() or 2) - 1, 1)


class ItemLoader:
    def __init__(self, category: str):
        self.category = category
        self.dataset = None

    def from_chunk(self, chunk) -> list[Item]:
        items = (parse(row, self.category) for row in chunk)
        return [item for item in items if item is not None]

    def chunks(self):
        size = len(self.dataset)
        for start in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(start, min(start + CHUNK_SIZE, size)))

    def load(self, workers: int = WORKERS) -> list[Item]:
        from datasets import load_dataset

        started = datetime.now()
        print(f"Loading {self.category}", flush=True)
        self.dataset = load_dataset(
            SOURCE_DATASET, f"raw_meta_{self.category}", split="full", trust_remote_code=True
        )
        total = (len(self.dataset) + CHUNK_SIZE - 1) // CHUNK_SIZE
        results: list[Item] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunks()), total=total):
                results.extend(batch)
        minutes = (datetime.now() - started).total_seconds() / 60
        print(f"{self.category}: {len(results):,} items in {minutes:.1f} min", flush=True)
        return results
