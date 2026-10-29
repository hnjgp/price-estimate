"""Step 2 - add an LLM-written summary to every item using Groq's batch API.

Resumable workflow (batches can take hours):

    python scripts/summarize.py submit     # create + upload all batches
    python scripts/summarize.py collect    # run repeatedly until everything is done
    python scripts/summarize.py push       # publish items_lite / items_full to the Hub
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from price_estimate.config import settings
from price_estimate.data.items import Item
from price_estimate.data.summarizer import BatchRunner


def load_items() -> list[Item]:
    owner = settings.my_hf_user or settings.data_hf_user
    train, val, test = Item.from_hub(settings.dataset_name("items_raw", owner=owner))
    return train + val + test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["submit", "collect", "push"])
    args = parser.parse_args()

    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    folder = Path("batches") / ("lite" if settings.lite_mode else "full")
    runner = BatchRunner(folder)
    items = load_items()

    if args.action == "submit":
        runner.create(items)
        runner.submit_all(client, items)
        runner.save()
        print(f"Submitted {len(runner.jobs)} batches")
        return

    runner.load()
    done = runner.collect(client, items)
    runner.save()
    print(f"{done}/{len(runner.jobs)} batches complete")

    if args.action == "push":
        missing = [i for i, item in enumerate(items) if not item.summary]
        if missing:
            raise SystemExit(f"{len(missing)} items still have no summary; run collect again")
        for item in items:
            item.full = None
            item.id = None
        n_val = 1_000 if settings.lite_mode else 10_000
        test, val, train = items[-n_val:], items[-2 * n_val : -n_val], items[: -2 * n_val]
        Item.push_to_hub(
            settings.dataset_name("items", owner=settings.my_hf_user), train, val, test
        )


if __name__ == "__main__":
    main()
