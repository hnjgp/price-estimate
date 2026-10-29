"""Turn item summaries into prompt/completion pairs for fine-tuning and push them to the Hub.

    python training/prepare_prompts.py --cutoff 110

Train/validation completions are rounded to whole dollars (easier target);
test completions keep the exact price for honest evaluation.
"""

from __future__ import annotations

import argparse

from price_estimate.config import settings
from price_estimate.data.items import Item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=110, help="max summary tokens")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    train, val, test = Item.from_hub(settings.dataset_name())
    counts = [item.count_tokens(tokenizer) for item in train + val + test]
    truncated = sum(c > args.cutoff for c in counts)
    print(
        f"{truncated:,} of {len(counts):,} summaries will be truncated ({truncated / len(counts):.1%})"
    )

    for item in train + val:
        item.make_prompts(tokenizer, args.cutoff, do_round=True)
    for item in test:
        item.make_prompts(tokenizer, args.cutoff, do_round=False)
    print(test[0].prompt + test[0].completion)

    if not settings.my_hf_user:
        raise SystemExit("Set MY_HF_USER in .env to push the prompt dataset")
    target = settings.dataset_name("items_prompts", owner=settings.my_hf_user)
    Item.push_prompts_to_hub(target, train, val, test)
    print(f"Pushed {target}")


if __name__ == "__main__":
    main()
