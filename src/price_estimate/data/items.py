"""The ``Item`` record: one product with a known price.

A single class is used across the whole project (data prep, classic ML,
fine-tuning prompts and the agents), so there is one source of truth for the
prompt format.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    title: str
    category: str
    price: float
    full: str | None = None  # cleaned raw text scraped from the product page
    weight: float | None = None  # pounds, 0 when unknown
    summary: str | None = None  # short LLM-rewritten description
    prompt: str | None = None
    completion: str | None = None
    id: int | None = None

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    # ------------------------------------------------------------------ text
    @property
    def text(self) -> str:
        """Best available description: the summary if we have one, else the raw text."""
        return self.summary or self.full or self.title

    # --------------------------------------------------------------- prompts
    def make_prompts(self, tokenizer, max_tokens: int, do_round: bool) -> None:
        """Build a prompt/completion pair for supervised fine-tuning.

        The summary is truncated to ``max_tokens`` so that every training
        example fits comfortably in the context window. Training examples use a
        rounded price (``do_round=True``) because the model only has to learn
        whole dollars; test examples keep the exact price.
        """
        tokens = tokenizer.encode(self.text, add_special_tokens=False)
        body = (
            tokenizer.decode(tokens[:max_tokens]).rstrip()
            if len(tokens) > max_tokens
            else self.text
        )
        self.prompt = f"{QUESTION}\n\n{body}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def count_tokens(self, tokenizer) -> int:
        return len(tokenizer.encode(self.text, add_special_tokens=False))

    def count_prompt_tokens(self, tokenizer) -> int:
        return len(
            tokenizer.encode(
                (self.prompt or "") + (self.completion or ""), add_special_tokens=False
            )
        )

    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}

    # ------------------------------------------------------------------- hub
    @staticmethod
    def push_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ) -> None:
        from datasets import Dataset, DatasetDict

        DatasetDict(
            {
                "train": Dataset.from_list([i.model_dump() for i in train]),
                "validation": Dataset.from_list([i.model_dump() for i in val]),
                "test": Dataset.from_list([i.model_dump() for i in test]),
            }
        ).push_to_hub(dataset_name)

    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ) -> None:
        """Push prompt/completion pairs in the format TRL's SFTTrainer expects."""
        from datasets import Dataset, DatasetDict

        DatasetDict(
            {
                "train": Dataset.from_list([i.to_datapoint() for i in train]),
                "val": Dataset.from_list([i.to_datapoint() for i in val]),
                "test": Dataset.from_list([i.to_datapoint() for i in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        from datasets import load_dataset

        ds = load_dataset(dataset_name)
        fields = set(cls.model_fields)

        def build(rows):
            return [
                cls.model_validate({k: v for k, v in row.items() if k in fields}) for row in rows
            ]

        return build(ds["train"]), build(ds["validation"]), build(ds["test"])
