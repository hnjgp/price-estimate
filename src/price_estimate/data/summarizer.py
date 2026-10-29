"""Rewrite raw product text into short, uniform summaries.

Raw product pages are long and inconsistent. A small LLM rewrites each one into
the same five-line format, which makes every downstream model (bag-of-words,
neural network, fine-tuned LLM, RAG) work better and cheaper.

Two paths:
* ``Preprocessor``  - one item at a time through LiteLLM (any provider, incl. local Ollama)
* ``BatchJob``      - hundreds of thousands of items through Groq's batch API (cheap, async)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from price_estimate.config import settings
from price_estimate.data.items import Item

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

BATCH_MODEL = "openai/gpt-oss-20b"


def messages_for(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]


class Preprocessor:
    """Summarise a single description with any LiteLLM-supported model."""

    def __init__(self, model_name: str | None = None, base_url: str | None = None):
        self.model_name = model_name or settings.preprocessor_model
        self.reasoning_effort = "low" if "gpt-oss" in self.model_name else None
        self.base_url = base_url
        if self.model_name.startswith("ollama") and not base_url:
            self.base_url = "http://localhost:11434"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def preprocess(self, text: str) -> str:
        from litellm import completion  # imported lazily: litellm is slow to import

        response = completion(
            messages=messages_for(text),
            model=self.model_name,
            reasoning_effort=self.reasoning_effort,
            api_base=self.base_url,
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        self.total_cost += response._hidden_params.get("response_cost") or 0.0
        return response.choices[0].message.content


def make_request(item: Item, model: str = BATCH_MODEL) -> str:
    """One JSONL line in the OpenAI-compatible batch format."""
    body = {
        "model": model,
        "messages": messages_for(item.full or item.title),
        "reasoning_effort": "low",
    }
    return json.dumps(
        {"custom_id": str(item.id), "method": "POST", "url": "/v1/chat/completions", "body": body}
    )


def parse_result_line(line: str) -> tuple[int, str]:
    data = json.loads(line)
    content = data["response"]["body"]["choices"][0]["message"]["content"]
    return int(data["custom_id"]), content


@dataclass
class BatchJob:
    """One chunk of items submitted as a single Groq batch."""

    start: int
    end: int
    folder: Path
    file_id: str | None = None
    batch_id: str | None = None
    output_file_id: str | None = None
    done: bool = False

    @property
    def request_file(self) -> Path:
        return self.folder / "requests" / f"{self.start}_{self.end}.jsonl"

    @property
    def result_file(self) -> Path:
        return self.folder / "results" / f"{self.start}_{self.end}.jsonl"

    def write_requests(self, items: list[Item]) -> None:
        self.request_file.parent.mkdir(parents=True, exist_ok=True)
        with self.request_file.open("w", encoding="utf-8") as f:
            for item in items[self.start : self.end]:
                f.write(make_request(item) + "\n")

    def submit(self, client) -> None:
        # Binary mode: the upload API needs bytes (the original opened "rb" with an
        # encoding argument, which raises ValueError in Python).
        with self.request_file.open("rb") as f:
            self.file_id = client.files.create(file=f, purpose="batch").id
        self.batch_id = client.batches.create(
            completion_window="24h", endpoint="/v1/chat/completions", input_file_id=self.file_id
        ).id

    def poll(self, client) -> bool:
        response = client.batches.retrieve(self.batch_id)
        if response.status == "completed":
            self.output_file_id = response.output_file_id
            return True
        return False

    def download_and_apply(self, client, items: list[Item]) -> None:
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        client.files.content(self.output_file_id).write_to_file(str(self.result_file))
        with self.result_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    idx, summary = parse_result_line(line)
                    items[idx].summary = summary
        self.done = True


@dataclass
class BatchRunner:
    """Create, submit, poll and persist a set of batch jobs. Resumable via ``save``/``load``."""

    folder: Path
    batch_size: int = 1_000
    jobs: list[BatchJob] = field(default_factory=list)

    @property
    def state_file(self) -> Path:
        return self.folder / "batches.pkl"

    def create(self, items: list[Item]) -> None:
        for idx, item in enumerate(items):
            item.id = idx
        self.jobs = [
            BatchJob(start, min(start + self.batch_size, len(items)), self.folder)
            for start in range(0, len(items), self.batch_size)
        ]

    def submit_all(self, client, items: list[Item]) -> None:
        for job in self.jobs:
            if job.batch_id is None:
                job.write_requests(items)
                job.submit(client)

    def collect(self, client, items: list[Item]) -> int:
        for job in self.jobs:
            if not job.done and job.batch_id and job.poll(client):
                job.download_and_apply(client, items)
        return sum(job.done for job in self.jobs)

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("wb") as f:
            pickle.dump(self.jobs, f)

    def load(self) -> None:
        with self.state_file.open("rb") as f:
            self.jobs = pickle.load(f)
