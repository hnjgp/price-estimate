"""Central configuration.

Every tunable value (model names, dataset owners, paths) lives here and can be
overridden with an environment variable or a ``.env`` file. Code elsewhere never
hard-codes a username or a model name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is a core dependency, but keep import failures harmless
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]


@dataclass
class Settings:
    # Hugging Face
    data_hf_user: str = field(default_factory=lambda: os.getenv("DATA_HF_USER", "ed-donner"))
    my_hf_user: str = field(default_factory=lambda: os.getenv("MY_HF_USER", ""))
    lite_mode: bool = field(default_factory=lambda: _bool("LITE_MODE", True))

    # Models
    frontier_model: str = field(default_factory=lambda: os.getenv("FRONTIER_MODEL", "gpt-5.1"))
    scanner_model: str = field(default_factory=lambda: os.getenv("SCANNER_MODEL", "gpt-5-mini"))
    messaging_model: str = field(
        default_factory=lambda: os.getenv("MESSAGING_MODEL", "claude-sonnet-4-5")
    )
    preprocessor_model: str = field(
        default_factory=lambda: os.getenv("PREPROCESSOR_MODEL", "ollama/llama3.2")
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Fine-tuned specialist (QLoRA adapter served on Modal)
    base_model: str = "meta-llama/Llama-3.2-3B"
    modal_app_name: str = field(default_factory=lambda: os.getenv("MODAL_APP", "pricer-service"))

    # Paths
    artifacts_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    )
    vectorstore_dir: Path = field(
        default_factory=lambda: Path(os.getenv("VECTORSTORE_DIR", "products_vectorstore"))
    )
    memory_file: Path = field(default_factory=lambda: Path(os.getenv("MEMORY_FILE", "memory.json")))

    # Agent behaviour
    deal_threshold: float = field(
        default_factory=lambda: float(os.getenv("DEAL_THRESHOLD", "50"))
    )

    def dataset_name(self, kind: str = "items", owner: str | None = None) -> str:
        """Return e.g. ``ed-donner/items_lite`` for the configured mode."""
        size = "lite" if self.lite_mode else "full"
        return f"{owner or self.data_hf_user}/{kind}_{size}"

    @property
    def dnn_path(self) -> Path:
        return self.artifacts_dir / "deep_neural_network.pth"

    @property
    def ensemble_path(self) -> Path:
        return self.artifacts_dir / "ensemble_weights.json"


settings = Settings()
