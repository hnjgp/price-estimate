"""A deep residual network that prices products from a hashed bag-of-words.

Why it works well for a "plain" network:
* HashingVectorizer turns text into a fixed 5,000-dim binary vector with no vocabulary to store.
* Prices are trained in log space and standardised, so a $5 and a $900 item get comparable
  gradients, and L1 loss keeps outliers from dominating.
* Residual blocks with LayerNorm let a 10-layer, 4096-wide network train stably.

Improvement over the course version: the normalisation constants (mean/std of
log-price) are saved **inside** the checkpoint. The original hard-coded them in
the inference code, so retraining on different data silently produced wrong
prices. ``load`` still accepts an old weights-only checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from price_estimate.data.items import Item
from price_estimate.models.base import Pricer

# Constants of the published course checkpoint (used only for legacy weights-only files)
LEGACY_Y_MEAN = 4.434937953948975
LEGACY_Y_STD = 1.0328539609909058


@dataclass
class DNNConfig:
    n_features: int = 5000
    num_layers: int = 10
    hidden_size: int = 4096
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 64
    seed: int = 42


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_vectorizer(n_features: int) -> HashingVectorizer:
    return HashingVectorizer(n_features=n_features, stop_words="english", binary=True)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class DeepNeuralNetwork(nn.Module):
    def __init__(
        self, input_size: int, num_layers: int = 10, hidden_size: int = 4096, dropout: float = 0.2
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(hidden_size, dropout) for _ in range(max(num_layers - 2, 0))
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)


class DNNPricer(Pricer):
    name = "deep_neural_network"

    def __init__(self, config: DNNConfig | None = None, device: torch.device | str | None = None):
        self.config = config or DNNConfig()
        self.device = torch.device(device) if device else best_device()
        self.vectorizer = make_vectorizer(self.config.n_features)
        self.model = self._build()
        self.y_mean = LEGACY_Y_MEAN
        self.y_std = LEGACY_Y_STD

    def _build(self) -> DeepNeuralNetwork:
        c = self.config
        return DeepNeuralNetwork(c.n_features, c.num_layers, c.hidden_size, c.dropout).to(
            self.device
        )

    # ----------------------------------------------------------------- data
    def _vectors(self, texts: list[str]) -> torch.Tensor:
        return torch.FloatTensor(self.vectorizer.transform(texts).toarray())

    def _normalise(self, prices: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor((np.log1p(prices) - self.y_mean) / self.y_std).unsqueeze(1)

    def _denormalise(self, output: torch.Tensor) -> torch.Tensor:
        return torch.expm1(output * self.y_std + self.y_mean)

    # ------------------------------------------------------------- training
    def fit(
        self, train: list[Item], val: list[Item], epochs: int = 5, verbose: bool = True
    ) -> list[dict]:
        """Train from scratch; returns per-epoch history (train loss, val loss, val MAE $)."""
        from tqdm.auto import tqdm

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        y_train = np.array([i.price for i in train], dtype=float)
        log_prices = np.log1p(y_train)
        self.y_mean, self.y_std = float(log_prices.mean()), float(log_prices.std() or 1.0)

        x_train = self._vectors([i.text for i in train])
        x_val = self._vectors([i.text for i in val]).to(self.device)
        y_val = torch.FloatTensor([i.price for i in val]).unsqueeze(1).to(self.device)
        y_val_norm = self._normalise(np.array([i.price for i in val], dtype=float)).to(self.device)

        loader = DataLoader(
            TensorDataset(x_train, self._normalise(y_train)),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        self.model = self._build()
        loss_fn = nn.L1Loss()
        optimizer = AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=0)

        history = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            losses = []
            for batch_x, batch_y in tqdm(loader, disable=not verbose, desc=f"epoch {epoch}"):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                out = self.model(x_val)
                val_loss = loss_fn(out, y_val_norm).item()
                val_mae = torch.abs(self._denormalise(out) - y_val).mean().item()
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
            history.append(record)
            if verbose:
                print(
                    f"epoch {epoch}: train {record['train_loss']:.4f} val {val_loss:.4f} val MAE ${val_mae:.2f}"
                )
        return history

    # ------------------------------------------------------------ inference
    def price(self, text: str) -> float:
        self.model.eval()
        with torch.no_grad():
            out = self.model(self._vectors([text]).to(self.device))[0]
            return max(float(self._denormalise(out).item()), 0.0)

    # --------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "config": asdict(self.config),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str | None = None) -> DNNPricer:
        device = torch.device(device) if device else best_device()
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            pricer = cls(DNNConfig(**checkpoint["config"]), device)
            pricer.model.load_state_dict(checkpoint["state_dict"])
            pricer.y_mean, pricer.y_std = checkpoint["y_mean"], checkpoint["y_std"]
        else:  # legacy file: a bare state_dict with the default architecture
            pricer = cls(DNNConfig(), device)
            pricer.model.load_state_dict(checkpoint)
        pricer.model.eval()
        return pricer
