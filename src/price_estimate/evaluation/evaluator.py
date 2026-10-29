"""Run any price predictor over a test set and report how good it is.

A *predictor* is any callable ``item -> price`` (a float, or a string such as
"$123.45" that an LLM returned). This uniform interface is what lets us compare
a constant baseline, XGBoost, a neural network and GPT on exactly the same items.

Unlike the course notebooks, the evaluator works headless: it returns an
``EvalResult``, prints a one-line summary, and can save interactive charts to
HTML instead of requiring a Jupyter display.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from price_estimate.data.items import Item
from price_estimate.evaluation import metrics

Predictor = Callable[[Item], float | str]

ANSI = {"green": "\033[92m", "orange": "\033[93m", "red": "\033[91m"}
RESET = "\033[0m"
NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


def parse_price(value: float | int | str | None) -> float:
    """Extract a non-negative price from a model reply like "$1,299.99" or "about 40"."""
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return max(float(value), 0.0)
    match = NUMBER.search(value.replace("$", "").replace(",", ""))
    return max(float(match.group()), 0.0) if match else 0.0


def title_of(predictor: Predictor) -> str:
    name = getattr(predictor, "name", None) or getattr(
        predictor, "__name__", type(predictor).__name__
    )
    return name.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")


@dataclass
class EvalResult:
    title: str
    titles: list[str] = field(default_factory=list)
    truths: list[float] = field(default_factory=list)
    guesses: list[float] = field(default_factory=list)
    failures: int = 0

    @property
    def errors(self) -> list[float]:
        return [abs(g - t) for t, g in zip(self.truths, self.guesses, strict=True)]

    @property
    def colors(self) -> list[str]:
        return [metrics.color_for(e, t) for e, t in zip(self.errors, self.truths, strict=True)]

    @property
    def summary(self) -> dict[str, float]:
        return metrics.summarize(self.truths, self.guesses)

    def line(self) -> str:
        s = self.summary
        return (
            f"{self.title}: MAE ${s['mae']:,.2f} ± {s['mae_ci95']:,.2f} | RMSLE {s['rmsle']:.3f} | "
            f"R² {s['r2'] * 100:.1f}% | hits {s['hit_rate'] * 100:.0f}% | n={int(s['n'])}"
            + (f" | failures={self.failures}" if self.failures else "")
        )

    def to_dict(self) -> dict:
        return {"title": self.title, "failures": self.failures, **self.summary}

    def save_charts(self, folder: str | Path) -> Path:
        """Write a scatter (truth vs guess) and running-error chart to one HTML file."""
        import plotly.graph_objects as go

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        top = max(max(self.truths), max(self.guesses))
        palette = {"green": "seagreen", "orange": "orange", "red": "crimson"}
        scatter = go.Figure()
        scatter.add_trace(
            go.Scatter(
                x=self.truths,
                y=self.guesses,
                mode="markers",
                marker=dict(size=6, color=[palette[c] for c in self.colors]),
                text=self.titles,
                hovertemplate="%{text}<br>actual $%{x:,.2f}<br>guess $%{y:,.2f}<extra></extra>",
            )
        )
        scatter.add_trace(
            go.Scatter(
                x=[0, top], y=[0, top], mode="lines", line=dict(dash="dash"), hoverinfo="skip"
            )
        )
        scatter.update_layout(
            title=self.line(),
            xaxis_title="Actual price ($)",
            yaxis_title="Predicted price ($)",
            showlegend=False,
            template="plotly_white",
        )
        running, total = [], 0.0
        for i, err in enumerate(self.errors, 1):
            total += err
            running.append(total / i)
        trend = go.Figure(go.Scatter(y=running, mode="lines"))
        trend.update_layout(
            title="Running mean absolute error",
            xaxis_title="Items evaluated",
            yaxis_title="MAE ($)",
            template="plotly_white",
        )
        path = folder / f"{re.sub(r'[^a-z0-9]+', '_', self.title.lower()).strip('_')}.html"
        path.write_text(
            scatter.to_html(full_html=True, include_plotlyjs="cdn")
            + trend.to_html(full_html=False, include_plotlyjs=False),
            encoding="utf-8",
        )
        return path


class Tester:
    def __init__(
        self,
        predictor: Predictor,
        data: Sequence[Item],
        title: str | None = None,
        size: int = 200,
        workers: int = 5,
        verbose: bool = True,
    ):
        self.predictor = predictor
        self.data = data
        self.size = min(size, len(data))
        self.workers = workers
        self.verbose = verbose
        self.result = EvalResult(title or title_of(predictor))

    def _predict(self, i: int) -> tuple[str, float, float, bool]:
        item = self.data[i]
        title = item.title if len(item.title) <= 40 else item.title[:40] + "..."
        try:
            return title, parse_price(self.predictor(item)), item.price, True
        except Exception as error:  # one bad API call should not kill a 200-item run
            if self.verbose:
                print(f"\n[{self.result.title}] item {i} failed: {error}")
            return title, 0.0, item.price, False

    def run(self) -> EvalResult:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for title, guess, truth, ok in pool.map(self._predict, range(self.size)):
                self.result.titles.append(title)
                self.result.guesses.append(guess)
                self.result.truths.append(truth)
                self.result.failures += not ok
                if self.verbose:
                    color = metrics.color_for(abs(guess - truth), truth)
                    print(f"{ANSI[color]}${abs(guess - truth):.0f}{RESET} ", end="", flush=True)
        if self.verbose:
            print("\n" + self.result.line())
        return self.result


def evaluate(
    predictor: Predictor,
    data: Sequence[Item],
    size: int = 200,
    workers: int = 5,
    title: str | None = None,
    verbose: bool = True,
    charts_dir: str | Path | None = None,
) -> EvalResult:
    result = Tester(predictor, data, title=title, size=size, workers=workers, verbose=verbose).run()
    if charts_dir:
        result.save_charts(charts_dir)
    return result
