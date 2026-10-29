"""Builds the pricing ensemble from whatever members are available on this machine."""

from __future__ import annotations

from price_estimate.agents.base import YELLOW, Agent
from price_estimate.config import settings
from price_estimate.models.ensemble import EnsemblePricer, load_weights


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = YELLOW

    def __init__(self, collection=None, members: dict | None = None, preprocess: bool = True):
        self.log("initializing")
        members = members if members is not None else self._build_members(collection)
        if not members:
            raise RuntimeError("No pricing model could be loaded; see the log for details")
        preprocessor = self._preprocessor() if preprocess else None
        self.pricer = EnsemblePricer(members, load_weights(settings.ensemble_path), preprocessor)
        self.log(f"ready with members {list(members)} weights {self.pricer.weights}")

    def _try(self, label: str, factory):
        try:
            member = factory()
            self.log(f"loaded {label}")
            return member
        except Exception as error:
            self.log(f"skipping {label}: {error}")
            return None

    def _build_members(self, collection) -> dict:
        members = {}

        def frontier():
            from price_estimate.models.llm import RAGPricer
            from price_estimate.rag import SimilarProducts, get_encoder

            return RAGPricer(SimilarProducts(collection, get_encoder()))

        def specialist():
            from price_estimate.models.specialist import SpecialistPricer

            return SpecialistPricer()

        def neural_network():
            from price_estimate.models.dnn import DNNPricer

            return DNNPricer.load(settings.dnn_path)

        factories = {"specialist": specialist, "neural_network": neural_network}
        if collection is not None:
            factories = {"frontier": frontier, **factories}
        for name, factory in factories.items():
            member = self._try(name, factory)
            if member is not None:
                members[name] = member
        return members

    def _preprocessor(self):
        from price_estimate.data.summarizer import Preprocessor

        return Preprocessor().preprocess

    def price(self, description: str) -> float:
        self.log("estimating")
        estimate = self.pricer.price(description)
        parts = ", ".join(f"{k}=${v:.2f}" for k, v in self.pricer.last_breakdown.items())
        self.log(f"estimate ${estimate:.2f} ({parts})")
        return estimate
