"""Client for the fine-tuned Llama-3.2-3B pricer deployed on Modal.

The model itself is served by ``modal_services/pricer_service.py``; this class
only calls it remotely, so the local machine needs no GPU.
"""

from __future__ import annotations

from price_estimate.config import settings
from price_estimate.models.base import Pricer


class SpecialistPricer(Pricer):
    name = "fine_tuned_llama"

    def __init__(self, app_name: str | None = None, class_name: str = "Pricer"):
        import modal

        cls = modal.Cls.from_name(app_name or settings.modal_app_name, class_name)
        self.remote = cls()

    def price(self, text: str) -> float:
        return float(self.remote.price.remote(text))
