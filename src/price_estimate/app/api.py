"""REST API for price estimates.

    uvicorn price_estimate.app.api:app --reload
    curl -X POST localhost:8000/predict -H "content-type: application/json" \
         -d '{"description": "Shure MV7+ podcast microphone, USB-C and XLR"}'

The pricer is built once, lazily, from whatever members are available
(frontier RAG, fine-tuned specialist, neural network). Tests override
``get_pricer`` with a fake.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from price_estimate import __version__

app = FastAPI(title="Price Estimate API", version=__version__)


class PriceRequest(BaseModel):
    description: str = Field(min_length=3, max_length=4000)


class PriceResponse(BaseModel):
    estimate: float
    breakdown: dict[str, float]
    members: list[str]


@lru_cache(maxsize=1)
def get_pricer():
    from price_estimate.agents.ensemble_agent import EnsembleAgent
    from price_estimate.rag import get_collection

    try:
        collection = get_collection()
    except Exception:
        collection = None
    return EnsembleAgent(collection, preprocess=False).pricer


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/predict", response_model=PriceResponse)
def predict(request: PriceRequest, pricer: Annotated[object, Depends(get_pricer)]) -> PriceResponse:
    try:
        estimate = pricer.price(request.description)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return PriceResponse(
        estimate=round(estimate, 2),
        breakdown={k: round(v, 2) for k, v in pricer.last_breakdown.items()},
        members=list(pricer.weights),
    )
