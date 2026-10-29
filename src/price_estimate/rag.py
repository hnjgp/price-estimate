"""Vector store of training products, used to give an LLM priced examples (RAG).

Each training summary is embedded with a small sentence-transformer and stored in
Chroma together with its price and category. At prediction time we fetch the 5
most similar products and show them, with prices, to the LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from price_estimate.config import CATEGORIES, settings
from price_estimate.data.items import Item

COLLECTION = "products"
CATEGORY_COLORS = dict(
    zip(
        CATEGORIES,
        ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"],
        strict=True,
    )
)


def get_encoder(model_name: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name or settings.embedding_model)


def get_collection(path: str | Path | None = None):
    import chromadb

    client = chromadb.PersistentClient(path=str(path or settings.vectorstore_dir))
    return client.get_or_create_collection(COLLECTION)


def build_vectorstore(items: Sequence[Item], collection, encoder, batch_size: int = 1000) -> int:
    """Embed and add items; skips ids already present so it can be resumed."""
    from tqdm.auto import tqdm

    start = collection.count()
    for i in tqdm(range(start, len(items), batch_size)):
        batch = items[i : i + batch_size]
        collection.add(
            ids=[f"doc_{j}" for j in range(i, i + len(batch))],
            documents=[item.text for item in batch],
            embeddings=encoder.encode([item.text for item in batch]).astype(float).tolist(),
            metadatas=[{"category": item.category, "price": item.price} for item in batch],
        )
    return collection.count()


class SimilarProducts:
    """Look up the k nearest priced products for a description."""

    def __init__(self, collection, encoder, k: int = 5):
        self.collection, self.encoder, self.k = collection, encoder, k

    def find(self, text: str) -> tuple[list[str], list[float]]:
        vector = self.encoder.encode([text]).astype(float).tolist()
        results = self.collection.query(query_embeddings=vector, n_results=self.k)
        documents = list(results["documents"][0])
        prices = [float(m["price"]) for m in results["metadatas"][0]]
        return documents, prices


def plot_data(collection, max_points: int = 2000, dimensions: int = 3):
    """Return (documents, t-SNE coordinates, colours) for visualising the store."""
    from sklearn.manifold import TSNE

    result = collection.get(include=["embeddings", "documents", "metadatas"], limit=max_points)
    vectors = np.array(result["embeddings"])
    colors = [CATEGORY_COLORS.get(m["category"], "gray") for m in result["metadatas"]]
    perplexity = min(30, max(len(vectors) - 1, 1))
    reduced = TSNE(n_components=dimensions, random_state=42, perplexity=perplexity).fit_transform(
        vectors
    )
    return result["documents"], reduced, colors
