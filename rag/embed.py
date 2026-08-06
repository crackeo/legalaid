"""Embedding backends.

Two implementations behind one interface:

- VoyageEmbedder: real semantic embeddings via the Voyage AI API
  (Anthropic's recommended embedding provider). Needs VOYAGE_API_KEY.
- HashEmbedder: deterministic character-n-gram hashing. No network, no
  key — used for tests and as an offline fallback. Retrieval quality then
  rests mostly on BM25, which is strong for legal text.

Pick with embedder_from_env(): Voyage when the key is present, hash otherwise.
"""

import hashlib
import os

import numpy as np


class HashEmbedder:
    """Deterministic n-gram hashing embedder (offline fallback / tests)."""

    name = "hash-v1"
    dim = 512

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = " " + " ".join(text.lower().split()) + " "
            for n in (3, 4, 5):
                for j in range(len(t) - n):
                    h = int.from_bytes(
                        hashlib.blake2b(t[j:j + n].encode(), digest_size=4).digest(),
                        "big",
                    )
                    out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class VoyageEmbedder:
    """Voyage AI embeddings (https://docs.voyageai.com)."""

    name = "voyage-3.5"
    dim = 1024

    def __init__(self, api_key: str | None = None, model: str = "voyage-3.5"):
        self.api_key = api_key or os.environ["VOYAGE_API_KEY"]
        self.model = model
        self.name = model

    def embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        vectors: list[list[float]] = []
        for start in range(0, len(texts), 128):  # API batch limit
            resp = httpx.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts[start:start + 128]},
                timeout=120,
            )
            resp.raise_for_status()
            vectors.extend(item["embedding"] for item in resp.json()["data"])
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)


def embedder_from_env():
    if os.environ.get("VOYAGE_API_KEY"):
        return VoyageEmbedder()
    return HashEmbedder()
