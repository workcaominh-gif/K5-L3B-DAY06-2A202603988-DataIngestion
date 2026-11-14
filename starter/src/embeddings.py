"""Real local embedding model via sentence-transformers. No mock fallback: errors propagate."""
from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, question: str) -> list[float]: ...


# BGE v1.5 recommends this instruction for short retrieval queries (passages get none).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class MyEmbedder:
    """BAAI/bge-small-en-v1.5 by default: 384 dims, L2-normalized so cosine == dot product."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily; heavy

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)
        return vecs.tolist()

    def embed_query(self, question: str) -> list[float]:
        vec = self._model.encode([self.query_prefix + question], normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)[0]
        return vec.tolist()
