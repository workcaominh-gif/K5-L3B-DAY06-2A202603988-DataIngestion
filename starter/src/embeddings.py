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
    """Local sentence-transformers model. Each model gets the query/document handling its authors recommend:
    - BGE v1.5: instruction prefix on queries only
    - EmbeddingGemma: built-in "query" / "document" prompts (encode_query / encode_document)
    - all-MiniLM-L6-v2: no prompts
    All vectors are L2-normalized so cosine == dot product. No mock fallback: errors propagate."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32) -> None:
        import os

        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=self.device, token=os.environ.get("HF_TOKEN"))
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        name = model_name.lower()
        self.mode = "bge" if "bge" in name else "prompts" if "gemma" in name else "plain"

    def _enc(self, texts, fn):
        return fn(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False,
                  convert_to_numpy=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        fn = self._model.encode_document if self.mode == "prompts" else self._model.encode
        return self._enc(texts, fn).tolist()

    def embed_query(self, question: str) -> list[float]:
        if self.mode == "prompts":
            return self._enc([question], self._model.encode_query)[0].tolist()
        prefix = BGE_QUERY_PREFIX if self.mode == "bge" else ""
        return self._enc([prefix + question], self._model.encode)[0].tolist()
