"""A small in-memory vector index: embeddings in a numpy matrix, cosine similarity by dot product."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chunking import Chunk

DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Encoder(Protocol):
    """Anything that turns texts into vectors, like a `SentenceTransformer`."""

    def encode(self, sentences: list[str], normalize_embeddings: bool = True) -> np.ndarray: ...


def load_encoder(model_id: str = DEFAULT_ENCODER) -> Encoder:
    """Load the multilingual sentence encoder (about 470 MB, downloaded once, 384 dimensions)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity with the question, between -1 and 1


class VectorIndex:
    """Embeds the chunks once; each search embeds the question and ranks the chunks.

    With a few dozen chunks a numpy matrix is all that is needed. A vector database such as
    FAISS or Chroma only pays off with far larger corpora.
    """

    def __init__(self, chunks: list[Chunk], encoder: Encoder):
        self.chunks = list(chunks)
        self.encoder = encoder
        # Vectors are normalised, so the dot product is the cosine similarity.
        self.embeddings = np.asarray(encoder.encode([c.embed_text for c in self.chunks], normalize_embeddings=True))

    def search(self, question: str, k: int = 2) -> list[Hit]:
        """The `k` chunks closest in meaning to the question, best first."""
        vector = np.asarray(self.encoder.encode([question], normalize_embeddings=True))[0]
        similarities = self.embeddings @ vector
        order = np.argsort(-similarities)[:k]
        return [Hit(self.chunks[i], float(similarities[i])) for i in order]
