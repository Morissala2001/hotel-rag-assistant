"""Shared fixtures. No test downloads a model: a fake encoder and a fake generator stand in for them."""

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from hotel_rag.chunking import chunk_sections
from hotel_rag.documents import load_sections
from hotel_rag.index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "sample_hotel"
STOP_WORDS = {"the", "a", "an", "is", "are", "do", "does", "can", "i", "you", "to", "of", "in", "at", "for", "and",
              "or", "what", "which", "how", "there", "it", "on", "by", "be", "from", "with", "all"}


class FakeEncoder:
    """Bag-of-words hashing: texts sharing words get similar vectors, with no model to download."""

    dim = 512

    def encode(self, sentences, normalize_embeddings=True):
        vectors = np.zeros((len(sentences), self.dim))
        for row, sentence in enumerate(sentences):
            for word in re.findall(r"[a-z0-9]+", sentence.lower()):
                if word not in STOP_WORDS:
                    vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms


class FakeGenerator:
    """Records the prompts it receives and replies with a fixed text."""

    def __init__(self, reply="Fake answer."):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply

    def stream(self, prompt):
        self.prompts.append(prompt)
        yield from (word + " " for word in self.reply.split())


@pytest.fixture(scope="session")
def sections():
    return load_sections(DOCS)


@pytest.fixture(scope="session")
def page_index(sections):
    return VectorIndex(chunk_sections(sections, "pages"), FakeEncoder())


@pytest.fixture(scope="session")
def window_index(sections):
    return VectorIndex(chunk_sections(sections, "windows"), FakeEncoder())
