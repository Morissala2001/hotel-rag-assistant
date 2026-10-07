"""Cutting sections into the units that are embedded, retrieved and put in the prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .documents import Section

CHUNKINGS = ("pages", "windows")
DEFAULT_CHUNKING = "windows"
# Number of passages put in the prompt, per strategy: the values that were evaluated.
DEFAULT_K = {"pages": 2, "windows": 3}


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    text: str  # what is shown to the language model
    embed_text: str  # what is embedded to find it

    @property
    def markdown(self) -> str:
        return f"## {self.title}\n\n{self.text}"


def split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split on sentence ends, ignoring line breaks; fragments of `min_chars` or fewer are dropped."""
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [p for p in parts if len(p) > min_chars]


def chunk_sections(sections: list[Section], strategy: str = DEFAULT_CHUNKING, window: int = 2) -> list[Chunk]:
    """Turn sections into chunks.

    - `windows` (default): overlapping windows of `window` consecutive sentences, each embedded
      together with its section title. The language model reads a few sentences instead of whole
      pages: retrieval is more precise (the right section is in the top 2 for every evaluation
      question) and prompts are shorter and faster.
    - `pages`: one chunk per section, embedded with its Markdown heading. Short documentation is
      already organised in topics, so a topic is a natural chunk.
    """
    if strategy == "pages":
        return [Chunk(s.source, s.title, s.text, s.markdown) for s in sections]
    if strategy == "windows":
        chunks = []
        for s in sections:
            sentences = split_sentences(s.text) or [" ".join(s.text.split())]
            for i in range(max(1, len(sentences) - window + 1)):
                text = " ".join(sentences[i:i + window])
                chunks.append(Chunk(s.source, s.title, text, f"{s.title}. {text}"))
        return chunks
    raise ValueError(f"strategy must be one of {CHUNKINGS}, got {strategy!r}")
