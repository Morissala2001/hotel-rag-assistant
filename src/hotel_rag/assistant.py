"""The assistant: three ways of answering a guest, from the least to the most reliable."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from .documents import Section
from .generation import TextGenerator
from .index import Hit, VectorIndex
from .prompts import DEFAULT_HOTEL, REFUSAL, build_prompt

MODES = ("rag", "full", "none")
MODE_LABELS = {
    "rag": "RAG: only the most relevant passages",
    "full": "Full documentation in the prompt",
    "none": "Language model alone",
}


@dataclass(frozen=True)
class Prepared:
    """Everything decided before the model writes a word."""

    prompt: str
    mode: str
    hits: list[Hit]  # the retrieved passages (empty unless mode is "rag")
    context_words: int  # how much documentation the model has to read
    gated: bool = False  # True when the question was refused without calling the model


@dataclass(frozen=True)
class Answer:
    text: str
    prepared: Prepared
    seconds: float

    @property
    def hits(self) -> list[Hit]:
        return self.prepared.hits


class Assistant:
    """Answers questions about a hotel.

    - `rag`: embed the question, retrieve the `k` closest chunks, put only those in the prompt.
    - `full`: put the whole documentation in the prompt.
    - `none`: ask the model with no documentation at all (it will make things up).

    `min_score` is an optional gate: in `rag` mode, a question whose best match is less similar
    than this is refused without calling the model.
    """

    def __init__(
        self,
        sections: list[Section],
        index: VectorIndex,
        generator: TextGenerator,
        hotel_name: str = DEFAULT_HOTEL,
        k: int = 3,
        min_score: float | None = None,
    ):
        self.sections = sections
        self.index = index
        self.generator = generator
        self.hotel_name = hotel_name
        self.k = k
        self.min_score = min_score

    def prepare(self, question: str, mode: str = "rag") -> Prepared:
        if mode == "none":
            return Prepared(question, mode, [], 0)
        if mode == "full":
            context = "\n\n".join(s.markdown for s in self.sections)
            return Prepared(build_prompt(context, question, self.hotel_name), mode, [], len(context.split()))
        if mode == "rag":
            hits = self.index.search(question, self.k)
            context = "\n\n".join(h.chunk.markdown for h in hits)
            gated = self.min_score is not None and hits[0].score < self.min_score
            return Prepared(build_prompt(context, question, self.hotel_name), mode, hits, len(context.split()), gated)
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    def ask(self, question: str, mode: str = "rag") -> Answer:
        start = time.time()
        prepared = self.prepare(question, mode)
        text = REFUSAL if prepared.gated else self.generator.generate(prepared.prompt)
        return Answer(text, prepared, time.time() - start)

    def ask_stream(self, question: str, mode: str = "rag") -> tuple[Prepared, Iterator[str]]:
        """Like `ask`, but the answer arrives piece by piece. Returns the preparation first."""
        prepared = self.prepare(question, mode)
        pieces = iter([REFUSAL]) if prepared.gated else self.generator.stream(prepared.prompt)
        return prepared, pieces
