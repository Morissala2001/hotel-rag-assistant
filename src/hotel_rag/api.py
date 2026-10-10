"""REST API of the assistant: `hotel-rag serve`, or `uvicorn hotel_rag.api:app`.

    POST /ask     {"question": "Is the rooftop pool heated?", "mode": "rag"}
                  -> the answer, the passages it comes from, and the time it took
    GET  /health  -> the model and settings in use, once the models are loaded

The models are loaded once, when the server starts. Interactive documentation is served at /docs.
The settings come from environment variables (handy in a container) or from `hotel-rag serve` options:
HOTEL_RAG_DOCS, HOTEL_RAG_MODEL, HOTEL_RAG_CHUNKING, HOTEL_RAG_K, HOTEL_RAG_MIN_SCORE, HOTEL_RAG_HOTEL_NAME.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints

from .assistant import Assistant
from .chunking import DEFAULT_CHUNKING, DEFAULT_K, chunk_sections
from .documents import load_sections
from .generation import DEFAULT_MODEL, HFGenerator
from .index import VectorIndex, load_encoder
from .prompts import DEFAULT_HOTEL

DEFAULT_DOCS = Path(__file__).resolve().parents[2] / "data" / "sample_hotel"


@dataclass(frozen=True)
class Settings:
    docs: Path = DEFAULT_DOCS
    model: str = DEFAULT_MODEL
    chunking: str = DEFAULT_CHUNKING
    k: int | None = None  # None: the default of the chunking (3 windows, 2 pages)
    min_score: float | None = None
    hotel_name: str = DEFAULT_HOTEL

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            docs=Path(env("HOTEL_RAG_DOCS", DEFAULT_DOCS)),
            model=env("HOTEL_RAG_MODEL", DEFAULT_MODEL),
            chunking=env("HOTEL_RAG_CHUNKING", DEFAULT_CHUNKING),
            k=int(env("HOTEL_RAG_K")) if env("HOTEL_RAG_K") else None,
            min_score=float(env("HOTEL_RAG_MIN_SCORE")) if env("HOTEL_RAG_MIN_SCORE") else None,
            hotel_name=env("HOTEL_RAG_HOTEL_NAME", DEFAULT_HOTEL),
        )

    def build_assistant(self) -> Assistant:
        sections = load_sections(self.docs)
        index = VectorIndex(chunk_sections(sections, self.chunking), load_encoder())
        return Assistant(sections, index, HFGenerator(self.model), self.hotel_name,
                         self.k or DEFAULT_K[self.chunking], self.min_score)


class Question(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)] = Field(
        examples=["Is the rooftop pool heated?"])
    mode: Literal["rag", "full", "none"] = Field("rag", description="rag: retrieved passages only (the default); "
                                                 "full: the whole documentation; none: the model alone")


class Source(BaseModel):
    title: str
    score: float = Field(description="cosine similarity with the question")
    text: str


class Reply(BaseModel):
    answer: str
    mode: str
    sources: list[Source] = Field(description="the passages given to the model, best first (rag mode only)")
    seconds: float
    context_words: int = Field(description="how much documentation the model had to read")
    refused_without_model: bool = Field(description="true when the question was refused by the similarity gate")


class Health(BaseModel):
    status: str
    model: str
    chunking: str
    k: int
    passages: int


def create_app(settings: Settings | None = None, make_assistant: Callable[[], Assistant] | None = None) -> FastAPI:
    """The API. `make_assistant` replaces the real models (tests use fakes)."""
    settings = settings or Settings.from_env()
    state: dict[str, Assistant] = {}
    # One answer at a time: on a CPU, parallel generations only slow each other down
    generating = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["assistant"] = (make_assistant or settings.build_assistant)()
        yield
        state.clear()

    app = FastAPI(title="Hotel RAG assistant", version="0.1.0", lifespan=lifespan,
                  description="Answers guests from the hotel's PDF documentation, with its sources.")

    @app.get("/health", response_model=Health)
    def health() -> Health:
        assistant = state["assistant"]
        return Health(status="ok", model=settings.model, chunking=settings.chunking, k=assistant.k,
                      passages=len(assistant.index.chunks))

    @app.post("/ask", response_model=Reply)
    def ask(question: Question) -> Reply:  # a plain function: FastAPI runs it in a worker thread
        with generating:
            answer = state["assistant"].ask(question.question, question.mode)
        return Reply(
            answer=answer.text, mode=answer.prepared.mode, seconds=round(answer.seconds, 2),
            sources=[Source(title=h.chunk.title, score=round(h.score, 3), text=h.chunk.text) for h in answer.hits],
            context_words=answer.prepared.context_words, refused_without_model=answer.prepared.gated,
        )

    return app


app = create_app()  # what `uvicorn hotel_rag.api:app` serves; the models load when it starts
