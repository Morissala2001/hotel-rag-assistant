"""Checks with the real models. Run them with: HOTEL_RAG_RUN_SLOW=1 uv run pytest -m slow"""

import os

import pytest

from hotel_rag.assistant import Assistant
from hotel_rag.chunking import chunk_sections
from hotel_rag.documents import load_sections
from hotel_rag.evaluation import evaluate_retrieval, is_correct, load_questions
from hotel_rag.generation import LIGHT_MODEL, HFGenerator
from hotel_rag.index import VectorIndex, load_encoder

from conftest import DOCS, ROOT

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("HOTEL_RAG_RUN_SLOW") != "1", reason="set HOTEL_RAG_RUN_SLOW=1 to run"),
]


@pytest.fixture(scope="module")
def sections():
    return load_sections(DOCS)


@pytest.fixture(scope="module")
def index(sections):
    return VectorIndex(chunk_sections(sections), load_encoder())


def test_real_retrieval_finds_the_right_page_most_of_the_time(index):
    questions, out_of_scope = load_questions(ROOT / "eval" / "questions.json")
    report = evaluate_retrieval(index, questions, out_of_scope)
    assert report["recall"][2] >= 0.85


def test_real_model_answers_a_simple_question(sections, index):
    assistant = Assistant(sections, index, HFGenerator(LIGHT_MODEL))  # the light model keeps this check fast
    answer = assistant.ask("What time does check-in start?", "rag")
    assert is_correct(answer.text, ["14:00", "2 pm", "2:00 pm"])
