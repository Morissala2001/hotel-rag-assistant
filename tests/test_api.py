"""The REST API answers, validates its input and starts from the command line (fake encoder and generator)."""

import pytest
from conftest import FakeGenerator

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hotel_rag import cli  # noqa: E402
from hotel_rag.api import Settings, create_app  # noqa: E402
from hotel_rag.assistant import Assistant  # noqa: E402
from hotel_rag.prompts import REFUSAL  # noqa: E402


@pytest.fixture
def make_client(sections, window_index):
    def make_client(min_score=None, reply="No, the rooftop pool is not heated."):
        generator = FakeGenerator(reply)
        assistant = Assistant(sections, window_index, generator, k=3, min_score=min_score)
        client = TestClient(create_app(Settings(model="fake-model"), make_assistant=lambda: assistant))
        return client, generator

    return make_client


def test_health_reports_the_settings_once_the_models_are_loaded(make_client, window_index):
    client, _ = make_client()
    with client:  # starting the app loads the models
        body = client.get("/health").json()
    assert body == {"status": "ok", "model": "fake-model", "chunking": "windows", "k": 3,
                    "passages": len(window_index.chunks)}


def test_a_question_gets_the_answer_and_its_sources(make_client):
    client, generator = make_client()
    with client:
        response = client.post("/ask", json={"question": "  Is the rooftop pool heated?  "})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "No, the rooftop pool is not heated."
    assert body["mode"] == "rag" and not body["refused_without_model"]
    assert len(body["sources"]) == 3 and "pool" in body["sources"][0]["text"].lower()
    assert body["sources"][0]["score"] >= body["sources"][-1]["score"]  # best first
    assert body["context_words"] > 0 and body["seconds"] >= 0
    assert "Is the rooftop pool heated?" in generator.prompts[0]  # the question is stripped and sent


def test_the_three_modes_read_different_amounts_of_documentation(make_client):
    client, _ = make_client()
    with client:
        words = {mode: client.post("/ask", json={"question": "Can I bring my dog?", "mode": mode}).json()
                 for mode in ("rag", "full", "none")}
    assert words["none"]["context_words"] == 0 and words["none"]["sources"] == []
    assert 0 < words["rag"]["context_words"] < words["full"]["context_words"]
    assert words["full"]["sources"] == []


def test_the_similarity_gate_refuses_without_calling_the_model(make_client):
    client, generator = make_client(min_score=0.99)
    with client:
        body = client.post("/ask", json={"question": "Do you accept payments in Bitcoin?"}).json()
    assert body["refused_without_model"] and body["answer"] == REFUSAL
    assert generator.prompts == []


@pytest.mark.parametrize("payload", [{"question": "   "}, {"question": "x" * 501}, {"question": "Hi", "mode": "magic"},
                                     {"mode": "rag"}])
def test_invalid_questions_are_rejected_with_422(make_client, payload):
    client, generator = make_client()
    with client:
        assert client.post("/ask", json=payload).status_code == 422
    assert generator.prompts == []


def test_settings_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOTEL_RAG_DOCS", str(tmp_path))
    monkeypatch.setenv("HOTEL_RAG_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    monkeypatch.setenv("HOTEL_RAG_CHUNKING", "pages")
    monkeypatch.setenv("HOTEL_RAG_K", "2")
    monkeypatch.setenv("HOTEL_RAG_MIN_SCORE", "0.35")
    assert Settings.from_env() == Settings(tmp_path, "Qwen/Qwen2.5-0.5B-Instruct", "pages", 2, 0.35)


def test_hotel_rag_serve_starts_uvicorn_with_the_options(monkeypatch):
    import uvicorn

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: started.update(app=app, host=host, port=port))
    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "8123", "--chunking", "pages"]) == 0
    assert (started["host"], started["port"]) == ("0.0.0.0", 8123)
    assert {"/ask", "/health"} <= {route.path for route in started["app"].routes}  # models load only when it runs
