"""The chat interface renders and answers, with the models replaced by fakes."""

from pathlib import Path

import pytest
from conftest import FakeEncoder, FakeGenerator

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import hotel_rag.generation  # noqa: E402
import hotel_rag.index  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(hotel_rag.index, "load_encoder", lambda *args, **kwargs: FakeEncoder())
    monkeypatch.setattr(hotel_rag.generation, "HFGenerator", lambda *args, **kwargs: FakeGenerator("It is free."))
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_first_screen(app):
    assert not app.exception
    assert app.title[0].value == "Casa Aurora assistant"
    assert len(app.button) >= 4  # the example questions


def test_a_question_gets_an_answer_with_sources(app):
    app.chat_input[0].set_value("Wi-Fi network password").run()
    assert not app.exception
    roles = [m.name for m in app.chat_message]
    assert roles == ["user", "assistant"]
    assert "It is free." in app.chat_message[1].markdown[0].value
    assert any("Wi-Fi and connectivity" in m.value for m in app.chat_message[1].markdown)
    assert any("words of documentation read" in c.value for c in app.chat_message[1].caption)


def test_the_gate_skips_the_model(app):
    app.sidebar.toggle[0].set_value(True).run()  # the similarity slider is only enabled once the gate is on
    app.sidebar.slider[1].set_value(0.5)
    app.chat_input[0].set_value("zebra giraffe elephant").run()
    assert not app.exception
    assert any("the model was not called" in c.value for c in app.chat_message[1].caption)
    assert "I don't know" in app.chat_message[1].markdown[0].value
