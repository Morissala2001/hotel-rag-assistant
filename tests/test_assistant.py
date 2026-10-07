import pytest
from conftest import FakeGenerator

from hotel_rag.assistant import Assistant
from hotel_rag.prompts import REFUSAL


def make(sections, index, reply="Fake answer.", **kwargs):
    generator = FakeGenerator(reply)
    return Assistant(sections, index, generator, **kwargs), generator


def test_rag_puts_only_the_retrieved_passages_in_the_prompt(sections, page_index):
    assistant, generator = make(sections, page_index, k=2)
    answer = assistant.ask("dogs cats leash", "rag")
    prompt = generator.prompts[0]
    assert "## Pets" in prompt
    assert "## Rooftop pool" not in prompt  # not retrieved
    assert len(answer.hits) == 2 and answer.hits[0].chunk.title == "Pets"
    assert 0 < answer.prepared.context_words < 400
    assert prompt.endswith("answer exactly: \"" + REFUSAL + "\"")


def test_full_mode_puts_the_whole_documentation_in_the_prompt(sections, page_index):
    assistant, generator = make(sections, page_index)
    answer = assistant.ask("dogs cats leash", "full")
    assert all(s.markdown in generator.prompts[0] for s in sections)
    assert answer.hits == []
    assert answer.prepared.context_words > 1000


def test_none_mode_sends_the_bare_question(sections, page_index):
    assistant, generator = make(sections, page_index)
    assistant.ask("Is the rooftop pool heated?", "none")
    assert generator.prompts == ["Is the rooftop pool heated?"]


def test_rag_context_is_much_smaller_than_full_context(sections, page_index):
    assistant, _ = make(sections, page_index, k=2)
    rag = assistant.prepare("airport transfer booked", "rag").context_words
    full = assistant.prepare("airport transfer booked", "full").context_words
    assert rag * 4 < full


def test_the_gate_refuses_without_calling_the_model(sections, page_index):
    assistant, generator = make(sections, page_index, min_score=0.99)
    answer = assistant.ask("zebra giraffe elephant", "rag")
    assert answer.text == REFUSAL and answer.prepared.gated
    assert generator.prompts == []


def test_the_gate_lets_relevant_questions_through(sections, page_index):
    # With the fake encoder, relevant questions score 0.19 to 0.43 and unrelated ones stay below 0.1.
    assistant, generator = make(sections, page_index, min_score=0.12)
    answer = assistant.ask("airport transfer booked", "rag")
    assert not answer.prepared.gated and answer.text == "Fake answer."
    assert len(generator.prompts) == 1
    assert assistant.ask("zebra giraffe elephant", "rag").prepared.gated
    assert len(generator.prompts) == 1  # the unrelated question never reached the model


def test_no_gate_by_default(sections, page_index):
    assistant, generator = make(sections, page_index)
    assert not assistant.ask("zebra giraffe elephant", "rag").prepared.gated
    assert len(generator.prompts) == 1


def test_streaming_returns_the_preparation_and_the_pieces(sections, page_index):
    assistant, _ = make(sections, page_index, reply="It opens at noon.")
    prepared, pieces = assistant.ask_stream("sauna massage rooms", "rag")
    assert prepared.hits[0].chunk.title == "Wellness room"
    assert "".join(pieces).strip() == "It opens at noon."


def test_streaming_a_gated_question_yields_the_refusal(sections, page_index):
    assistant, generator = make(sections, page_index, min_score=0.99)
    _, pieces = assistant.ask_stream("zebra giraffe elephant", "rag")
    assert "".join(pieces) == REFUSAL and generator.prompts == []


def test_unknown_mode(sections, page_index):
    assistant, _ = make(sections, page_index)
    with pytest.raises(ValueError, match="mode"):
        assistant.prepare("q", "magic")
