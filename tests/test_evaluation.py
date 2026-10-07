import pytest
from conftest import DOCS, ROOT, FakeGenerator

from hotel_rag.assistant import Assistant
from hotel_rag.documents import load_sections
from hotel_rag.evaluation import (Question, auc, evaluate_generation, evaluate_retrieval, is_correct, is_refusal,
                                  load_questions, normalize)
from hotel_rag.prompts import REFUSAL


@pytest.mark.parametrize("answer, expected", [
    (REFUSAL, True),
    ("I do not know.", True),
    ("Check-in starts at 14:00.", False),
])
def test_refusal_detection(answer, expected):
    assert is_refusal(answer) is expected


def test_normalisation_ignores_punctuation_and_case():
    assert normalize("14:00") == normalize("14 00") == "1400"
    assert is_correct("The network is CasaAurora_Guest.", ["casaaurora_guest"])
    assert is_correct("Breakfast ends at 10:30.", ["10:30"])


def test_a_refusal_is_never_correct():
    assert not is_correct("I don't know, but maybe 14:00?", ["14:00"])


def test_auc():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5], [0.5]) == 0.5


def test_shipped_questions_match_the_documentation(sections):
    in_scope, out_of_scope = load_questions(ROOT / "eval" / "questions.json")
    assert (len(in_scope), len(out_of_scope)) == (26, 8)
    titles = {s.title for s in sections}
    assert all(q.section in titles for q in in_scope)
    assert all(q.facts for q in in_scope)


def test_retrieval_report_on_distinctive_questions(page_index):
    questions = [
        Question("Wi-Fi network password", "Wi-Fi and connectivity", []),
        Question("airport transfer booked", "Parking and transport", []),
        Question("sauna massage rooms", "Wellness room", []),
    ]
    report = evaluate_retrieval(page_index, questions, ["zebra giraffe elephant"])
    assert report["recall"][1] == 1.0 and report["mrr"] == 1.0
    assert report["gate_auc"] == 1.0  # on-topic questions score higher than the off-topic one


def test_generation_report_counts_refusals_and_correct_answers(sections, page_index):
    in_scope, out_of_scope = load_questions(ROOT / "eval" / "questions.json")
    refusing = Assistant(sections, page_index, FakeGenerator(REFUSAL))
    report = evaluate_generation(refusing, in_scope, out_of_scope, "rag")
    assert report["false_refusals"] == 26 and report["correct"] == 0
    assert report["refused_out_of_scope"] == 8

    known = [Question("What time does check-in start?", "Front desk and arrival", ["14:00"])]
    chatty = Assistant(sections, page_index, FakeGenerator("Check-in starts at 14:00."))
    report = evaluate_generation(chatty, known, ["Is there a casino?"], "rag")
    assert report["correct"] == 1 and report["refused_out_of_scope"] == 0
    assert report["records"][0]["answer"] == "Check-in starts at 14:00."


def test_sample_documents_are_reproducible(tmp_path):
    """Regenerating the PDFs gives byte-identical files, so the committed ones are not hand edited."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_sample_docs", ROOT / "scripts" / "make_sample_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for path in module.write_documents(tmp_path):
        assert path.read_bytes() == (DOCS / path.name).read_bytes()
    assert [s.title for s in load_sections(tmp_path)] == [s.title for s in load_sections(DOCS)]
