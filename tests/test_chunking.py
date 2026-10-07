import pytest

from hotel_rag.chunking import chunk_sections, split_sentences
from hotel_rag.documents import Section


def test_pages_make_one_chunk_per_section(sections):
    chunks = chunk_sections(sections, "pages")
    assert len(chunks) == len(sections)
    assert chunks[0].embed_text == sections[0].markdown  # embedded with its heading
    assert chunks[0].markdown == sections[0].markdown


def test_windows_overlap_by_one_sentence():
    section = Section("a.pdf", "Pool", "The pool opens in May. It is not heated at all. Towels are provided there.")
    chunks = chunk_sections([section], "windows", window=2)
    assert [c.text for c in chunks] == [
        "The pool opens in May. It is not heated at all.",
        "It is not heated at all. Towels are provided there.",
    ]
    assert chunks[0].embed_text.startswith("Pool. ")  # each window is embedded with its section title


def test_windows_cover_every_section(sections):
    chunks = chunk_sections(sections, "windows")
    assert len(chunks) > len(sections)
    assert {c.title for c in chunks} == {s.title for s in sections}
    assert all(len(c.text.split()) < 80 for c in chunks)  # a few sentences, not a whole page


def test_a_short_section_gives_a_single_window():
    section = Section("a.pdf", "Tiny", "Only one sentence is here, and it is long enough.")
    assert len(chunk_sections([section], "windows")) == 1


def test_split_sentences_ignores_line_breaks_and_fragments():
    text = "First sentence is here and long.\nSecond one follows right after! Ok. Third one ends it all?"
    assert split_sentences(text) == [
        "First sentence is here and long.", "Second one follows right after!", "Third one ends it all?",
    ]


def test_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        chunk_sections([], "paragraphs")
