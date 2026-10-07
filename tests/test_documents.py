import pytest

from hotel_rag.documents import Section, load_sections


def test_sample_pdfs_give_fifteen_sections(sections):
    assert len(sections) == 15
    assert {s.source for s in sections} == {
        "dining_and_wellness.pdf", "families_pets_accessibility.pdf", "guest_information.pdf",
        "local_area_and_events.pdf", "rooms_and_rates.pdf",
    }


def test_titles_are_the_first_line_and_footers_are_dropped(sections):
    titles = [s.title for s in sections]
    assert "Wi-Fi and connectivity" in titles and "Rooftop pool" in titles
    for s in sections:
        assert "Internal documentation" not in s.text  # the footer
        assert not s.text.startswith(s.title)  # the title is not repeated in the body
        assert len(s.text.split()) > 40


def test_section_markdown():
    assert Section("a.pdf", "Pets", "Dogs are welcome.").markdown == "## Pets\n\nDogs are welcome."


def test_markdown_files_are_split_on_second_level_headings(tmp_path):
    (tmp_path / "guide.md").write_text(
        "# Hotel guide\nIgnored preamble.\n\n## Pool\nOpen from May.\n\n## Parking\nNo car park.\nUse the garage.\n",
        encoding="utf-8",
    )
    sections = load_sections(tmp_path)
    assert [(s.title, s.text) for s in sections] == [
        ("Pool", "Open from May."), ("Parking", "No car park.\nUse the garage."),
    ]


def test_files_are_read_in_name_order(tmp_path):
    (tmp_path / "b.md").write_text("## Second\ntext b\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("## First\ntext a\n", encoding="utf-8")
    assert [s.title for s in load_sections(tmp_path)] == ["First", "Second"]


def test_missing_or_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sections(tmp_path / "nope")
    with pytest.raises(ValueError, match="No PDF or Markdown"):
        load_sections(tmp_path)
