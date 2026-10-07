import pytest

from hotel_rag.prompts import DEFAULT_HOTEL, INSTRUCTION, REFUSAL, build_prompt


@pytest.mark.parametrize("query, title", [
    ("Wi-Fi network password", "Wi-Fi and connectivity"),
    ("airport transfer booked", "Parking and transport"),
    ("sauna massage rooms", "Wellness room"),
    ("dogs cats leash", "Pets"),
    ("baby cots high chairs", "Families and children"),
])
def test_search_finds_the_right_page(page_index, query, title):
    hits = page_index.search(query, k=2)
    assert hits[0].chunk.title == title


def test_hits_are_sorted_and_limited_to_k(page_index):
    hits = page_index.search("pool towels children", k=3)
    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(-1.0001 <= h.score <= 1.0001 for h in hits)


def test_window_search_returns_small_passages(window_index):
    hit = window_index.search("airport transfer booked", k=1)[0]
    assert hit.chunk.title == "Parking and transport"
    assert len(hit.chunk.text.split()) < 60


def test_prompt_format():
    prompt = build_prompt("## Pool\n\nOpen in May.", "Is it heated?", "Hotel X")
    assert prompt == (
        "You are the virtual assistant of Hotel X.\nHere is the hotel's official documentation:\n\n"
        "## Pool\n\nOpen in May.\n\nGuest question: Is it heated?\n" + INSTRUCTION
    )


def test_instruction_gives_the_model_an_honest_way_out():
    assert REFUSAL in INSTRUCTION
    assert DEFAULT_HOTEL in build_prompt("c", "q")
