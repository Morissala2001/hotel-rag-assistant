"""Chat interface of the hotel assistant.

    uv run --extra app streamlit run app.py

The models run on your machine and are downloaded from the Hugging Face Hub on first use.
"""

import time
from pathlib import Path

import streamlit as st

from hotel_rag.assistant import MODE_LABELS, MODES, Assistant
from hotel_rag.chunking import DEFAULT_K, chunk_sections
from hotel_rag.documents import load_sections
from hotel_rag.generation import DEFAULT_MODEL, LIGHT_MODEL, HFGenerator
from hotel_rag.index import VectorIndex, load_encoder

DOCS = Path(__file__).parent / "data" / "sample_hotel"
MODELS = {
    DEFAULT_MODEL: "Qwen2.5 1.5B (default: more accurate, about 3 GB, ~20 s per answer on CPU)",
    LIGHT_MODEL: "Qwen2.5 0.5B (light: about 1 GB, faster, but refuses valid questions)",
}
SHORT_MODES = {"rag": "RAG", "full": "Full documentation", "none": "Model alone"}
EXAMPLES = [
    "What time does check-in start?",
    "Can I bring my dog?",
    "Is the rooftop pool heated?",
    "Do you accept payments in Bitcoin?",
]

st.set_page_config(page_title="Hotel assistant", page_icon=":material/hotel:", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="Loading the sentence encoder…")
def get_encoder():
    return load_encoder()


@st.cache_resource(show_spinner="Loading the language model (the first time, it is downloaded)…")
def get_generator(model_id: str):
    return HFGenerator(model_id)


@st.cache_resource(show_spinner="Reading the documentation…")
def get_index(chunking: str):
    sections = load_sections(DOCS)
    return sections, VectorIndex(chunk_sections(sections, chunking), get_encoder())


with st.sidebar:
    st.header("Settings")
    mode = st.segmented_control(
        "Answer mode", MODES, format_func=SHORT_MODES.get, default="rag", selection_mode="single",
    ) or "rag"
    st.caption(MODE_LABELS[mode])
    model_id = st.selectbox("Language model", list(MODELS), format_func=MODELS.get)
    chunking = st.selectbox(
        "Passages", ["windows", "pages"],
        format_func={"windows": "Two-sentence windows (default)", "pages": "Whole pages"}.get,
    )
    k = st.slider("Passages in the prompt", 1, 4, DEFAULT_K[chunking], disabled=mode != "rag")
    gate = st.toggle("Refuse off-topic questions", disabled=mode != "rag")
    min_score = st.slider("Minimum similarity", 0.1, 0.8, 0.35, 0.05, disabled=not gate or mode != "rag")
    if st.button("Clear conversation", icon=":material/delete:"):
        st.session_state.messages = []

st.title("Casa Aurora assistant")
st.caption("Answers from the hotel's PDF documentation, with its sources. Everything is fictional.")

messages = st.session_state.setdefault("messages", [])


def render_details(sources, caption):
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for title, score, text in sources:
                st.markdown(f"**{title}** · similarity {score:.2f}")
                st.caption(text)
    st.caption(caption)


for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message["role"] == "assistant":
            render_details(message["sources"], message["caption"])

examples = st.empty()  # the suggestions disappear as soon as a question is asked
if not messages:
    with examples.container():
        st.write("Try a question:")
        for example in EXAMPLES:
            if st.button(example, key=example):
                st.session_state.pending = example
                st.rerun()

question = st.chat_input("Ask about the hotel") or st.session_state.pop("pending", None)
if question:
    examples.empty()
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        sections, index = get_index(chunking)
        assistant = Assistant(sections, index, get_generator(model_id), k=k, min_score=min_score if gate else None)
        prepared, pieces = assistant.ask_stream(question, mode)
        start = time.time()
        text = st.write_stream(pieces)
        caption = f"{SHORT_MODES[mode]} · {time.time() - start:.1f} s · {prepared.context_words} words of documentation read"
        if prepared.gated:
            caption += " · the model was not called (best match below the similarity threshold)"
        sources = [(h.chunk.title, h.score, h.chunk.text) for h in prepared.hits]
        render_details(sources, caption)
    messages.append({"role": "user", "text": question})
    messages.append({"role": "assistant", "text": text, "sources": sources, "caption": caption})
