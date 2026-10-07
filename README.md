# 🏨 Hotel RAG assistant

*A hotel assistant that answers guests from the hotel's PDF documentation, shows the passages it used, and says "I don't know" instead of making things up.*

It runs entirely on your machine with open models: no API key, no data sent anywhere. The project has a **chat interface** (Streamlit), a **CLI** (`hotel-rag`) and a small **library** (`src/hotel_rag`) with a built-in **evaluation**, so that results are measured instead of judged on a few examples.

## Why retrieval?

The same question, *"Is the rooftop pool heated?"*, asked in the three modes the project can compare (default model, laptop CPU, no GPU):

| Mode | What the model sees | Answer | Time |
|---|---|---|---|
| **Model alone** | nothing | rambles about rooftop heating in general, no answer | 20 s |
| **Full documentation** | all 1,290 words, for every question | "No, the rooftop pool is not heated. The water temperature remains around 24 degrees Celsius **throughout the year**." (the documents say *in summer*) | 93 s |
| **RAG** (default) | the 3 most relevant passages, 73 words | "No, the rooftop pool is not heated; it has a temperature around 24 degrees Celsius in summer." | **15 s** |

A language model cannot know a hotel's rules, so alone it cannot answer. Putting the whole documentation in the prompt works for a small hotel but is slow, costs more with every extra page, and lets the model get lost in irrelevant text. Retrieval gives it only what it needs, and the interface shows exactly which passages were used.

## Results

Evaluated on 34 questions in [`eval/questions.json`](eval/questions.json): 26 that the documents answer, and 8 off-topic ones (Bitcoin, casino, weather...) that must be refused. An answer is correct when it contains the expected fact; an off-topic question is handled when the assistant gives the exact refusal sentence. Times are per answer on a laptop CPU (Intel i7-13700H, no GPU).

| Language model | Passages | Correct answers | Refused although the answer is in the documents | Off-topic questions refused | Time |
|---|---|---|---|---|---|
| Qwen2.5 0.5B | whole pages, top 2 | 18/26 (69%) | 7 | 7/8 | 7 s |
| Qwen2.5 0.5B | 2-sentence windows, top 3 | 17/26 (65%) | 5 | 5/8 | 7 s |
| Qwen2.5 1.5B | whole pages, top 2 | 22/26 (85%) | 2 | 8/8 | 25 s |
| **Qwen2.5 1.5B (default)** | **2-sentence windows, top 3** | **24/26 (92%)** | **1** | **8/8** | **18 s** |

What the measurements say:

- **The language model is the bottleneck.** The small model finds the right passage but often answers "I don't know" anyway. The larger one is 16 to 27 points better, and refuses every off-topic question.
- **Finer passages help the larger model, not the small one.** Cutting the documents into 2-sentence windows makes retrieval more precise and prompts shorter, which speeds up generation. Only the larger model can use that precision.

Retrieval alone (`hotel-rag eval --retrieval-only`), on the same questions:

| Passages | Right section first | Right section in the top 2 | MRR |
|---|---|---|---|
| whole pages | 92.3% | 96.2% | 0.950 |
| 2-sentence windows | 96.2% | 100% | 0.981 |

## Run it

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.13, and about 4 GB of free memory for the default model.

```bash
git clone https://github.com/Morissala2001/hotel-rag-assistant.git
cd hotel-rag-assistant
uv sync --all-extras
uv run streamlit run app.py
```

The models are downloaded from the Hugging Face Hub the first time they are used (about 3.6 GB: the language model, 3.1 GB, and the sentence encoder, 470 MB) and cached in `~/.cache/huggingface`. Nothing is stored in this repository, and the first question is slower while the models load. On a modest machine, pick the light model (about 1 GB, more than twice as fast, less accurate) in the sidebar.

**Command line**

```bash
uv run hotel-rag ask "Can I bring my dog?"
uv run hotel-rag compare "Is the rooftop pool heated?"   # the three modes side by side
uv run hotel-rag eval --retrieval-only                    # fast, no language model
uv run hotel-rag eval                                     # retrieval and answers
uv run hotel-rag eval --model Qwen/Qwen2.5-0.5B-Instruct --chunking pages -k 2
```

**Tests** (fast, offline: a fake encoder and a fake model stand in for the real ones)

```bash
uv run pytest
HOTEL_RAG_RUN_SLOW=1 uv run pytest -m slow   # also runs the real models
```

## How it works

1. **Documents** (`documents.py`): every page of every PDF is a section: the first line is its title, the last line is a footer that is dropped. Markdown files work too (one section per `## ` heading).
2. **Passages** (`chunking.py`): by default, overlapping windows of two sentences, each embedded with its section title. Whole pages are available too.
3. **Embeddings** (`index.py`): a multilingual sentence encoder (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions). Vectors are normalised, so cosine similarity is a dot product in a numpy matrix: with a few dozen passages, no vector database is needed.
4. **Retrieval**: the question is embedded and the 3 closest passages are kept.
5. **Prompt** (`prompts.py`): role, passages, guest question, then an instruction to answer in one or two sentences using only the documentation, and otherwise to reply with a fixed refusal sentence. That sentence gives the model an honest way out.
6. **Generation** (`generation.py`): `Qwen2.5-1.5B-Instruct` through a `transformers` pipeline, with greedy decoding so that the same question always gets the same answer. The interface streams the answer as it is written.
7. **Optional gate**: refuse a question whose best match is less similar than a threshold, without calling the model. It is off by default: the prompt already makes the default model refuse all 8 off-topic questions, and similarity alone separates on-topic from off-topic questions only imperfectly (AUC 0.89).

## Limits

- **The evaluation set is small** (26 + 8 questions, written by the author from the documents). A difference of one or two questions is within noise; read the table as a trend.
- **The default model still misses 2 of 26 questions**: it refuses *"What time do I have to leave the room on the last day?"* although the right passage was retrieved, and answers *"Can children use the wellness room?"* with an unrelated fact. The light model also mixes up numbers: it said the wellness room costs 95 euros, which is the price of a room.
- **It is slow on a CPU**: about 18 s per answer. A GPU, or a quantised model with `llama.cpp`, would be much faster.
- **English documentation and prompts.** The encoder is multilingual, so questions in other languages retrieve the right passages, but this is not evaluated.
- **No conversation memory**: every question is answered on its own.
- **Document layout**: PDFs are expected to have a title on the first line of each page and a footer on the last one; otherwise use Markdown.

## Structure

```
├── app.py                         # Streamlit chat interface
├── data/sample_hotel/             # 5 PDFs: a fictional hotel (15 pages)
├── eval/questions.json            # 26 answerable + 8 off-topic questions
├── scripts/make_sample_docs.py    # writes the PDFs (no dependency)
├── src/hotel_rag/
│   ├── documents.py               # PDF / Markdown -> sections
│   ├── chunking.py                # pages or sentence windows
│   ├── index.py                   # embeddings and similarity search
│   ├── prompts.py                 # prompt and refusal sentence
│   ├── generation.py              # language model, with streaming
│   ├── assistant.py               # the three modes, optional gate
│   ├── evaluation.py              # recall, MRR, answer accuracy
│   └── cli.py                     # the `hotel-rag` command
└── tests/
```

The hotel, its town and every detail of its documentation are invented, so that a language model cannot know them and a correct answer proves that retrieval worked.

## Ideas for improvement

- Keep the conversation in the prompt (multi-turn questions such as "and for children?").
- Hybrid search (BM25 plus embeddings) and a re-ranker.
- Let users upload their own PDFs in the interface.
- A larger, independently written evaluation set, and an LLM judge for open-ended answers.
- Quantised models for fast CPU inference.

## License

[MIT](LICENSE). The sample documentation is fictional and part of this repository.
