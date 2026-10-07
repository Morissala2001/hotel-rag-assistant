"""The language model: a small instruction-tuned model from the Hugging Face Hub, run locally."""

from __future__ import annotations

from threading import Thread
from typing import Iterator, Protocol

# The default model answered 24 of 26 evaluation questions correctly, against 18 for the light one
# (see the README). It needs about 3 GB of disk and a few GB of memory.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LIGHT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # about 1 GB, twice as fast, but refuses valid questions
MAX_NEW_TOKENS = 80  # about sixty words: enough for one or two sentences


class TextGenerator(Protocol):
    """What the assistant needs from a language model."""

    def generate(self, prompt: str) -> str: ...

    def stream(self, prompt: str) -> Iterator[str]: ...


class HFGenerator:
    """A `transformers` text-generation pipeline.

    Decoding is greedy (`do_sample=False`): the same prompt always gives the same answer, which
    makes results reproducible and comparable. The model is downloaded from the Hub on first use
    and cached in `~/.cache/huggingface`; it is never stored in this repository.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, max_new_tokens: int = MAX_NEW_TOKENS):
        import transformers
        from transformers import pipeline

        transformers.logging.set_verbosity_error()  # keep the output readable
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipe = pipeline("text-generation", model=model_id)

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt: str) -> str:
        output = self._pipe(self._messages(prompt), max_new_tokens=self.max_new_tokens, do_sample=False)
        return output[0]["generated_text"][-1]["content"]

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield the answer piece by piece while the model writes it (for the chat interface)."""
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self._pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
        errors: list[Exception] = []

        def run() -> None:
            try:
                self._pipe(self._messages(prompt), max_new_tokens=self.max_new_tokens, do_sample=False, streamer=streamer)
            except Exception as error:  # noqa: BLE001 - re-raised in the caller's thread below
                errors.append(error)
                streamer.on_finalized_text("", stream_end=True)  # unblock the reader

        thread = Thread(target=run, daemon=True)
        thread.start()
        yield from streamer
        thread.join()
        if errors:
            raise errors[0]
