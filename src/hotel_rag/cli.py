"""Command line: `hotel-rag ask | compare | eval | serve`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assistant import MODES, Assistant
from .chunking import CHUNKINGS, DEFAULT_CHUNKING, DEFAULT_K, chunk_sections
from .documents import load_sections
from .evaluation import (evaluate_generation, evaluate_retrieval, format_generation, format_retrieval,
                         load_questions)
from .generation import DEFAULT_MODEL, HFGenerator
from .index import VectorIndex, load_encoder
from .prompts import DEFAULT_HOTEL

DEFAULT_DOCS = Path("data/sample_hotel")
DEFAULT_QUESTIONS = Path("eval/questions.json")


def _add_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="folder of PDF / Markdown documentation")
    parser.add_argument("--chunking", choices=CHUNKINGS, default=DEFAULT_CHUNKING, help="how documents are cut into passages")
    parser.add_argument("-k", type=int, default=None,
                        help="number of passages put in the prompt (rag mode); default: 3 for windows, 2 for pages")
    parser.add_argument("--min-score", type=float, default=None,
                        help="refuse questions whose best match is less similar than this (rag mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="language model id on the Hugging Face Hub")
    parser.add_argument("--hotel-name", default=DEFAULT_HOTEL)


def _k(args) -> int:
    return args.k or DEFAULT_K[args.chunking]


def _build_index(args):
    sections = load_sections(args.docs)
    return sections, VectorIndex(chunk_sections(sections, args.chunking), load_encoder())


def _build_assistant(args) -> Assistant:
    sections, index = _build_index(args)
    print(f"Loading {args.model} ...", file=sys.stderr)
    return Assistant(sections, index, HFGenerator(args.model), args.hotel_name, _k(args), args.min_score)


def _print_answer(question: str, answer) -> None:
    print(f"Guest    : {question}")
    if answer.hits:
        print("Sources  : " + ", ".join(f"{h.chunk.title} ({h.score:.2f})" for h in answer.hits))
    if answer.prepared.gated:
        print("Gate     : best match below --min-score, the model was not called")
    print(f"Assistant: {answer.text}")
    print(f"           [{answer.prepared.mode}] {answer.seconds:.1f} s, {answer.prepared.context_words} words of context")
    print("-" * 80)


def cmd_ask(args) -> None:
    assistant = _build_assistant(args)
    _print_answer(args.question, assistant.ask(args.question, args.mode))


def cmd_compare(args) -> None:
    assistant = _build_assistant(args)
    for mode in MODES:
        _print_answer(args.question, assistant.ask(args.question, mode))


def cmd_eval(args) -> None:
    questions, out_of_scope = load_questions(args.questions)
    sections, index = _build_index(args)
    print(f"{len(sections)} sections, {len(index.chunks)} chunks ({args.chunking})")
    report = {"retrieval": evaluate_retrieval(index, questions, out_of_scope), "generation": {}}
    print(format_retrieval(report["retrieval"]))
    if not args.retrieval_only:
        print(f"Loading {args.model} ...", file=sys.stderr)
        assistant = Assistant(sections, index, HFGenerator(args.model), args.hotel_name, _k(args), args.min_score)
        for mode in args.modes:
            result = evaluate_generation(
                assistant, questions, out_of_scope, mode,
                progress=lambda done, total: print(f"\r  {mode}: {done}/{total}", end="", file=sys.stderr),
            )
            print(file=sys.stderr)
            report["generation"][mode] = result
            print(format_generation(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Report written to {args.output}")


def cmd_serve(args) -> None:
    try:
        import uvicorn

        from .api import Settings, create_app
    except ImportError as error:
        raise ValueError(f"the API needs its extra dependencies ({error.name}): uv sync --extra api") from error
    settings = Settings(args.docs, args.model, args.chunking, args.k, args.min_score, args.hotel_name)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hotel-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ask", help="ask one question")
    _add_options(p)
    p.add_argument("question")
    p.add_argument("--mode", choices=MODES, default="rag")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("compare", help="ask the same question in the three modes")
    _add_options(p)
    p.add_argument("question")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("eval", help="measure retrieval and answers on the labelled questions")
    _add_options(p)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--retrieval-only", action="store_true", help="skip the language model (fast)")
    p.add_argument("--modes", nargs="+", choices=MODES, default=["rag"])
    p.add_argument("--output", type=Path, default=None, help="write the full report as JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="serve the REST API (FastAPI): POST /ask, GET /health, docs at /docs")
    _add_options(p)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept connections from other machines")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
