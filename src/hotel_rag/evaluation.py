"""Measuring the assistant on a labelled question set, instead of judging it by a few examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assistant import Assistant
from .index import VectorIndex


@dataclass(frozen=True)
class Question:
    question: str
    section: str  # title of the section that holds the answer
    facts: list[str]  # any of these appearing in the answer makes it correct


def load_questions(path: str | Path) -> tuple[list[Question], list[str]]:
    """Read `{"in_scope": [{question, section, facts}], "out_of_scope": [question]}`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    in_scope = [Question(q["question"], q["section"], q["facts"]) for q in data["in_scope"]]
    return in_scope, list(data["out_of_scope"])


def normalize(text: str) -> str:
    """Lower case, letters and digits only: '14:00' and '14 00' match, so do '10:30' and '10.30'."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def is_refusal(answer: str) -> bool:
    flat = normalize(answer)
    return "idontknow" in flat or "idonotknow" in flat


def is_correct(answer: str, facts: list[str]) -> bool:
    flat = normalize(answer)
    return not is_refusal(answer) and any(normalize(f) in flat for f in facts)


def auc(positives: list[float], negatives: list[float]) -> float:
    """Chance that a random positive scores higher than a random negative (ties count half)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def evaluate_retrieval(index: VectorIndex, questions: list[Question], out_of_scope: list[str], ks=(1, 2, 3)) -> dict:
    """Does the search put the right section first? Can its scores tell off-topic questions apart?"""
    ranks, top1_in = [], []
    for q in questions:
        hits = index.search(q.question, k=len(index.chunks))
        titles = [h.chunk.title for h in hits]
        ranks.append(titles.index(q.section) + 1 if q.section in titles else len(titles) + 1)
        top1_in.append(hits[0].score)
    top1_out = [index.search(q, k=1)[0].score for q in out_of_scope]
    ranks = np.array(ranks)
    return {
        "n_in_scope": len(questions),
        "n_out_of_scope": len(out_of_scope),
        "recall": {k: float(np.mean(ranks <= k)) for k in ks},
        "mrr": float(np.mean(1 / ranks)),
        "top1_in_scope": float(np.mean(top1_in)),
        "top1_out_of_scope": float(np.mean(top1_out)) if top1_out else None,
        "gate_auc": auc(top1_in, top1_out) if top1_out else None,
    }


def evaluate_generation(assistant: Assistant, questions: list[Question], out_of_scope: list[str], mode: str = "rag",
                        progress=None) -> dict:
    """Ask every question and check the answers against the expected facts and refusals."""
    records = []
    for i, q in enumerate(questions):
        answer = assistant.ask(q.question, mode)
        records.append({
            "question": q.question, "in_scope": True, "answer": answer.text, "seconds": answer.seconds,
            "context_words": answer.prepared.context_words, "expected_section": q.section,
            "section_retrieved": q.section in [h.chunk.title for h in answer.hits] if mode == "rag" else None,
            "correct": is_correct(answer.text, q.facts), "refused": is_refusal(answer.text),
        })
        if progress:
            progress(len(records), len(questions) + len(out_of_scope))
    for q in out_of_scope:
        answer = assistant.ask(q, mode)
        records.append({
            "question": q, "in_scope": False, "answer": answer.text, "seconds": answer.seconds,
            "context_words": answer.prepared.context_words, "refused": is_refusal(answer.text),
        })
        if progress:
            progress(len(records), len(questions) + len(out_of_scope))
    inside = [r for r in records if r["in_scope"]]
    outside = [r for r in records if not r["in_scope"]]
    return {
        "mode": mode,
        "n_in_scope": len(inside),
        "correct": sum(r["correct"] for r in inside),
        "false_refusals": sum(r["refused"] for r in inside),
        "wrong": sum(not r["correct"] and not r["refused"] for r in inside),
        "n_out_of_scope": len(outside),
        "refused_out_of_scope": sum(r["refused"] for r in outside),
        "mean_seconds": float(np.mean([r["seconds"] for r in records])),
        "mean_context_words": float(np.mean([r["context_words"] for r in records])),
        "records": records,
    }


def format_retrieval(r: dict) -> str:
    lines = [f"Retrieval ({r['n_in_scope']} in-scope and {r['n_out_of_scope']} off-topic questions)"]
    lines += [f"  recall@{k}: {v:.1%}" for k, v in r["recall"].items()]
    lines.append(f"  MRR: {r['mrr']:.3f}")
    if r["gate_auc"] is not None:
        lines.append(f"  top-1 similarity: {r['top1_in_scope']:.2f} in scope vs {r['top1_out_of_scope']:.2f} off-topic "
                     f"(separation AUC {r['gate_auc']:.2f})")
    return "\n".join(lines)


def format_generation(g: dict) -> str:
    n, m = g["n_in_scope"], g["n_out_of_scope"]
    return "\n".join([
        f"Generation, mode '{g['mode']}'",
        f"  correct answers: {g['correct']}/{n} ({g['correct'] / n:.1%})"
        f" | wrong: {g['wrong']} | refused although the answer is in the documents: {g['false_refusals']}",
        f"  off-topic questions refused: {g['refused_out_of_scope']}/{m}",
        f"  average: {g['mean_seconds']:.1f} s per answer, {g['mean_context_words']:.0f} words of context",
    ])
