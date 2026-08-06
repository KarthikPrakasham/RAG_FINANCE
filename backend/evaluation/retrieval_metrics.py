"""
Golden-dataset loading and page->chunk_id resolution — the two pieces of
plumbing every ablation script (Stages 1-7) needs that llama_index has no
equivalent for (it's project-specific: resolving our page-based ground
truth to a given run's content-hash chunk_ids). See
../EVALUATION_METHODOLOGY.md Part A.

The actual metric formulas (Recall@k / MRR / NDCG@k) used to live here as
hand-rolled functions — they now live in llamaindex_eval.py, computed via
llama_index.core.evaluation.retrieval's HitRate/MRR/discounted_gain
(verified mathematically identical to the old formulas before swapping;
see that module's docstring for the one real formula difference that
surfaced along the way, in how NDCG's IDCG gets capped).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_DEFAULT_GOLDEN_PATH = Path(__file__).resolve().parent / "golden_dataset.jsonl"


@dataclass
class GoldenQuery:
    id: str
    question: str
    category: str
    expected_source_doc: str | None
    relevant_pages: list[int]
    reference_answer: str
    retrieval_eval: bool


def load_golden_dataset(
    path: Path = _DEFAULT_GOLDEN_PATH,
    retrieval_only: bool = True,
) -> list[GoldenQuery]:
    """Load golden_dataset.jsonl.

    retrieval_only=True (default) filters out rows with retrieval_eval=false
    (the FRED CSV and guardrail questions, which don't map to page-based PDF
    retrieval) — pass False only when you need the full set, e.g. for
    acceptance_tests.py.
    """
    queries: list[GoldenQuery] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = GoldenQuery(
                id=row["id"],
                question=row["question"],
                category=row["category"],
                expected_source_doc=row.get("expected_source_doc"),
                relevant_pages=row.get("relevant_pages", []),
                reference_answer=row.get("reference_answer", ""),
                retrieval_eval=row.get("retrieval_eval", True),
            )
            if retrieval_only and not q.retrieval_eval:
                continue
            queries.append(q)
    return queries


def relevant_chunk_ids(query: GoldenQuery, chunks: Iterable) -> set[str]:
    """Resolve a golden query's (source_doc, relevant_pages) label to the
    actual chunk_ids present in *this run's* chunk list — chunk_id is a
    content hash that changes with parser/chunk-strategy, so this must be
    recomputed per ablation variant rather than stored in the golden file."""
    if query.expected_source_doc is None or not query.relevant_pages:
        return set()
    pages = set(query.relevant_pages)
    return {
        c.chunk_id
        for c in chunks
        if c.source_doc == query.expected_source_doc and c.page_num in pages
    }
