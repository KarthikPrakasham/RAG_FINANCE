"""
Calibrates Settings.grounding_threshold against real cosine-similarity
scores from the live Pinecone index, using golden_dataset.jsonl as ground
truth for which retrieved chunks are actually correct.

Why this exists: grounding_threshold=0.65 (the shipped default) was never
validated — same category of issue as chunk_size=500 and hybrid_alpha=0.5
before it. A quick manual check found the correct chunk for a real question
scored only 0.586, meaning the threshold as shipped would reject correct
retrievals outright. This script quantifies the actual score distributions
for correct vs. incorrect matches across the full golden set, so the
threshold is picked from data instead of a guess.

Run from backend/:  python -m evaluation.calibrate_grounding_threshold
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from evaluation.retrieval_metrics import load_golden_dataset, relevant_chunk_ids
from ingestion.parsers.parser import chunk_pages, parse_all_pdfs
from retrieval.dense_retriever import retrieve_dense
from retrieval.embed_query import embed_query

# Must match production ingestion exactly (ingest_embed.py) so chunk_ids
# line up with what's actually sitting in the live index.
PARSER_BACKEND = "pdfplumber"
CHUNK_STRATEGY = "sentence"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
TOP_K = 10


def run() -> tuple[list[float], list[float]]:
    golden = load_golden_dataset()
    pages = parse_all_pdfs(backend=PARSER_BACKEND)
    chunks = chunk_pages(
        pages, strategy=CHUNK_STRATEGY, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )

    correct_scores: list[float] = []
    incorrect_scores: list[float] = []

    for query in golden:
        expected = relevant_chunk_ids(query, chunks)
        if not expected:
            continue
        vec = embed_query(query.question)
        retrieved = retrieve_dense(vec, top_k=TOP_K)
        for r in retrieved:
            if r.chunk_id in expected:
                correct_scores.append(r.score)
            else:
                incorrect_scores.append(r.score)

    return correct_scores, incorrect_scores


def _stats(name: str, scores: list[float]) -> None:
    if not scores:
        print(f"{name}: (no scores)")
        return
    scores_sorted = sorted(scores)
    n = len(scores_sorted)
    mean = sum(scores_sorted) / n
    print(
        f"{name}: n={n} min={scores_sorted[0]:.3f} p25={scores_sorted[n // 4]:.3f} "
        f"mean={mean:.3f} p75={scores_sorted[3 * n // 4]:.3f} max={scores_sorted[-1]:.3f}"
    )


def print_report(correct: list[float], incorrect: list[float]) -> None:
    print("Grounding threshold calibration (live Pinecone index, 21 golden queries)")
    _stats("Correct-chunk scores  ", correct)
    _stats("Incorrect-chunk scores", incorrect)

    if correct and incorrect:
        # Simple midpoint heuristic between the worst correct match and the
        # best incorrect one — a threshold here separates them the way
        # decision-boundary logic normally would, but read the full
        # distributions above before trusting this blindly on a 21-query set.
        worst_correct = min(correct)
        best_incorrect = max(incorrect)
        suggested = (worst_correct + best_incorrect) / 2
        print(
            f"\nWorst correct match: {worst_correct:.3f} | Best incorrect match: {best_incorrect:.3f}"
        )
        print(f"Suggested grounding_threshold (midpoint): {suggested:.3f}")
        if worst_correct < best_incorrect:
            print(
                "Note: correct and incorrect score ranges OVERLAP — no single threshold "
                "perfectly separates them on this golden set. The suggested value minimizes "
                "the split but some misclassification is unavoidable at this scale (21 queries)."
            )


if __name__ == "__main__":
    correct_scores, incorrect_scores = run()
    print_report(correct_scores, incorrect_scores)
