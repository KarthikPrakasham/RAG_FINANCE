"""
Stage 2b — hierarchical_chunk_sizes x chunk_overlap sweep, nested under the
Stage 1+2 joint grid winner (parser=pymupdf, strategy=hierarchical — see
stage1_2_joint_grid.py).

Every earlier script that used "hierarchical" (stage1_2_joint_grid.py,
stage3-7) called chunk_pages(pages, strategy="hierarchical") without
overriding hierarchical_chunk_sizes or chunk_overlap, so both silently sat
at their library defaults ([800, 400, 200] and 50) — never tested against
alternatives. This sweep closes that gap the same way Stage 6 swept alpha
under its fixed retrieval mode: fix parser+strategy, vary only these two
hyperparameters.

Scope note: this does NOT re-check whether a different parser would win
under a different size profile (that would mean a 3-parser x 12-config = 36
run grid). Stage 1's parser-level differences were small enough that this
felt like a reasonable place to stop nesting interaction checks — see
../../EVALUATION_METHODOLOGY.md Part C for the general rationale on when a
joint check is/isn't worth running.

Run from backend/:  python -m evaluation.ablation.stage2b_hierarchical_sizes
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from api.core.config import get_settings
from evaluation.inmemory_search import dense_topk
from evaluation.llamaindex_eval import CallableRetriever, build_nodes, evaluate_retriever
from evaluation.retrieval_metrics import load_golden_dataset
from ingestion.embeddings import _get_embed_model
from ingestion.parsers.parser import chunk_pages, parse_all_pdfs

PARSER_BACKEND = "pymupdf"  # Stage 1+2 joint grid winner
CHUNK_STRATEGY = "hierarchical"  # Stage 1+2 joint grid winner
TOP_K = 10

HIERARCHICAL_SIZES_GRID = (
    [1000, 500, 250],
    [800, 400, 200],  # current default
    [600, 300, 150],
    [400, 200, 100],
)
CHUNK_OVERLAP_GRID = (0, 50, 100)  # 50 is the current default


def run() -> list[dict]:
    settings = get_settings()
    golden = load_golden_dataset()
    embed_model = _get_embed_model(
        settings.embedding_model, settings.embedding_dimensions,
        settings.openai_api_key, settings.embedding_batch_size,
    )
    query_vectors = embed_model.get_text_embedding_batch(
        [q.question for q in golden], show_progress=False
    )
    query_vec_by_text = dict(zip((q.question for q in golden), query_vectors))

    pages = parse_all_pdfs(backend=PARSER_BACKEND)

    rows = []
    for sizes in HIERARCHICAL_SIZES_GRID:
        for overlap in CHUNK_OVERLAP_GRID:
            chunks = chunk_pages(
                pages,
                strategy=CHUNK_STRATEGY,
                chunk_overlap=overlap,
                hierarchical_chunk_sizes=sizes,
            )
            chunks = [c for c in chunks if c.is_leaf]

            chunk_vectors_list = embed_model.get_text_embedding_batch(
                [c.text for c in chunks], show_progress=False
            )
            chunk_vectors = {c.chunk_id: v for c, v in zip(chunks, chunk_vectors_list)}

            def _retrieve_fn(query_text: str, _chunk_vectors=chunk_vectors) -> list[str]:
                return dense_topk(query_vec_by_text[query_text], _chunk_vectors, TOP_K)

            retriever = CallableRetriever(retrieve_fn=_retrieve_fn, node_by_id=build_nodes(chunks))
            metrics = evaluate_retriever(retriever, golden, chunks, k_values=(1, 3, 10))
            rows.append(
                {
                    "config": f"sizes={sizes} overlap={overlap}",
                    "n_chunks": len(chunks),
                    **metrics,
                }
            )
    return rows


def print_table(rows: list[dict]) -> None:
    print("Stage 2b - hierarchical_chunk_sizes x chunk_overlap sweep (parser=pymupdf)")
    print(f"{'Config':38s} {'chunks':>7s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s}")
    for r in rows:
        print(
            f"{r['config']:38s} {r['n_chunks']:7d} "
            f"{r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@10']:6.3f} {r['mrr']:6.3f}"
        )
    winner = max(rows, key=lambda r: (r["recall@3"], r["recall@1"], r["mrr"]))
    print(
        f"\nWinner: {winner['config']} - highest Recall@3 ({winner['recall@3']:.3f}) among "
        f"{len(rows)} size/overlap combinations, scored on {winner['n_queries']} golden queries."
    )


if __name__ == "__main__":
    results = run()
    print_table(results)
