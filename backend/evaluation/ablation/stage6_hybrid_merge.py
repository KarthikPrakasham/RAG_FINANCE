"""
Stage 6 ablation — hybrid fusion method: Pinecone-native alpha-scaled single
query (swept over alpha in {0, .25, .5, .75, 1}) vs. Reciprocal Rank Fusion
(RRF) of two independently-ranked dense/sparse lists.

These are genuinely different mechanisms, not two names for the same thing:
alpha-scaling combines dense+sparse into ONE score per candidate from a
single pass (what Pinecone's server-side hybrid query does); RRF ranks
dense and sparse separately, then merges by 1/(rrf_k + rank). See
../../EVALUATION_METHODOLOGY.md Part C, Stage 6, and inmemory_search.py's
module docstring.

Fixes parser=pymupdf, chunk=hierarchical (Stage 1+2 joint grid winner —
see stage1_2_joint_grid.py), embedding=text-embedding-3-large (Stage 3
winner). Output: the alpha sweep curve (to justify
Settings.hybrid_alpha) plus a straight fusion-method winner.

Run from backend/:  python -m evaluation.ablation.stage6_hybrid_merge
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from api.core.config import get_settings
from evaluation.inmemory_search import dense_topk, hybrid_alpha_topk, rrf_merge, sparse_topk
from evaluation.llamaindex_eval import CallableRetriever, build_nodes, evaluate_retriever
from evaluation.retrieval_metrics import load_golden_dataset
from ingestion.embeddings import _get_embed_model
from ingestion.parsers.parser import chunk_pages, parse_all_pdfs
from ingestion.sparse_index import encode_sparse, encode_sparse_queries, fit_and_save_bm25

PARSER_BACKEND = "pdfplumber"
CHUNK_STRATEGY = "sentence"
TOP_K = 10
RRF_CANDIDATE_K = 20  # depth of each individual list fed into RRF, before cutting to TOP_K
ALPHA_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)


def _sparse_vec_to_dict(sv) -> dict[int, float]:
    return dict(zip(sv.indices, sv.values))


def _build_dense_and_sparse():
    settings = get_settings()
    golden = load_golden_dataset()
    pages = parse_all_pdfs(backend=PARSER_BACKEND)
    chunks = chunk_pages(pages, strategy=CHUNK_STRATEGY)
    if CHUNK_STRATEGY == "hierarchical":
        chunks = [c for c in chunks if c.is_leaf]
    chunk_texts = [c.text for c in chunks]

    embed_model = _get_embed_model(
        settings.embedding_model, settings.embedding_dimensions,
        settings.openai_api_key, settings.embedding_batch_size,
    )
    chunk_dense = dict(zip((c.chunk_id for c in chunks), embed_model.get_text_embedding_batch(chunk_texts, show_progress=False)))
    query_dense = dict(zip((q.question for q in golden), embed_model.get_text_embedding_batch(
        [q.question for q in golden], show_progress=False
    )))

    encoder = fit_and_save_bm25(chunk_texts, save_path=Path(__file__).resolve().parent / "_stage6_bm25.json")
    chunk_sparse = {
        c.chunk_id: _sparse_vec_to_dict(sv)
        for c, sv in zip(chunks, encode_sparse(chunk_texts, encoder))
    }
    query_sparse = {
        q.question: _sparse_vec_to_dict(sv)
        for q, sv in zip(golden, encode_sparse_queries([q.question for q in golden], encoder))
    }
    return golden, chunks, chunk_dense, query_dense, chunk_sparse, query_sparse


def run() -> tuple[list[dict], list[dict]]:
    golden, chunks, chunk_dense, query_dense, chunk_sparse, query_sparse = _build_dense_and_sparse()
    node_by_id = build_nodes(chunks)

    # --- Alpha sweep (Pinecone-native fusion) ---
    alpha_rows = []
    for alpha in ALPHA_SWEEP:
        def _hybrid_fn(query_text: str, _alpha=alpha) -> list[str]:
            return hybrid_alpha_topk(
                query_dense[query_text], query_sparse[query_text], chunk_dense, chunk_sparse, _alpha, TOP_K
            )

        retriever = CallableRetriever(retrieve_fn=_hybrid_fn, node_by_id=node_by_id)
        metrics = evaluate_retriever(retriever, golden, chunks, (1, 3, 10))
        alpha_rows.append({"config": f"alpha={alpha}", **metrics})

    # --- RRF ---
    def _rrf_fn(query_text: str) -> list[str]:
        dense_ranked = dense_topk(query_dense[query_text], chunk_dense, RRF_CANDIDATE_K)
        sparse_ranked = sparse_topk(query_sparse[query_text], chunk_sparse, RRF_CANDIDATE_K)
        return rrf_merge(dense_ranked, sparse_ranked, TOP_K)

    rrf_retriever = CallableRetriever(retrieve_fn=_rrf_fn, node_by_id=node_by_id)
    rrf_row = {"config": "RRF", **evaluate_retriever(rrf_retriever, golden, chunks, (1, 3, 10))}

    return alpha_rows, [rrf_row]


def print_tables(alpha_rows: list[dict], rrf_rows: list[dict]) -> None:
    print("Stage 6a - Alpha sweep (Pinecone-native fusion), parser=pymupdf, chunk=hierarchical")
    print(f"{'Config':14s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s}")
    for r in alpha_rows:
        print(f"{r['config']:14s} {r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@10']:6.3f} {r['mrr']:6.3f}")
    best_alpha = max(alpha_rows, key=lambda r: (r["recall@3"], r["recall@1"], r["mrr"]))

    print("\nStage 6b - Fusion method: best alpha-scaled config vs RRF")
    print(f"{'Config':14s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s}")
    all_rows = [best_alpha] + rrf_rows
    for r in all_rows:
        print(f"{r['config']:14s} {r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@10']:6.3f} {r['mrr']:6.3f}")
    winner = max(all_rows, key=lambda r: (r["recall@3"], r["recall@1"], r["mrr"]))
    print(
        f"\nWinner: {winner['config']} - highest Recall@3 ({winner['recall@3']:.3f}) among "
        f"the best alpha-scaled config and RRF, scored on {winner['n_queries']} golden queries. "
        f"Recommended Settings.hybrid_alpha: {best_alpha['config'].split('=')[1]} "
        f"(best of the sweep in 6a)."
    )


if __name__ == "__main__":
    alpha_results, rrf_results = run()
    print_tables(alpha_results, rrf_results)
