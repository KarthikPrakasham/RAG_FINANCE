"""
Stage 5 ablation — retrieval mode: dense-only vs sparse-only (BM25) vs
hybrid (Pinecone-native alpha-scaled fusion, default alpha=0.5).

Fixes parser=pymupdf, chunk=hierarchical (Stage 1+2 joint grid winner —
see stage1_2_joint_grid.py), embedding=text-embedding-3-large (Stage 3
winner). Hybrid uses Settings.hybrid_alpha as a starting point —
Stage 6 is where alpha itself gets tuned and the fusion *method* (this
alpha-scaled approach vs. RRF) gets compared. See
../../EVALUATION_METHODOLOGY.md Part C, Stage 5.

Run from backend/:  python -m evaluation.ablation.stage5_retrieval_mode
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from api.core.config import get_settings
from evaluation.inmemory_search import dense_topk, hybrid_alpha_topk, sparse_topk
from evaluation.llamaindex_eval import CallableRetriever, build_nodes, evaluate_retriever
from evaluation.retrieval_metrics import load_golden_dataset
from ingestion.embeddings import _get_embed_model
from ingestion.parsers.parser import chunk_pages, parse_all_pdfs
from ingestion.sparse_index import encode_sparse, encode_sparse_queries, fit_and_save_bm25

PARSER_BACKEND = "pdfplumber"
CHUNK_STRATEGY = "sentence"
TOP_K = 10


def _sparse_vec_to_dict(sv) -> dict[int, float]:
    return dict(zip(sv.indices, sv.values))


def run() -> list[dict]:
    settings = get_settings()
    golden = load_golden_dataset()

    pages = parse_all_pdfs(backend=PARSER_BACKEND)
    chunks = chunk_pages(pages, strategy=CHUNK_STRATEGY)
    if CHUNK_STRATEGY == "hierarchical":
        chunks = [c for c in chunks if c.is_leaf]
    chunk_texts = [c.text for c in chunks]

    # --- Dense ---
    embed_model = _get_embed_model(
        settings.embedding_model, settings.embedding_dimensions,
        settings.openai_api_key, settings.embedding_batch_size,
    )
    chunk_dense_list = embed_model.get_text_embedding_batch(chunk_texts, show_progress=False)
    chunk_dense = {c.chunk_id: v for c, v in zip(chunks, chunk_dense_list)}
    query_dense_list = embed_model.get_text_embedding_batch(
        [q.question for q in golden], show_progress=False
    )
    query_dense = dict(zip((q.question for q in golden), query_dense_list))

    # --- Sparse (BM25, fit on this run's full chunk corpus) ---
    encoder = fit_and_save_bm25(chunk_texts, save_path=Path(__file__).resolve().parent / "_stage5_bm25.json")
    chunk_sparse_vecs = encode_sparse(chunk_texts, encoder)
    chunk_sparse = {c.chunk_id: _sparse_vec_to_dict(sv) for c, sv in zip(chunks, chunk_sparse_vecs)}
    query_sparse_vecs = encode_sparse_queries([q.question for q in golden], encoder)
    query_sparse = {
        q.question: _sparse_vec_to_dict(sv) for q, sv in zip(golden, query_sparse_vecs)
    }

    node_by_id = build_nodes(chunks)
    rows = []

    def _dense_fn(query_text: str) -> list[str]:
        return dense_topk(query_dense[query_text], chunk_dense, TOP_K)

    dense_retriever = CallableRetriever(retrieve_fn=_dense_fn, node_by_id=node_by_id)
    rows.append({"config": "dense-only", **evaluate_retriever(dense_retriever, golden, chunks, (1, 3, 10))})

    def _sparse_fn(query_text: str) -> list[str]:
        return sparse_topk(query_sparse[query_text], chunk_sparse, TOP_K)

    sparse_retriever = CallableRetriever(retrieve_fn=_sparse_fn, node_by_id=node_by_id)
    rows.append({"config": "sparse-only (BM25)", **evaluate_retriever(sparse_retriever, golden, chunks, (1, 3, 10))})

    alpha = settings.hybrid_alpha

    def _hybrid_fn(query_text: str) -> list[str]:
        return hybrid_alpha_topk(
            query_dense[query_text], query_sparse[query_text], chunk_dense, chunk_sparse, alpha, TOP_K
        )

    hybrid_retriever = CallableRetriever(retrieve_fn=_hybrid_fn, node_by_id=node_by_id)
    rows.append(
        {
            "config": f"hybrid (alpha={alpha})",
            **evaluate_retriever(hybrid_retriever, golden, chunks, (1, 3, 10)),
        }
    )
    return rows


def print_table(rows: list[dict]) -> None:
    print("Stage 5 - Retrieval mode comparison (parser=pymupdf, chunk=hierarchical, dense=text-embedding-3-large)")
    print(f"{'Config':22s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s}")
    for r in rows:
        print(f"{r['config']:22s} {r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@10']:6.3f} {r['mrr']:6.3f}")
    winner = max(rows, key=lambda r: (r["recall@3"], r["recall@1"], r["mrr"]))
    print(
        f"\nWinner: {winner['config']} - highest Recall@3 ({winner['recall@3']:.3f}) among "
        f"{len(rows)} retrieval modes, scored on {winner['n_queries']} golden queries. "
        "Alpha itself is tuned in Stage 6, not here."
    )


if __name__ == "__main__":
    results = run()
    print_table(results)
