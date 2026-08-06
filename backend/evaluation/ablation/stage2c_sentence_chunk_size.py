"""
Stage 2c — chunk_size x chunk_overlap sweep for "sentence" chunking,
fixed at parser=pdfplumber (the chosen final config: pdfplumber+sentence,
picked over the Stage 1+2 joint grid's technically-higher-scoring
pymupdf+hierarchical winner for simplicity — no auto-merge retrieval logic
needed).

Every earlier script using "sentence" (Stage 1, Stage 2, the joint grid)
called chunk_pages(..., chunk_size=500, chunk_overlap=50) — Settings.py's
inherited defaults, never independently validated for this strategy. Stage
2b swept size/overlap too, but only for "hierarchical" (a different
parameter shape: a list of per-tree-level sizes, not a single chunk_size).
This closes the equivalent gap for "sentence" specifically, the same way
Stage 6 swept alpha under its fixed retrieval mode: fix parser+strategy,
vary only chunk_size and chunk_overlap.

Run from backend/:  python -m evaluation.ablation.stage2c_sentence_chunk_size
"""
from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from api.core.config import get_settings
from evaluation.inmemory_search import dense_topk
from evaluation.llamaindex_eval import CallableRetriever, build_nodes, evaluate_retriever
from evaluation.retrieval_metrics import load_golden_dataset
from ingestion.embeddings import _get_embed_model
from ingestion.parsers.parser import chunk_pages, parse_all_pdfs

PARSER_BACKEND = "pdfplumber"  # chosen final config
CHUNK_STRATEGY = "sentence"  # chosen final config
TOP_K = 10

CHUNK_SIZE_GRID = (250, 500, 800)
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
    for size in CHUNK_SIZE_GRID:
        for overlap in CHUNK_OVERLAP_GRID:
            if overlap >= size:
                continue  # overlap can't exceed (or equal) the chunk it's overlapping
            chunks = chunk_pages(
                pages, strategy=CHUNK_STRATEGY, chunk_size=size, chunk_overlap=overlap,
            )

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
                    "config": f"size={size} overlap={overlap}",
                    "n_chunks": len(chunks),
                    **metrics,
                }
            )
    return rows


def print_table(rows: list[dict]) -> None:
    print("Stage 2c - sentence chunk_size x chunk_overlap sweep (parser=pdfplumber)")
    print(f"{'Config':22s} {'chunks':>7s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s}")
    for r in rows:
        print(
            f"{r['config']:22s} {r['n_chunks']:7d} "
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
