"""
Stage 2 ablation — chunk strategy: token / sentence / sentence_window /
hierarchical / semantic / markdown.

Fixes the parser backend to Stage 1's winner (pdfplumber) and dense
embedding to Settings.embedding_model, varies only chunk_strategy, scores
retrieval quality (in-memory) against golden_dataset.jsonl. See
../../EVALUATION_METHODOLOGY.md Part C, Stage 2, and parser.py's module
docstring for the caveats behind each strategy.

Note on "hierarchical": chunk_pages() returns the *entire* tree (root +
intermediate + leaf nodes). Only leaf nodes are embedded/retrieved here,
per parser.py's own guidance — the root/intermediate nodes exist for
auto-merging retrieval, not for direct nearest-neighbor search, and
including them would just be duplicate/overlapping text competing against
itself in the ranking.

Run from backend/:  python -m evaluation.ablation.stage2_chunking
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
from ingestion.parsers.parser import ChunkStrategy, chunk_pages, parse_all_pdfs

PARSER_BACKEND = "pdfplumber"  # Stage 1 winner
CHUNK_STRATEGIES: tuple[ChunkStrategy, ...] = (
    "token", "sentence", "sentence_window", "hierarchical", "semantic", "markdown",
)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 10


def run() -> list[dict]:
    settings = get_settings()
    golden = load_golden_dataset()
    embed_model = _get_embed_model(
        settings.embedding_model,
        settings.embedding_dimensions,
        settings.openai_api_key,
        settings.embedding_batch_size,
    )

    query_vectors = embed_model.get_text_embedding_batch(
        [q.question for q in golden], show_progress=False
    )
    query_vec_by_text = dict(zip((q.question for q in golden), query_vectors))

    pages = parse_all_pdfs(backend=PARSER_BACKEND)

    rows = []
    for strategy in CHUNK_STRATEGIES:
        try:
            chunks = chunk_pages(
                pages, strategy=strategy, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            )
        except RuntimeError as exc:
            print(f"{strategy:16s} skipped - {exc}")
            continue

        total_nodes = len(chunks)
        if strategy == "hierarchical":
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
                "config": strategy,
                "n_chunks": len(chunks),
                "n_nodes_total": total_nodes,
                **metrics,
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    print("Stage 2 - Chunking strategy comparison (parser=pdfplumber)")
    print(f"{'Config':16s} {'chunks':>7s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s}")
    for r in rows:
        print(
            f"{r['config']:16s} {r['n_chunks']:7d} "
            f"{r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@10']:6.3f} {r['mrr']:6.3f}"
        )
    winner = max(rows, key=lambda r: (r["recall@3"], r["recall@1"], r["mrr"]))
    print(
        f"\nWinner: {winner['config']} - highest Recall@3 ({winner['recall@3']:.3f}) among "
        f"{len(rows)} chunk strategies, scored on {winner['n_queries']} golden queries "
        f"({winner['n_chunks']} chunks)."
    )


if __name__ == "__main__":
    results = run()
    print_table(results)
