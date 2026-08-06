"""
Stage 1 ablation — parser backend: pypdf vs pdfplumber vs pymupdf.

Fixes chunking to sentence/500/50 and dense embedding to
Settings.embedding_model, varies only the PDF parser backend, and scores
retrieval quality against golden_dataset.jsonl via a real llama_index
BaseRetriever + RetrieverEvaluator (see ../llamaindex_eval.py). Ranking
itself is still brute-force in-memory cosine similarity (../inmemory_search.py)
— only the scoring layer is llama_index-native. See
../../EVALUATION_METHODOLOGY.md Part C, Stage 1.

Run from backend/:  python -m evaluation.ablation.stage1_parsing
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
from ingestion.parsers.parser import ParserBackend, chunk_pages, clean_text_pct, parse_all_pdfs

PARSER_BACKENDS: tuple[ParserBackend, ...] = ("pypdf", "pdfplumber", "pymupdf")
CHUNK_STRATEGY = "sentence"
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

    # Query vectors don't depend on parser backend — embed once, reuse below.
    query_vectors = embed_model.get_text_embedding_batch(
        [q.question for q in golden], show_progress=False
    )
    query_vec_by_text = dict(zip((q.question for q in golden), query_vectors))

    rows = []
    for backend in PARSER_BACKENDS:
        pages = parse_all_pdfs(backend=backend)
        pct_clean = clean_text_pct(pages)
        chunks = chunk_pages(
            pages,
            strategy=CHUNK_STRATEGY,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
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
                "config": backend,
                "clean_text_pct": pct_clean,
                "n_chunks": len(chunks),
                **metrics,
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    print("Stage 1 - Parsing backend comparison")
    print(f"{'Config':12s} {'clean%':>7s} {'chunks':>7s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s}")
    for r in rows:
        print(
            f"{r['config']:12s} {r['clean_text_pct']*100:6.1f}% {r['n_chunks']:7d} "
            f"{r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@10']:6.3f} {r['mrr']:6.3f}"
        )
    winner = max(rows, key=lambda r: (r["recall@3"], r["recall@1"], r["mrr"]))
    print(
        f"\nWinner: {winner['config']} - highest Recall@3 ({winner['recall@3']:.3f}) among "
        f"{len(rows)} parser backends, scored on {winner['n_queries']} golden queries, with "
        f"{winner['clean_text_pct']*100:.1f}% of pages yielding non-trivial extracted text."
    )


if __name__ == "__main__":
    results = run()
    print_table(results)
