"""
Stage 3 ablation — embedding model: OpenAI text-embedding-3-large (current
production default, truncated to Settings.embedding_dimensions) vs a local
sentence-transformers model (all-MiniLM-L6-v2, general-purpose, free).

Fixes parser=pymupdf, chunk_strategy=hierarchical (Stage 1+2 joint grid
winner — see stage1_2_joint_grid.py, which found a real parser/chunking
interaction that the sequential pdfplumber+sentence pick missed), varies
only the embedding model. Scores retrieval quality (in-memory) plus Tier 3
operational columns: wall-clock embedding latency and, for the OpenAI
model, approximate $ cost from token count. See
../../EVALUATION_METHODOLOGY.md Part C, Stage 3.

hierarchical returns the full parent/child tree; only leaf chunks are
embedded/scored here, matching how the joint grid scored it.

Cost note: OPENAI_PRICE_PER_1K_TOKENS below is the published
text-embedding-3-large rate at the time this script was written — re-check
https://openai.com/pricing before quoting it in the final report, since
OpenAI revises pricing periodically. sentence-transformers models run
locally, so their "cost" is compute time, not a per-token API charge.

Run from backend/:  python -m evaluation.ablation.stage3_embedding
"""
from __future__ import annotations

import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from api.core.config import get_settings
from evaluation.inmemory_search import dense_topk
from evaluation.llamaindex_eval import CallableRetriever, build_nodes, evaluate_retriever
from evaluation.retrieval_metrics import load_golden_dataset
from ingestion.parsers.parser import chunk_pages, parse_all_pdfs

PARSER_BACKEND = "pdfplumber"  # Stage 1+2 joint grid winner
CHUNK_STRATEGY = "sentence"  # Stage 1+2 joint grid winner
TOP_K = 10

EMBEDDING_MODELS = ("text-embedding-3-large", "all-MiniLM-L6-v2", "text-embedding-3-small")
_OPENAI_MODEL_NAMES = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
OPENAI_PRICE_PER_1K_TOKENS = 0.00013  # $0.13 / 1M tokens — verify before final report

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _load_embedder(model_name: str):
    if model_name in _OPENAI_MODEL_NAMES:
        from ingestion.embeddings import _get_embed_model as _openai_embed_model

        settings = get_settings()
        return _openai_embed_model(
            model_name, settings.embedding_dimensions, settings.openai_api_key, settings.embedding_batch_size,
        )
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return HuggingFaceEmbedding(model_name=repo_id)


def run() -> list[dict]:
    golden = load_golden_dataset()
    pages = parse_all_pdfs(backend=PARSER_BACKEND)
    chunks = chunk_pages(pages, strategy=CHUNK_STRATEGY)
    if CHUNK_STRATEGY == "hierarchical":
        chunks = [c for c in chunks if c.is_leaf]
    chunk_texts = [c.text for c in chunks]
    total_tokens = sum(len(_ENCODING.encode(t)) for t in chunk_texts)

    rows = []
    for model_name in EMBEDDING_MODELS:
        embed_model = _load_embedder(model_name)

        t0 = time.perf_counter()
        chunk_vectors_list = embed_model.get_text_embedding_batch(chunk_texts, show_progress=False)
        embed_seconds = time.perf_counter() - t0
        chunk_vectors = {c.chunk_id: v for c, v in zip(chunks, chunk_vectors_list)}

        query_vectors = embed_model.get_text_embedding_batch(
            [q.question for q in golden], show_progress=False
        )
        query_vec_by_text = dict(zip((q.question for q in golden), query_vectors))

        def _retrieve_fn(query_text: str, _chunk_vectors=chunk_vectors) -> list[str]:
            return dense_topk(query_vec_by_text[query_text], _chunk_vectors, TOP_K)

        retriever = CallableRetriever(retrieve_fn=_retrieve_fn, node_by_id=build_nodes(chunks))
        metrics = evaluate_retriever(retriever, golden, chunks, k_values=(1, 3, 10))

        cost_usd = (
            (total_tokens / 1000) * OPENAI_PRICE_PER_1K_TOKENS
            if model_name in _OPENAI_MODEL_NAMES
            else 0.0
        )
        rows.append(
            {
                "config": model_name,
                "dims": len(chunk_vectors_list[0]) if chunk_vectors_list else 0,
                "embed_seconds": round(embed_seconds, 2),
                "cost_usd": round(cost_usd, 6),
                "is_local": model_name not in _OPENAI_MODEL_NAMES,
                **metrics,
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    print("Stage 3 - Embedding model comparison (parser=pymupdf, chunk=hierarchical)")
    print(f"{'Config':24s} {'dims':>5s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s} {'sec':>6s} {'$ (approx)':>11s}")
    for r in rows:
        cost_str = f"${r['cost_usd']:.6f}" if not r["is_local"] else "local/free"
        print(
            f"{r['config']:24s} {r['dims']:5d} "
            f"{r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@10']:6.3f} {r['mrr']:6.3f} "
            f"{r['embed_seconds']:6.2f} {cost_str:>11s}"
        )
    winner = max(rows, key=lambda r: (r["recall@3"], r["recall@1"], r["mrr"]))
    print(
        f"\nWinner: {winner['config']} - highest Recall@3 ({winner['recall@3']:.3f}) among "
        f"{len(rows)} embedding models, scored on {winner['n_queries']} golden queries. "
        "Weigh against cost/latency columns above before finalizing - a close second on "
        "quality that is free and local may still be the better production choice."
    )


if __name__ == "__main__":
    results = run()
    print_table(results)
