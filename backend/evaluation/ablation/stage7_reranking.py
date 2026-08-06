"""
Stage 7 ablation — reranking: none vs cross-encoder vs LLM-as-reranker, on
top of Stage 6's winning retrieval config (alpha=1.0, i.e. dense-only —
see stage6_hybrid_merge.py; with pymupdf+hierarchical chunking, sparse
contributes nothing at the sweep optimum).

Retrieves a RERANK_CANDIDATE_K-deep candidate pool per query (so Recall@K
over that pool is fixed by construction across all three variants), then
each reranker reorders it down to FINAL_K. What moves is ranking quality —
MRR and NDCG@FINAL_K — not whether the right chunk is present at all. See
../../EVALUATION_METHODOLOGY.md Part C, Stage 7.

LLM-as-reranker uses OpenAI (Settings.openai_api_key) rather than
Settings.llm_model (Gemini) — the .env GEMINI_API_KEY/ANTHROPIC_API_KEY
values in this environment are placeholders, not live keys; only OpenAI is
configured for real calls here.

Run from backend/:  python -m evaluation.ablation.stage7_reranking
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from api.core.config import get_settings
from evaluation.inmemory_search import hybrid_alpha_topk
from evaluation.llamaindex_eval import score_precomputed
from evaluation.retrieval_metrics import load_golden_dataset
from ingestion.embeddings import _get_embed_model
from ingestion.parsers.parser import chunk_pages, parse_all_pdfs
from ingestion.sparse_index import encode_sparse, encode_sparse_queries, fit_and_save_bm25

PARSER_BACKEND = "pdfplumber"
CHUNK_STRATEGY = "sentence"
ALPHA = 1.0  # Stage 6 winner (dense-only, at the sweep optimum)
RERANK_CANDIDATE_K = 10
FINAL_K = 3


def _sparse_vec_to_dict(sv) -> dict[int, float]:
    return dict(zip(sv.indices, sv.values))


def _build_candidates():
    settings = get_settings()
    golden = load_golden_dataset()
    pages = parse_all_pdfs(backend=PARSER_BACKEND)
    chunks = chunk_pages(pages, strategy=CHUNK_STRATEGY)
    if CHUNK_STRATEGY == "hierarchical":
        chunks = [c for c in chunks if c.is_leaf]
    chunk_text_by_id = {c.chunk_id: c.text for c in chunks}
    chunk_texts = [c.text for c in chunks]

    embed_model = _get_embed_model(
        settings.embedding_model, settings.embedding_dimensions,
        settings.openai_api_key, settings.embedding_batch_size,
    )
    chunk_dense = dict(zip((c.chunk_id for c in chunks), embed_model.get_text_embedding_batch(chunk_texts, show_progress=False)))
    query_dense = dict(zip((q.id for q in golden), embed_model.get_text_embedding_batch(
        [q.question for q in golden], show_progress=False
    )))

    encoder = fit_and_save_bm25(chunk_texts, save_path=Path(__file__).resolve().parent / "_stage7_bm25.json")
    chunk_sparse = {
        c.chunk_id: _sparse_vec_to_dict(sv)
        for c, sv in zip(chunks, encode_sparse(chunk_texts, encoder))
    }
    query_sparse = {
        q.id: _sparse_vec_to_dict(sv)
        for q, sv in zip(golden, encode_sparse_queries([q.question for q in golden], encoder))
    }

    candidates_by_query = {
        q.id: hybrid_alpha_topk(
            query_dense[q.id], query_sparse[q.id], chunk_dense, chunk_sparse, ALPHA, RERANK_CANDIDATE_K
        )
        for q in golden
    }
    return golden, chunks, chunk_text_by_id, candidates_by_query


def _rerank_none(candidates: list[str], **_) -> list[str]:
    return candidates[:FINAL_K]


def _rerank_cross_encoder(candidates: list[str], query_text: str, chunk_text_by_id: dict, cross_encoder) -> list[str]:
    pairs = [(query_text, chunk_text_by_id[cid]) for cid in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = [cid for _, cid in sorted(zip(scores, candidates), reverse=True, key=lambda p: p[0])]
    return ranked[:FINAL_K]


def _rerank_llm(candidates: list[str], query_text: str, chunk_text_by_id: dict, openai_client) -> list[str]:
    numbered = "\n".join(
        f"[{i}] {chunk_text_by_id[cid][:400]}" for i, cid in enumerate(candidates)
    )
    prompt = (
        f"Question: {query_text}\n\nCandidate passages:\n{numbered}\n\n"
        f"Return ONLY a JSON array of the {FINAL_K} candidate numbers (integers, most relevant "
        "first) that best answer the question, e.g. [2, 0, 5]."
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        text = text.strip("`").removeprefix("json").strip()
        indices = json.loads(text)
        ranked = [candidates[i] for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
        for cid in candidates:
            if cid not in ranked:
                ranked.append(cid)
        return ranked[:FINAL_K]
    except Exception as exc:
        print(f"  (LLM rerank fallback to no-op for one query: {exc})")
        return candidates[:FINAL_K]


def _score(name: str, results_by_query: dict[str, list[str]], golden, chunks) -> dict:
    metrics = score_precomputed(results_by_query, golden, chunks, k_values=(1, 3), ndcg_at=3)
    return {"config": name, **metrics}


def run() -> list[dict]:
    golden, chunks, chunk_text_by_id, candidates_by_query = _build_candidates()
    settings = get_settings()

    rows = []

    none_results = {q.id: _rerank_none(candidates_by_query[q.id]) for q in golden}
    rows.append(_score("none", none_results, golden, chunks))

    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ce_results = {
        q.id: _rerank_cross_encoder(candidates_by_query[q.id], q.question, chunk_text_by_id, cross_encoder)
        for q in golden
    }
    rows.append(_score("cross-encoder", ce_results, golden, chunks))

    from openai import OpenAI

    openai_client = OpenAI(api_key=settings.openai_api_key)
    llm_results = {
        q.id: _rerank_llm(candidates_by_query[q.id], q.question, chunk_text_by_id, openai_client)
        for q in golden
    }
    rows.append(_score("llm-as-reranker", llm_results, golden, chunks))

    return rows


def print_table(rows: list[dict]) -> None:
    print(f"Stage 7 - Reranking comparison (candidates=hybrid alpha={ALPHA}, top-{RERANK_CANDIDATE_K} -> top-{FINAL_K})")
    print(f"{'Config':16s} {'R@1':>6s} {'R@3':>6s} {'MRR':>6s} {'NDCG@3':>7s}")
    for r in rows:
        print(f"{r['config']:16s} {r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['mrr']:6.3f} {r['ndcg@3']:7.3f}")
    winner = max(rows, key=lambda r: (r["ndcg@3"], r["mrr"]))
    print(
        f"\nWinner: {winner['config']} - highest NDCG@3 ({winner['ndcg@3']:.3f}) among "
        f"{len(rows)} reranking strategies, scored on {winner['n_queries']} golden queries. "
        "Recall@k over the fixed candidate pool doesn't isolate reranker quality as cleanly as "
        "MRR/NDCG@3 do (see module docstring) - weigh those two columns most heavily, and factor "
        "in added latency/cost per reranker before finalizing."
    )


if __name__ == "__main__":
    results = run()
    print_table(results)
