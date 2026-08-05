"""
Stage 4 ablation — vector DB engine: Pinecone vs Chroma vs FAISS.

Fixes parser=pymupdf, chunk=hierarchical (Stage 1+2 joint grid winner —
see stage1_2_joint_grid.py), embedding=Settings.embedding_model (Stage 3) —
the SAME ~40 leaf chunk vectors are loaded into all three engines so
Recall@k is a parity check, not a quality comparison; latency and ops/cost
are the deciding columns for this stage (see
../../EVALUATION_METHODOLOGY.md Part C, Stage 4).

Pinecone write scope: this writes to the LIVE production index
(Settings.pinecone_index_name), but ONLY into the namespace
PINECONE_ABLATION_NAMESPACE below — never the namespace the production
chatbot reads from — and deletes that namespace again at the end of the
run. Chroma and FAISS are local-only (see chroma_upsert.py / faiss_upsert.py
docstrings). Run only with the user's confirmation that live Pinecone writes
are OK.

Run from backend/:  python -m evaluation.ablation.stage4_vectordb
"""
from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from api.core.config import get_settings
from evaluation.llamaindex_eval import CallableRetriever, build_nodes, evaluate_retriever
from evaluation.retrieval_metrics import load_golden_dataset
from ingestion import chroma_upsert, faiss_upsert
from ingestion.embeddings import EmbeddedChunk, _get_embed_model
from ingestion.parsers.parser import chunk_pages, parse_all_pdfs
from ingestion.pinecone_upsert import get_index, verify_index_dimensions

PARSER_BACKEND = "pdfplumber"  # Stage 1+2 joint grid winner
CHUNK_STRATEGY = "sentence"  # Stage 1+2 joint grid winner
TOP_K = 10
PINECONE_ABLATION_NAMESPACE = "ablation-stage4"

OPS_NOTES = {
    "pinecone": "Managed/hosted, native sparse+dense hybrid support, per-request cost, network latency.",
    "chroma": "Local, free, dense-only (no native hybrid), manual scaling/ops.",
    "faiss": "Local, free, fastest raw ANN, no metadata/filtering built in, manual scaling/ops.",
}


def _build_embedded_chunks() -> list[EmbeddedChunk]:
    settings = get_settings()
    pages = parse_all_pdfs(backend=PARSER_BACKEND)
    chunks = chunk_pages(pages, strategy=CHUNK_STRATEGY)
    if CHUNK_STRATEGY == "hierarchical":
        chunks = [c for c in chunks if c.is_leaf]
    embed_model = _get_embed_model(
        settings.embedding_model, settings.embedding_dimensions,
        settings.openai_api_key, settings.embedding_batch_size,
    )
    vectors = embed_model.get_text_embedding_batch([c.text for c in chunks], show_progress=False)
    return [EmbeddedChunk(chunk=c, values=v) for c, v in zip(chunks, vectors)]


def _time_retriever(retriever: CallableRetriever, golden) -> float:
    """Mean per-query latency (ms) of retriever.retrieve(query_text), timed
    as a separate pass from evaluate_retriever's own scoring pass — a
    second, deliberate round of retrieval calls per engine, since the two
    concerns (latency vs. quality) are cheap to keep as independent, simple
    loops rather than one combined timing+scoring function."""
    latencies = []
    for q in golden:
        t0 = time.perf_counter()
        retriever.retrieve(q.question)
        latencies.append((time.perf_counter() - t0) * 1000)
    return sum(latencies) / len(latencies) if latencies else 0.0


def run() -> list[dict]:
    settings = get_settings()
    golden = load_golden_dataset()
    embedded_chunks = _build_embedded_chunks()
    chunks = [ec.chunk for ec in embedded_chunks]

    embed_model = _get_embed_model(
        settings.embedding_model, settings.embedding_dimensions,
        settings.openai_api_key, settings.embedding_batch_size,
    )
    query_vectors = embed_model.get_text_embedding_batch(
        [q.question for q in golden], show_progress=False
    )
    query_vec_by_text = dict(zip((q.question for q in golden), query_vectors))
    node_by_id = build_nodes(chunks)

    rows = []

    # --- Chroma (local, in-memory) ---
    collection = chroma_upsert.get_collection(
        collection_name="stage4-ablation", persist_directory=None
    )
    chroma_upsert.upsert_embedded_chunks(embedded_chunks, collection)

    def _chroma_retrieve_fn(query_text: str) -> list[str]:
        return chroma_upsert.query_dense(collection, query_vec_by_text[query_text], TOP_K)

    chroma_retriever = CallableRetriever(retrieve_fn=_chroma_retrieve_fn, node_by_id=node_by_id)
    latency_ms = _time_retriever(chroma_retriever, golden)
    metrics = evaluate_retriever(chroma_retriever, golden, chunks, k_values=(1, 3, 10))
    rows.append({"config": "chroma", "latency_ms": round(latency_ms, 2), **metrics})

    # --- FAISS (local, in-memory) ---
    store = faiss_upsert.build_index(dim=settings.embedding_dimensions)
    faiss_upsert.upsert_embedded_chunks(embedded_chunks, store)

    def _faiss_retrieve_fn(query_text: str) -> list[str]:
        return faiss_upsert.query_dense(store, query_vec_by_text[query_text], TOP_K)

    faiss_retriever = CallableRetriever(retrieve_fn=_faiss_retrieve_fn, node_by_id=node_by_id)
    latency_ms = _time_retriever(faiss_retriever, golden)
    metrics = evaluate_retriever(faiss_retriever, golden, chunks, k_values=(1, 3, 10))
    rows.append({"config": "faiss", "latency_ms": round(latency_ms, 2), **metrics})

    # --- Pinecone (live, scratch namespace only) ---
    pc, index = get_index(settings)
    verify_index_dimensions(pc, settings)
    ablation_settings = settings.model_copy(update={"pinecone_namespace": PINECONE_ABLATION_NAMESPACE})
    try:
        from ingestion.pinecone_upsert import upsert_embedded_chunks as pinecone_upsert_fn

        pinecone_upsert_fn(embedded_chunks, ablation_settings)
        time.sleep(2)  # Pinecone upserts are eventually-consistent within a namespace

        def _pinecone_retrieve_fn(query_text: str) -> list[str]:
            resp = index.query(
                vector=query_vec_by_text[query_text], top_k=TOP_K, namespace=PINECONE_ABLATION_NAMESPACE,
                include_values=False, include_metadata=False,
            )
            return [m.id for m in resp.matches]

        pinecone_retriever = CallableRetriever(retrieve_fn=_pinecone_retrieve_fn, node_by_id=node_by_id)
        latency_ms = _time_retriever(pinecone_retriever, golden)
        metrics = evaluate_retriever(pinecone_retriever, golden, chunks, k_values=(1, 3, 10))
        rows.append({"config": "pinecone", "latency_ms": round(latency_ms, 2), **metrics})
    finally:
        index.delete(delete_all=True, namespace=PINECONE_ABLATION_NAMESPACE)
        print(f"(cleaned up Pinecone namespace '{PINECONE_ABLATION_NAMESPACE}')")

    return rows


def print_table(rows: list[dict]) -> None:
    print("Stage 4 - Vector DB comparison (identical vectors: pymupdf/hierarchical/text-embedding-3-large)")
    print(f"{'Config':10s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s} {'latency(ms)':>12s}  notes")
    for r in rows:
        print(
            f"{r['config']:10s} {r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@10']:6.3f} "
            f"{r['mrr']:6.3f} {r['latency_ms']:12.2f}  {OPS_NOTES.get(r['config'], '')}"
        )
    print(
        "\nRecall@k/MRR should be near-identical across engines by construction (same vectors). "
        "Winner should be chosen on latency + ops fit (managed hosting, hybrid support), not "
        "quality — see EVALUATION_METHODOLOGY.md Part C, Stage 4."
    )


if __name__ == "__main__":
    results = run()
    print_table(results)
