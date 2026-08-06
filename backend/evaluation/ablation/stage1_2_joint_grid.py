"""
Stage 1+2 joint grid — checks whether the sequential (OFAT) result from
stage1_parsing.py -> stage2_chunking.py (pick best parser holding chunking
fixed, then pick best chunk strategy holding that parser fixed) actually
matches the TRUE joint optimum over all (parser, chunk_strategy) pairs.

Why this exists: sequential/greedy ablation assumes parser choice and chunk
strategy don't interact — that the best parser is best regardless of chunk
strategy, and vice versa. That assumption could be wrong (different parsers
extract text with different whitespace/line-break artifacts, which could
shift where "sentence" boundaries fall). For most stage pairs in this
project, running the full grid is too expensive to be worth it, but parser x
chunk_strategy is only 3x6=18 configs on an 18-page corpus — cheap enough
to just check directly instead of assuming. See
../../EVALUATION_METHODOLOGY.md Part C for where this fits, and
stage2_chunking.py's docstring for the hierarchical-leaves-only caveat,
which applies here too.

Downstream stages (3-7) still proceed sequentially from whichever (parser,
chunk_strategy) pair wins here — re-gridding every later stage combination
would blow up combinatorially for little additional payoff, since those
dimensions (embedding model, vector DB, retrieval mode, reranking) are far
more orthogonal to text extraction than chunking is.

Run from backend/:  python -m evaluation.ablation.stage1_2_joint_grid
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
from ingestion.parsers.parser import ChunkStrategy, ParserBackend, chunk_pages, parse_all_pdfs

PARSER_BACKENDS: tuple[ParserBackend, ...] = ("pypdf", "pdfplumber", "pymupdf")
CHUNK_STRATEGIES: tuple[ChunkStrategy, ...] = (
    "token", "sentence", "sentence_window", "hierarchical", "semantic", "markdown",
)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 10

# The literal fixed values stage1_parsing.py and stage2_chunking.py used —
# reproducing their sequential logic from inside this grid's own data.
SEQUENTIAL_FIXED_STRATEGY = "sentence"


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

    pages_by_backend = {backend: parse_all_pdfs(backend=backend) for backend in PARSER_BACKENDS}

    rows = []
    for backend in PARSER_BACKENDS:
        pages = pages_by_backend[backend]
        for strategy in CHUNK_STRATEGIES:
            try:
                chunks = chunk_pages(
                    pages, strategy=strategy, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                )
            except RuntimeError as exc:
                print(f"{backend}/{strategy} skipped - {exc}")
                continue
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
            rows.append({"parser": backend, "strategy": strategy, "n_chunks": len(chunks), **metrics})
    return rows


def analyze(rows: list[dict]) -> None:
    print("Stage 1+2 joint grid - parser x chunk_strategy (chunk_size=400, overlap=50)")
    print(f"{'Parser':12s} {'Strategy':16s} {'chunks':>7s} {'R@1':>6s} {'R@3':>6s} {'R@10':>6s} {'MRR':>6s}")
    for r in rows:
        print(
            f"{r['parser']:12s} {r['strategy']:16s} {r['n_chunks']:7d} "
            f"{r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@10']:6.3f} {r['mrr']:6.3f}"
        )

    def rank_key(r: dict) -> tuple:
        return (r["recall@3"], r["recall@1"], r["mrr"])

    joint_winner = max(rows, key=rank_key)
    print(
        f"\nJoint optimum: parser={joint_winner['parser']}, strategy={joint_winner['strategy']} "
        f"(R@3={joint_winner['recall@3']:.3f}, R@1={joint_winner['recall@1']:.3f}, MRR={joint_winner['mrr']:.3f})"
    )

    # Reproduce the sequential (OFAT) result from this same grid's data.
    fixed_strategy_rows = [r for r in rows if r["strategy"] == SEQUENTIAL_FIXED_STRATEGY]
    sequential_parser = max(fixed_strategy_rows, key=rank_key)["parser"]
    same_parser_rows = [r for r in rows if r["parser"] == sequential_parser]
    sequential_strategy = max(same_parser_rows, key=rank_key)["strategy"]
    sequential_result = next(
        r for r in rows if r["parser"] == sequential_parser and r["strategy"] == sequential_strategy
    )
    print(
        f"Sequential (OFAT) result: parser={sequential_parser}, strategy={sequential_strategy} "
        f"(R@3={sequential_result['recall@3']:.3f}, R@1={sequential_result['recall@1']:.3f}, "
        f"MRR={sequential_result['mrr']:.3f})"
    )

    if (joint_winner["parser"], joint_winner["strategy"]) == (sequential_parser, sequential_strategy):
        print(
            "\nResult: sequential ablation MATCHES the joint optimum - no parser/chunking "
            "interaction detected on this corpus. Stages 3-7 proceeding from this pair is justified."
        )
    else:
        print(
            "\nResult: sequential ablation DIVERGES from the joint optimum - a real interaction "
            f"exists. Stages 3-7 should be re-run from parser={joint_winner['parser']}, "
            f"strategy={joint_winner['strategy']} instead of the sequential pick."
        )


if __name__ == "__main__":
    results = run()
    analyze(results)
