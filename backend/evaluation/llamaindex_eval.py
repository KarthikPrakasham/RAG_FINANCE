"""
LlamaIndex-native retrieval evaluation.

Replaces the hand-rolled recall_at_k/reciprocal_rank/ndcg_at_k formulas that
used to live in retrieval_metrics.py with llama_index.core.evaluation.retrieval's
own HitRate/MRR/NDCG metric classes and RetrieverEvaluator, wired through a
real llama_index BaseRetriever. Verified (by reading llama_index's source,
version 0.14.23) that the formulas were mathematically identical before
swapping — this is a library-adoption change, not a correctness fix.

Terminology note carried over from that verification: what this project
(and the grading rubric) calls "Recall@k" is llama_index's HitRate metric
(binary: was *any* relevant chunk in the top-k), not its separate `Recall`
class (`|retrieved ∩ expected| / |expected|`, which requires finding *all*
relevant chunks). The two coincide when a query has exactly one relevant
chunk, which is most — not all — of golden_dataset.jsonl (hierarchical
chunking can produce several leaf chunks over one labeled page). Keeping
the `recall@k` dict key name for continuity with every ablation script's
existing print_table(), but it is HitRate under the hood.

Two entry points, for two different situations:

- evaluate_retriever(): Stages 1, 1+2 joint grid, 2b, 3, 4, 5, 6 — cases
  where there's a real corpus to search. Wraps a `query_text -> ranked
  chunk_ids` function as a CallableRetriever (a real BaseRetriever) and
  scores it with RetrieverEvaluator (for MRR) + HitRate (for the @k
  breakdown, via slicing — RetrieverEvaluator itself scores whatever
  length list the retriever returns, so getting Recall@1/@3/@10 from one
  retrieval call means slicing after the fact, not three separate
  retrievers).

- score_precomputed(): Stage 7 (reranking) — there's no "retrieve by
  query" step, just a fixed candidate list already reordered by each
  reranker. Reranking is a llama_index BaseNodePostprocessor concept, not
  a BaseRetriever one, so forcing it through a fake retriever would be an
  artificial fit — this calls HitRate/MRR/NDCG directly on the
  precomputed lists instead.
"""
from __future__ import annotations

from typing import Callable, Iterable

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.evaluation.retrieval.evaluator import RetrieverEvaluator
from llama_index.core.evaluation.retrieval.metrics import HitRate, MRR, discounted_gain
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from evaluation.retrieval_metrics import GoldenQuery, relevant_chunk_ids

_hit_rate_metric = HitRate()


def _ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """NDCG@k with IDCG capped at k — the standard truncated definition.

    llama_index's own NDCG class computes IDCG over ALL len(expected_ids)
    positions, uncapped by k (its docstring assumes retrieved_ids is a
    retriever's full natural output, not manually sliced smaller than the
    relevant set). That assumption breaks here: hierarchical chunking gives
    many small leaf chunks per labeled page, so expected_ids often has
    9-20 members while we deliberately slice retrieved_ids to k=3 — feeding
    that straight into NDCG.compute() would normalize against an IDCG that
    assumes 9-20 ideal hits could fit in 3 slots, deflating every score by
    roughly half. Using discounted_gain (the library's actual formula
    primitive) directly, with IDCG capped at k, reproduces the standard
    NDCG@k definition instead."""
    expected_set = set(expected_ids)
    sliced = retrieved_ids[:k]
    dcg = sum(
        discounted_gain(rel=(rid in expected_set), i=i, mode="linear")
        for i, rid in enumerate(sliced, start=1)
    )
    ideal_len = min(len(expected_set), k)
    idcg = sum(discounted_gain(rel=True, i=i, mode="linear") for i in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0 else 0.0


class CallableRetriever(BaseRetriever):
    """Adapts any `query_text -> ranked list[chunk_id]` function (e.g. a
    closure around inmemory_search.dense_topk, or a live Chroma/FAISS/
    Pinecone query call) into a llama_index BaseRetriever, so it can be
    scored with RetrieverEvaluator like any other retriever."""

    def __init__(self, retrieve_fn: Callable[[str], list[str]], node_by_id: dict[str, TextNode]):
        self._retrieve_fn = retrieve_fn
        self._node_by_id = node_by_id
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        chunk_ids = self._retrieve_fn(query_bundle.query_str)
        return [
            NodeWithScore(node=self._node_by_id[cid], score=1.0 / (rank + 1))
            for rank, cid in enumerate(chunk_ids)
            if cid in self._node_by_id
        ]


def build_nodes(chunks: Iterable) -> dict[str, TextNode]:
    return {c.chunk_id: TextNode(id_=c.chunk_id, text=c.text) for c in chunks}


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def evaluate_retriever(
    retriever: BaseRetriever,
    golden: list[GoldenQuery],
    chunks: Iterable,
    k_values: tuple[int, ...] = (1, 3, 10),
) -> dict:
    """Score a real BaseRetriever against the golden dataset. One retrieval
    call per query (via RetrieverEvaluator, which also gives MRR); Recall@k
    for k < the retriever's own top_k is computed by slicing that same
    result and calling llama_index's HitRate directly — no extra retrieval
    calls needed."""
    chunks = list(chunks)
    ri_evaluator = RetrieverEvaluator.from_metric_names(["mrr"], retriever=retriever)

    hits: dict[int, list[float]] = {k: [] for k in k_values}
    rr: list[float] = []
    n_scored = 0

    for query in golden:
        expected = list(relevant_chunk_ids(query, chunks))
        if not expected:
            continue
        n_scored += 1
        result = ri_evaluator.evaluate(query=query.question, expected_ids=expected)
        retrieved_ids = result.retrieved_ids
        for k in k_values:
            sliced = retrieved_ids[:k]
            score = (
                _hit_rate_metric.compute(retrieved_ids=sliced, expected_ids=expected).score
                if sliced
                else 0.0
            )
            hits[k].append(score)
        rr.append(result.metric_vals_dict["mrr"])

    out = {f"recall@{k}": _mean(hits[k]) for k in k_values}
    out["mrr"] = _mean(rr)
    out["n_queries"] = n_scored
    return out


def score_precomputed(
    retrieved_ids_by_query: dict[str, list[str]],
    golden: list[GoldenQuery],
    chunks: Iterable,
    k_values: tuple[int, ...] = (1, 3),
    ndcg_at: int | None = None,
) -> dict:
    """Score already-computed ranked lists (e.g. Stage 7's post-rerank
    results) with llama_index's HitRate/MRR/NDCG classes directly — no
    BaseRetriever involved, since reranking isn't a retrieval step."""
    chunks = list(chunks)
    mrr_metric = MRR()

    hits: dict[int, list[float]] = {k: [] for k in k_values}
    rr: list[float] = []
    ndcg: list[float] = []
    n_scored = 0

    for query in golden:
        expected = list(relevant_chunk_ids(query, chunks))
        retrieved_ids = retrieved_ids_by_query.get(query.id, [])
        if not expected or not retrieved_ids:
            continue
        n_scored += 1
        for k in k_values:
            sliced = retrieved_ids[:k]
            score = (
                _hit_rate_metric.compute(retrieved_ids=sliced, expected_ids=expected).score
                if sliced
                else 0.0
            )
            hits[k].append(score)
        rr.append(mrr_metric.compute(retrieved_ids=retrieved_ids, expected_ids=expected).score)
        if ndcg_at is not None:
            ndcg.append(_ndcg_at_k(retrieved_ids, expected, ndcg_at))

    out = {f"recall@{k}": _mean(hits[k]) for k in k_values}
    out["mrr"] = _mean(rr)
    if ndcg_at is not None:
        out[f"ndcg@{ndcg_at}"] = _mean(ndcg)
    out["n_queries"] = n_scored
    return out
