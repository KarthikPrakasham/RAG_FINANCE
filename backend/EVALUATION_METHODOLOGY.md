# Evaluation Methodology — Consumer Credit Rights Assistant

Referenced by `data/REQUIREMENT.md` §7. This document defines *how* every
pipeline decision (parsing, chunking, embedding, vector DB, retrieval mode,
fusion method, reranking, LLM) is measured and justified, so that
`evaluation/evaluation_report.md` is filled with real numbers, not opinions.

Current scope: **Phase 0 (foundation) + Phase 1 (ingestion ablations,
Stages 1–4) + Phase 2 (retrieval ablations, Stages 5–7)**. Stage 8 (LLM
choice) and end-to-end RAGAS/DeepEval scoring are documented here for
completeness but implemented in a later phase.

## A. The golden dataset

**File**: `evaluation/golden_dataset.jsonl`, one JSON object per line:

```json
{
  "id": "q01",
  "question": "Can a lender consider my age when deciding on a loan?",
  "category": "ECOA",
  "expected_source_doc": "FedReserve_ECOA_Regulation_B.pdf",
  "relevant_pages": [2, 3],
  "reference_answer": "Short ground-truth answer used by RAGAS/DeepEval as the reference, and by humans to sanity-check retrieval.",
  "retrieval_eval": true
}
```

`relevant_pages` (not chunk_id) is the stored ground truth — page numbers are stable
across every parser backend and chunk strategy, whereas `chunk_id` is a content hash
that changes whenever chunk boundaries change. Each ablation script resolves
`relevant_pages` → the actual `chunk_id`s of that run's chunks (a chunk counts as
relevant if `(source_doc, page_num)` matches a labeled page) before computing
Recall@k/MRR/NDCG@k. `retrieval_eval: false` marks rows that don't fit page-based
PDF retrieval — the FRED CSV questions (routed to `trend_tool.py`, never chunked)
and the one guardrail/behavior acceptance question — so ablation scripts can filter
to `retrieval_eval: true` rows while `acceptance_tests.py` still uses the full file.

- **Size**: 20–25 questions, covering all 4 PDFs (at least 4–5 questions
  each) plus a couple of CSV/trend questions, plus the 5 acceptance-test
  questions from `REQUIREMENT.md` §6 verbatim (so acceptance testing and
  ablation scoring share the same ground truth).
- **`relevant_chunk_ids` is chunking-strategy-dependent** — chunk IDs are
  content hashes (see `parser.py`'s `chunk_id`), so they change whenever
  chunk boundaries change. To keep one golden file usable across every
  Stage 1/2 ablation, label relevance by **page number** first
  (`source_doc` + `page_num`, which is stable across all parsers/strategies),
  then resolve to chunk IDs per-run: a chunk counts as relevant if it
  overlaps a labeled relevant page. This is what `retrieval_metrics.py`'s
  helpers assume.
- **Construction**: manual, by a human who has read the 18 pages — the
  corpus is small enough that this is both feasible and far more
  trustworthy for a legal-compliance use case than LLM-synthesized
  questions, which tend to paraphrase chunk text back at itself rather
  than ask the way a real consumer would.

## B. Metrics — what to use, and when

### Tier 1 — Retrieval metrics (need: query → ground-truth relevant pages/chunks)

Answer "did we fetch the right passage?", independent of what any LLM says
afterward. Used for **every** ablation in Stages 1–7, since all of them
change what gets retrieved (or how it's ranked).

| Metric | Formula | Interpretation |
|---|---|---|
| **Recall@k** | `(1/|Q|) Σ 1[∃ relevant chunk in top-k]` | Binary hit/miss per query, averaged. Report @1, @3, @10 per the rubric. |
| **MRR** | `(1/|Q|) Σ 1/rank_of_first_relevant_hit` (0 if absent from the retrieved set) | Rewards ranking the right chunk *higher*, not just present. Decisive when two configs tie on Recall@k. |
| **NDCG@k** | `DCG@k / IDCG@k`, `DCG@k = Σ_{i=1}^{k} rel_i / log2(i+1)` | Needs *graded* relevance (0/1/2, not just yes/no). Use only if a question has one clearly-best chunk and other partially-relevant ones; otherwise MRR already captures the ranking signal with binary labels — don't force graded labels just to populate this column. |

**Implementation**: computed via `llama_index.core.evaluation.retrieval`
(`HitRate`, `MRR`, `discounted_gain`), through a real `BaseRetriever` +
`RetrieverEvaluator` (see `evaluation/llamaindex_eval.py`), not hand-rolled
formulas — verified mathematically identical to the original hand-rolled
version before switching. Two terminology/formula notes worth carrying into
the final report:
- What this project calls "Recall@k" is llama_index's `HitRate` metric
  (any relevant chunk in top-k), not its separate `Recall` class (`|∩| / |expected|`,
  requiring *all* relevant chunks) — they coincide only when a query has
  exactly one relevant chunk, which isn't guaranteed once hierarchical
  chunking is in play (see `golden_dataset.jsonl`'s multi-chunk-per-page
  queries).
- llama_index's `NDCG` class computes IDCG over *all* `len(expected_ids)`
  positions, uncapped — correct when scoring a retriever's full natural
  output, but wrong once `expected_ids` exceeds k (common with hierarchical
  chunking's many small leaf chunks per page). `score_precomputed()`
  instead calls `discounted_gain` directly with IDCG capped at k, the
  standard NDCG@k definition.

### Tier 2 — Generation/RAG metrics (need: query → generated answer → retrieved context)

Answer "given what we retrieved, did the LLM produce a good, grounded
answer?" Expensive (LLM-as-judge calls) — used only for **Stage 8** and the
final end-to-end score, not per-ablation in Stages 1–7.

| Metric | Catches |
|---|---|
| Faithfulness (RAGAS + DeepEval) | Answer states something not supported by retrieved context — critical given `REQUIREMENT.md`'s "never invent a legal protection" guardrail. |
| Context Precision | Retrieved chunks that were irrelevant noise, even if the right one was also retrieved. |
| Answer Relevancy | Answer doesn't actually address the question asked. |
| Hallucination (DeepEval) | Overlaps with Faithfulness but different judge — report both, they can disagree. |

### Tier 3 — Operational (every stage, secondary column)

Latency (p50 per query/per batch), cost ($ per 1K chunks embedded, $ per
query), and note on scalability (qualitative for an 18-page corpus — call
out that this is a *design* justification, not something a 4-PDF corpus can
demonstrate empirically). **Decisive for Stage 4** (vector DB), where
Recall@k is expected to be near-identical across DBs given identical
vectors.

## C. Stage-by-stage plan

Each stage's script: fix everything upstream to the winner of the previous
stage, vary only this stage's dimension, score every variant against the
full golden dataset with Tier 1 metrics (+ Tier 3 where noted), print an
ablation table, and state a winner with a one-paragraph justification.

| Stage | File | Fixed | Varies | Metrics |
|---|---|---|---|---|
| 1 — Parsing | `ablation/stage1_parsing.py` | chunking=sentence/500/50, dense embed | `pypdf` / `pdfplumber` / `pymupdf` | Recall@1/3/10, MRR, + `clean_text_pct` |
| 2 — Chunking | `ablation/stage2_chunking.py` | winning parser | `token`/`sentence`/`sentence_window`/`hierarchical`/`semantic`/`markdown` | Recall@1/3/10, MRR, chunk count |
| 1+2 — Joint check | `ablation/stage1_2_joint_grid.py` | chunk_size=500/overlap=50 for size-based strategies | all 3 parsers × all 6 strategies (18 configs) | Recall@1/3/10, MRR — verifies the sequential Stage 1→2 pick actually matches the joint optimum (see note below) |
| 2b — Hierarchical size sweep | `ablation/stage2b_hierarchical_sizes.py` | parser+strategy fixed to the 1+2 joint winner | `hierarchical_chunk_sizes` × `chunk_overlap` (12 configs) | Recall@1/3/10, MRR — only run when the joint winner is `hierarchical`, since its size parameter isn't the scalar `chunk_size` used by `token`/`sentence` |
| 2c — Sentence size sweep | `ablation/stage2c_sentence_chunk_size.py` | parser+strategy fixed to `pdfplumber`+`sentence` (chosen over the joint-optimal `pymupdf`+`hierarchical` for simplicity — see note below) | `chunk_size` × `chunk_overlap` (9 configs) | Recall@1/3/10, MRR — validates the scalar size/overlap actually used by the shipped `sentence` config, rather than leaving it at Settings.py's inherited 500/50 default |
| 3 — Embedding | `ablation/stage3_embedding.py` | winning parser+chunking | `text-embedding-3-large` vs a `sentence-transformers` model | Recall@1/3/10, MRR, $/1K chunks, latency |
| 4 — Vector DB | `ablation/stage4_vectordb.py` | winning embedding (same vectors) | Pinecone / Chroma / FAISS | Recall@k parity check, latency, ops cost/features |
| 5 — Retrieval mode | `ablation/stage5_retrieval_mode.py` | winning Stages 1–4 | dense-only / sparse-only (BM25) / hybrid | Recall@1/3/10, MRR |
| 6 — Fusion method | `ablation/stage6_hybrid_merge.py` | winning hybrid config | Pinecone-native alpha-scaled single query vs. manual two-query RRF; alpha swept over {0, .25, .5, .75, 1} | Recall@1/3/10, MRR per alpha — pick final `hybrid_alpha` |
| 7 — Reranking | `ablation/stage7_reranking.py` | winning Stage 6 config, top-20 candidate pool | none / cross-encoder / LLM-as-reranker, cut to top-3 | MRR, NDCG@3 movement (Recall@20 pool is constant by construction), added latency |
| 8 — LLM *(later phase)* | `ablation/stage8_llm.py` | winning Stages 1–7 | GPT / Gemini / Claude | RAGAS + DeepEval (Faithfulness, Answer Relevancy, Context Precision) |

**Why Stage 1+2 gets a joint grid and no other stage pair does**: sequential
(OFAT) ablation assumes adjacent stages don't interact — that the best
parser is best regardless of chunk strategy. That assumption held for every
other stage boundary tested (Stage 4's three DB engines are checked on
literally identical vectors by construction, so there's no interaction to
miss; Stages 5-7 operate on already-fixed chunks, so they don't feed back
into parsing/chunking choices). But parser × chunk_strategy is only 3×6=18
configs on an 18-page corpus — cheap enough to check directly rather than
assume, and on this corpus the check mattered: the sequential pick
diverged from the true joint optimum. `stage1_parsing.py` and
`stage2_chunking.py` are kept as-is (they're still useful as the marginal,
one-dimension-at-a-time view).

**Final decision on Stage 1+2 — score vs. simplicity.** The joint grid's
actual optimum is `pymupdf` + `hierarchical` (Recall@3 = 1.000). `hierarchical`
chunking returns a full parent/child tree; using it properly in production
retrieval means building leaf-to-parent auto-merging via `parent_chunk_id`
in the retrieval layer — real, unbuilt work, not just a config flag. The
shipped config instead uses the sequential pick, `pdfplumber` + `sentence`
(Recall@3 = 0.952, no auto-merge machinery needed), a deliberate
simplicity-over-score trade-off — document it in the final report as
exactly that, not as "the ablation's top result," since it measurably isn't.
Stages 3 onward all build on `pdfplumber` + `sentence`. Stage 2c
(`chunk_size` × `chunk_overlap` sweep, scoped to this pair specifically)
then found the inherited default of 500/50 wasn't optimal either — 250/50
scores higher (Recall@3 = 1.000) — so the shipped chunking config is
`pdfplumber` + `sentence` + `chunk_size=250` + `chunk_overlap=50`.

## D. Ablation table format (used identically in every stage script's output and in `evaluation_report.md`)

```
Stage N — <name>
Config              Recall@1  Recall@3  Recall@10   MRR   <tier-3 cols>
<variant A>            ...       ...        ...      ...
<variant B>            ...       ...        ...      ...   ← winner
Winner: <variant B> — <one paragraph: why, tied to this corpus's
characteristics (18 pages, dense legal terminology, 4 documents that must
not be conflated for citation purposes)>.
```

## E. Execution order (hard dependencies)

`golden_dataset.jsonl` → `retrieval_metrics.py` → Stage 1 & Stage 2 (individually)
→ Stage 1+2 joint grid (authoritative winner) → Stage 3 → (Stage 4 in
parallel, doesn't gate Stage 5) → Stage 5 → Stage 6 → Stage 7. Each stage
script imports the previous stage's winning config as a constant rather
than re-discovering it, so the final pipeline config is always the
composition of every table's winner.
