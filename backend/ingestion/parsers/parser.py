"""
Stage 1 (parsing) + Stage 2 (chunking) of the ingestion pipeline for the four
Federal Reserve PDF chapters, plus a structured-row loader for the FRED CSV.

Stage 1 ablation — parser backend: pypdf / pdfplumber / pymupdf.
Stage 2 ablation — chunk strategy: LlamaIndex node parsers —
    token            TokenTextSplitter          fixed-size token window + overlap
    sentence         SentenceSplitter           sentence-aware, general purpose
    sentence_window  SentenceWindowNodeParser   one node per sentence + surrounding window
    hierarchical     HierarchicalNodeParser     multi-tier parent/child chunks
    semantic         SemanticSplitterNodeParser embedding-distance breakpoints
    markdown         MarkdownNodeParser         splits on markdown headers

Caveats worth knowing before picking a winner:
- "semantic" needs an embedding model *at chunk time*, so it couples Stage 2 to
  a Stage 3 embedding choice instead of keeping them independent ablations.
  Default is OpenAI's text-embedding-3-small (needs OPENAI_API_KEY in the
  environment); pass a sentence-transformers model name to use HuggingFace
  locally instead — see _OPENAI_EMBED_MODELS / _get_embed_model.
- "markdown" only does something useful if the source text has real markdown
  headers (#, ##, ...) — plain PDF-extracted text doesn't, so on these 4 PDFs
  it degrades to one node per page.
- "hierarchical" returns the *entire* tree (root + intermediate + leaf nodes),
  not just leaves — see level/parent_chunk_id/is_leaf on Chunk. Callers doing
  auto-merging retrieval should embed leaves and merge up via parent_chunk_id.

The FRED CSV is loaded separately via parse_fred_csv() and is never chunked or
embedded: trend questions need arithmetic over rows (see
structured_data/trend_tool.py), not nearest-neighbor text retrieval.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
import pypdf
import tiktoken
from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import NodeRelationship

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Filename -> law tag. Must match the Citation.law values in api/schemas/chat.py.
LAW_TAGS: dict[str, str] = {
    "FedReserve_ECOA_Regulation_B.pdf": "ECOA/RegB",
    "FedReserve_Fair_Housing_Act.pdf": "FairHousingAct",
    "FedReserve_Fair_Lending_Overview.pdf": "Overview",
    "FedReserve_Consumer_Compliance_Handbook_Intro.pdf": "HandbookIntro",
}

FRED_CSV_NAME = "FRED_Total_Consumer_Credit.csv"

ParserBackend = Literal["pypdf", "pdfplumber", "pymupdf"]
ChunkStrategy = Literal[
    "token", "sentence", "sentence_window", "hierarchical", "semantic", "markdown"
]

_ENCODING = tiktoken.get_encoding("cl100k_base")
_DEFAULT_HIERARCHICAL_SIZES = [800, 400, 200]


@dataclass
class ParsedPage:
    source_doc: str
    law: str
    page_num: int  # 1-indexed
    text: str
    parser: ParserBackend


@dataclass
class Chunk:
    chunk_id: str
    source_doc: str
    law: str
    section: str
    page_num: int
    chunk_index: int
    text: str
    content_hash: str
    parser: ParserBackend
    chunk_strategy: ChunkStrategy
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    level: int | None = None            # hierarchical: tree depth, 0 = root
    parent_chunk_id: str | None = None  # hierarchical: parent chunk's chunk_id
    is_leaf: bool = True                # hierarchical: False for root/intermediate nodes
    window: str | None = None           # sentence_window: surrounding-sentence context


def _law_tag(path: Path) -> str:
    law = LAW_TAGS.get(path.name)
    if law is None:
        raise ValueError(f"Unknown source document (no law tag mapped): {path.name}")
    return law


# ---------------------------------------------------------------------------
# Stage 1 — PDF parsing backends
# ---------------------------------------------------------------------------

def parse_pdf_pypdf(path: Path) -> list[ParsedPage]:
    law = _law_tag(path)
    reader = pypdf.PdfReader(str(path))
    return [
        ParsedPage(
            source_doc=path.name, law=law, page_num=i + 1,
            text=(page.extract_text() or "").strip(), parser="pypdf",
        )
        for i, page in enumerate(reader.pages)
    ]


def parse_pdf_pdfplumber(path: Path) -> list[ParsedPage]:
    law = _law_tag(path)
    pages: list[ParsedPage] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            pages.append(ParsedPage(
                source_doc=path.name, law=law, page_num=i + 1,
                text=(page.extract_text() or "").strip(), parser="pdfplumber",
            ))
    return pages


def parse_pdf_pymupdf(path: Path) -> list[ParsedPage]:
    law = _law_tag(path)
    pages: list[ParsedPage] = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc):
            pages.append(ParsedPage(
                source_doc=path.name, law=law, page_num=i + 1,
                text=page.get_text().strip(), parser="pymupdf",
            ))
    return pages


_PDF_BACKENDS = {
    "pypdf": parse_pdf_pypdf,
    "pdfplumber": parse_pdf_pdfplumber,
    "pymupdf": parse_pdf_pymupdf,
}


def parse_pdf(path: Path, backend: ParserBackend = "pypdf") -> list[ParsedPage]:
    try:
        parse_fn = _PDF_BACKENDS[backend]
    except KeyError:
        raise ValueError(f"Unknown parser backend: {backend!r}. Choose from {list(_PDF_BACKENDS)}")
    return parse_fn(path)


def parse_all_pdfs(data_dir: Path = DATA_DIR, backend: ParserBackend = "pypdf") -> list[ParsedPage]:
    """Parse the four known Federal Reserve chapters, in a fixed order."""
    pages: list[ParsedPage] = []
    for filename in LAW_TAGS:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected source PDF not found: {path}")
        pages.extend(parse_pdf(path, backend=backend))
    return pages


def clean_text_pct(pages: list[ParsedPage]) -> float:
    """Stage 1 metric: fraction of pages with non-trivial extracted text."""
    if not pages:
        return 0.0
    non_empty = sum(1 for p in pages if len(p.text) > 20)
    return round(non_empty / len(pages), 4)


# ---------------------------------------------------------------------------
# Stage 2 — chunking strategies (LlamaIndex node parsers)
# ---------------------------------------------------------------------------

# Model names that select the OpenAI embeddings API instead of a local
# HuggingFace/sentence-transformers model — matches the Stage 3 shortlist.
_OPENAI_EMBED_MODELS = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}


@lru_cache(maxsize=4)
def _get_embed_model(model_name: str):
    """Lazily load the embedding model SemanticSplitterNodeParser needs.

    Imports are local (not at module top) since both backends pull in heavy
    optional dependencies (torch/transformers, or the openai client) that only
    "semantic" chunking actually needs.
    """
    if model_name in _OPENAI_EMBED_MODELS:
        from llama_index.embeddings.openai import OpenAIEmbedding

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — required to use "
                f"{model_name!r} for semantic chunking."
            )
        return OpenAIEmbedding(model=model_name, api_key=api_key)

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return HuggingFaceEmbedding(model_name=repo_id)


def _build_node_parser(
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    window_size: int,
    hierarchical_chunk_sizes: list[int] | None,
    embed_model_name: str,
) -> NodeParser:
    if strategy == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=_ENCODING.encode,
        )
    if strategy == "sentence":
        return SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=_ENCODING.encode,
        )
    if strategy == "sentence_window":
        return SentenceWindowNodeParser.from_defaults(window_size=window_size)
    if strategy == "hierarchical":
        return HierarchicalNodeParser.from_defaults(
            chunk_sizes=hierarchical_chunk_sizes or _DEFAULT_HIERARCHICAL_SIZES,
            chunk_overlap=chunk_overlap,
        )
    if strategy == "semantic":
        return SemanticSplitterNodeParser(
            embed_model=_get_embed_model(embed_model_name),
            buffer_size=1,
            breakpoint_percentile_threshold=95,
        )
    if strategy == "markdown":
        return MarkdownNodeParser.from_defaults()
    raise ValueError(f"Unknown chunk strategy: {strategy!r}")


def chunk_pages(
    pages: list[ParsedPage],
    strategy: ChunkStrategy = "sentence",
    chunk_size: int = 250,
    chunk_overlap: int = 50,
    window_size: int = 3,
    hierarchical_chunk_sizes: list[int] | None = None,
    embed_model_name: str = "text-embedding-3-small",
) -> list[Chunk]:
    """Split parsed pages into retrieval chunks, one page at a time so every
    chunk stays traceable to a single (source_doc, page_num) for citations."""
    node_parser = _build_node_parser(
        strategy, chunk_size, chunk_overlap, window_size, hierarchical_chunk_sizes, embed_model_name,
    )
    has_scalar_size = strategy in ("token", "sentence")

    chunks: list[Chunk] = []
    for page in pages:
        if not page.text:
            continue

        document = Document(
            text=page.text,
            metadata={"source_doc": page.source_doc, "law": page.law, "page_num": page.page_num},
        )
        nodes = node_parser.get_nodes_from_documents([document])
        if not nodes:
            continue

        # First pass: assign content-hash chunk_ids so the second pass can
        # resolve each node's PARENT relationship to *our* chunk_id.
        chunk_id_by_node_id: dict[str, str] = {}
        for idx, node in enumerate(nodes):
            content_hash = hashlib.sha256(
                f"{page.source_doc}:{page.page_num}:{idx}:{node.text}".encode("utf-8")
            ).hexdigest()
            chunk_id_by_node_id[node.node_id] = content_hash[:16]

        node_by_id = {node.node_id: node for node in nodes}
        levels: dict[str, int] = {}

        def _level(node_id: str) -> int:
            if node_id in levels:
                return levels[node_id]
            parent = node_by_id[node_id].relationships.get(NodeRelationship.PARENT)
            result = 0 if parent is None else _level(parent.node_id) + 1
            levels[node_id] = result
            return result

        for idx, node in enumerate(nodes):
            parent_info = node.relationships.get(NodeRelationship.PARENT)
            content_hash = hashlib.sha256(
                f"{page.source_doc}:{page.page_num}:{idx}:{node.text}".encode("utf-8")
            ).hexdigest()

            chunks.append(Chunk(
                chunk_id=chunk_id_by_node_id[node.node_id],
                source_doc=page.source_doc,
                law=page.law,
                section=node.metadata.get("header_path", "") if strategy == "markdown" else "",
                page_num=page.page_num,
                chunk_index=idx,
                text=node.text,
                content_hash=content_hash,
                parser=page.parser,
                chunk_strategy=strategy,
                chunk_size=chunk_size if has_scalar_size else None,
                chunk_overlap=chunk_overlap if has_scalar_size else None,
                level=_level(node.node_id) if strategy == "hierarchical" else None,
                parent_chunk_id=(
                    chunk_id_by_node_id.get(parent_info.node_id) if parent_info else None
                ),
                is_leaf=(NodeRelationship.CHILD not in node.relationships),
                window=node.metadata.get("window") if strategy == "sentence_window" else None,
            ))
    return chunks


def parse_and_chunk(
    data_dir: Path = DATA_DIR,
    parser_backend: ParserBackend = "pypdf",
    chunk_strategy: ChunkStrategy = "sentence",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    window_size: int = 3,
    hierarchical_chunk_sizes: list[int] | None = None,
    embed_model_name: str = "text-embedding-3-small",
) -> list[Chunk]:
    """Full Stage 1 + Stage 2 pipeline: parse the 4 PDFs, then chunk them."""
    pages = parse_all_pdfs(data_dir, backend=parser_backend)
    return chunk_pages(
        pages,
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        window_size=window_size,
        hierarchical_chunk_sizes=hierarchical_chunk_sizes,
        embed_model_name=embed_model_name,
    )


# ---------------------------------------------------------------------------
# FRED CSV — structured rows only, deliberately NOT chunked
# ---------------------------------------------------------------------------

def parse_fred_csv(data_dir: Path = DATA_DIR, filename: str = FRED_CSV_NAME) -> pd.DataFrame:
    """
    Load the FRED consumer-credit time series as structured rows.

    This feeds structured_data/fred_loader.py (SQLite) and trend_tool.py
    (delta/trend arithmetic) — it is intentionally excluded from
    chunk_pages()/parse_and_chunk() and never gets a chunk_id or embedding.
    """
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected FRED CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["observation_date"])
    return df.rename(columns={"observation_date": "date", "TOTALSL": "total_consumer_credit"})


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    print("=== Stage 1 — parser backend comparison ===")
    for backend in ("pypdf", "pdfplumber", "pymupdf"):
        parsed_pages = parse_all_pdfs(backend=backend)
        print(f"{backend:12s} pages={len(parsed_pages):3d} clean_text_pct={clean_text_pct(parsed_pages)}")

    all_pages = parse_all_pdfs(backend="pypdf")

    print("\n=== Stage 2 — chunk strategy comparison (chunk_size=500, overlap=50) ===")
    for strategy in ("token", "sentence", "sentence_window", "hierarchical", "semantic", "markdown"):
        try:
            doc_chunks = chunk_pages(all_pages, strategy=strategy, chunk_size=500, chunk_overlap=50)
        except RuntimeError as exc:
            print(f"{strategy:16s} skipped — {exc}")
            continue
        leaf_only = sum(1 for c in doc_chunks if c.is_leaf)
        print(f"{strategy:16s} total={len(doc_chunks):3d}  leaf={leaf_only:3d}")

    print("\n=== FRED CSV (structured, not chunked) ===")
    fred_df = parse_fred_csv()
    print(f"rows={len(fred_df)} range={fred_df['date'].min().date()} -> {fred_df['date'].max().date()}")
