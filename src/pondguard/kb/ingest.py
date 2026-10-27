"""Knowledge-base ingestion: PDF pages -> cleaned chunks with metadata -> Chroma.

Agentic concept: agentic RAG needs a store whose every passage can be cited. Each
chunk keeps its document id, title, page and a disease tag, so a diagnosis can cite
"CIBA-EHP-2022:p13" and a reviewer can open that page.

Pipeline: pypdf text per page -> clean (join hyphenated line breaks, drop repeated
headers and footers, drop table-of-contents-like text) -> chunk each page to about
2,000 characters (about 500 tokens) with 320 characters of overlap -> embed ->
Chroma persistent collection `disease_kb` (plus the embedder's name).

Chunks never cross a page boundary, so every citation has an exact page.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from pypdf import PdfReader

from pondguard.kb.embeddings import Embedder
from pondguard.paths import DATA_DIR

CHUNK_CHARS = 2000
OVERLAP_CHARS = 320
FRONT_MATTER_PAGES = 10  # training manuals open with programmes and contents pages
MIN_CHUNK_CHARS = 300
MAX_DIGIT_RATIO = 0.10  # contents pages and tables of numbers are mostly digits
COLLECTION = "disease_kb"

# Keyword tagger for the `disease` metadata (spec §6.7). Most frequent match wins.
DISEASE_KEYWORDS: dict[str, list[str]] = {
    "wssv": ["white spot", "wssv", "wsd"],
    "ehp": ["ehp", "enterocytozoon", "hepatopenaei", "microsporidia"],
    "white_faeces": ["white feces", "white faeces", "white faecal", "white fecal", "wfs"],
    "ahpnd": ["ahpnd", "acute hepatopancreatic", "early mortality syndrome", "ems"],
    "vibriosis": ["vibrio", "luminous", "luminescen"],
    "water_quality_stress": [
        "dissolved oxygen",
        "ammonia",
        "nitrite",
        "water quality",
        "salinity",
        "alkalinity",
    ],
}


@dataclass(frozen=True)
class SourceDoc:
    id: str
    title: str
    path: Path


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    title: str
    page: int
    disease: str


def read_sources(csv_path: Path, raw_dir: Path, only: list[str] | None = None) -> list[SourceDoc]:
    """Downloaded documents from data/kb/sources.csv (optionally just the ids in `only`)."""
    docs = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if only is not None and row["id"] not in only:
                continue
            if only is None and row["status"] != "downloaded":
                continue
            pdf = raw_dir / f"{row['id']}.pdf"
            if not pdf.exists():
                raise FileNotFoundError(f"{pdf} is missing; download it (see sources.csv)")
            docs.append(SourceDoc(row["id"], row["title"], pdf))
    if only is not None and {d.id for d in docs} != set(only):
        missing = set(only) - {d.id for d in docs}
        raise ValueError(f"not in sources.csv: {sorted(missing)}")
    return docs


# --- cleaning -----------------------------------------------------------------------


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def find_boilerplate(pages: list[str], min_share: float = 0.25) -> set[str]:
    """Lines (digits masked) that repeat on many pages: running headers and footers."""
    counts: Counter[str] = Counter()
    for text in pages:
        keys = {_line_key(ln) for ln in text.splitlines() if 3 < len(ln.strip()) < 140}
        counts.update(keys)
    threshold = max(3, int(len(pages) * min_share))
    return {k for k, n in counts.items() if n >= threshold}


def clean_page(text: str, boilerplate: set[str]) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    lines = [
        ln
        for ln in text.splitlines()
        if _line_key(ln) not in boilerplate and not ln.strip().isdigit()
    ]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def looks_like_contents(text: str) -> bool:
    digits = sum(c.isdigit() for c in text)
    return len(text) > 0 and digits / len(text) > MAX_DIGIT_RATIO or text.count("....") >= 5


# --- chunking and tagging ----------------------------------------------------------------


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Cut at word boundaries into pieces of about `size` characters, sharing `overlap`."""
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > 0 else end
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def tag_disease(text: str) -> str:
    lowered = text.lower()
    scores = {
        disease: sum(len(re.findall(rf"\b{re.escape(k)}", lowered)) for k in keywords)
        for disease, keywords in DISEASE_KEYWORDS.items()
    }
    best, score = max(scores.items(), key=lambda kv: kv[1])
    return best if score > 0 else "none"


def chunk_pages(
    doc_id: str, title: str, pages: list[str], skip_front: int = FRONT_MATTER_PAGES
) -> list[Chunk]:
    boilerplate = find_boilerplate(pages)
    chunks: list[Chunk] = []
    for number, raw in enumerate(pages, start=1):
        if number <= skip_front:
            continue
        text = clean_page(raw, boilerplate)
        if len(text) < MIN_CHUNK_CHARS or looks_like_contents(text):
            continue
        for i, piece in enumerate(split_text(text)):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    id=f"{doc_id}:p{number}:c{i}",
                    text=piece,
                    source=doc_id,
                    title=title,
                    page=number,
                    disease=tag_disease(piece),
                )
            )
    return chunks


def read_pdf_pages(path: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


# --- store ---------------------------------------------------------------------------------


def collection_name(embedder: Embedder) -> str:
    return f"{COLLECTION}_{embedder.name}"


def open_store(store_dir: Path) -> chromadb.ClientAPI:
    store_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(store_dir))


def ingest(
    docs: list[SourceDoc], embedder: Embedder, store_dir: Path | None = None, batch: int = 64
) -> dict[str, int]:
    """Rebuild the disease_kb collection from `docs`. Returns chunk counts per document."""
    client = open_store(store_dir or DATA_DIR / "chroma")
    name = collection_name(embedder)
    try:
        client.delete_collection(name)
    except Exception:  # first run: nothing to delete
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    counts: dict[str, int] = {}
    for doc in docs:
        chunks = chunk_pages(doc.id, doc.title, read_pdf_pages(doc.path))
        counts[doc.id] = len(chunks)
        for start in range(0, len(chunks), batch):
            group = chunks[start : start + batch]
            collection.add(
                ids=[c.id for c in group],
                documents=[c.text for c in group],
                embeddings=embedder.embed_documents([c.text for c in group]),
                metadatas=[
                    {"source": c.source, "title": c.title, "page": c.page, "disease": c.disease}
                    for c in group
                ],
            )
    return counts
