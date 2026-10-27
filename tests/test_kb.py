"""Knowledge base: cleaning, chunking, tagging, and retrieval with cited metadata.

Tests use the offline HashEmbedder, so they never download the real embedding model.
"""

import pytest

from pondguard.kb.embeddings import HashEmbedder
from pondguard.kb.ingest import (
    FRONT_MATTER_PAGES,
    SourceDoc,
    chunk_pages,
    clean_page,
    find_boilerplate,
    ingest,
    looks_like_contents,
    split_text,
    tag_disease,
)
from pondguard.kb.retriever import Retriever
from pondguard.paths import DATA_DIR

HEADER = "Training manual on shrimp disease management, 12 Dec 2022"


def _page(body: str, n: int) -> str:
    return f"{HEADER} {n}\n{body}\n{n}"


EHP_BODY = (
    "Enterocytozoon hepatopenaei (EHP) infects the hepatopancreas of Penaeus vannamei. "
    "White faecal strands floating on the surface of the pond are a common sign of EHP "
    "infection, along with slow growth and wide size variation among shrimp. "
) * 3
VIBRIO_BODY = (
    "Vibriosis is caused by Vibrio bacteria. Affected shrimp show red discoloration, "
    "lethargy and luminescence at night, especially in ponds with high organic load. "
) * 3
WATER_BODY = (
    "Dissolved oxygen falls before dawn in dense ponds. Low dissolved oxygen, rising "
    "ammonia and nitrite stress shrimp and reduce feeding in the early morning hours. "
) * 3


def _pages() -> list[str]:
    bodies = [EHP_BODY, VIBRIO_BODY, WATER_BODY] * 4
    return [_page(b, i + 1) for i, b in enumerate(bodies)]


def test_boilerplate_headers_and_page_numbers_are_removed():
    pages = _pages()
    bp = find_boilerplate(pages)
    cleaned = clean_page(pages[0], bp)
    assert "Training manual" not in cleaned
    assert not cleaned.endswith(" 1")
    assert cleaned.startswith("Enterocytozoon")


def test_hyphenated_line_breaks_are_joined():
    assert clean_page("hepato-\npancreas disease", set()) == "hepatopancreas disease"


def test_contents_like_text_is_detected():
    toc = "Role of carriers 25 - 29 6 Diagnosis 30 - 41 7 Management 42 - 50 8 " * 6
    assert looks_like_contents(toc)
    assert not looks_like_contents(EHP_BODY)


def test_split_text_overlaps_and_respects_size():
    text = ("word " * 1200).strip()
    pieces = split_text(text, size=2000, overlap=320)
    assert len(pieces) >= 3
    assert all(len(p) <= 2000 for p in pieces)
    assert pieces[0][-100:].split()[-1] in pieces[1]  # the overlap shares words


@pytest.mark.parametrize(
    "text,expected",
    [
        (EHP_BODY, "ehp"),
        (VIBRIO_BODY, "vibriosis"),
        (WATER_BODY, "water_quality_stress"),
        ("Nothing about shrimp health appears in this sentence at all.", "none"),
    ],
)
def test_disease_tagger(text, expected):
    assert tag_disease(text) == expected


def test_chunks_carry_page_and_never_include_front_matter():
    chunks = chunk_pages("DOC", "A Manual", _pages(), skip_front=FRONT_MATTER_PAGES)
    assert chunks and min(c.page for c in chunks) > FRONT_MATTER_PAGES
    assert all(c.id.startswith("DOC:p") and c.source == "DOC" for c in chunks)
    assert {c.disease for c in chunks} <= {"ehp", "vibriosis", "water_quality_stress"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    pages = _pages()
    monkeypatch.setattr("pondguard.kb.ingest.read_pdf_pages", lambda path: pages)
    monkeypatch.setattr("pondguard.kb.ingest.FRONT_MATTER_PAGES", 0)
    monkeypatch.setattr(
        "pondguard.kb.ingest.chunk_pages",
        lambda d, t, p, skip_front=0: chunk_pages(d, t, p, skip_front=0),
    )
    doc = SourceDoc("DOC", "A Manual", tmp_path / "doc.pdf")
    embedder = HashEmbedder()
    counts = ingest([doc], embedder, tmp_path / "chroma")
    return Retriever(embedder, tmp_path / "chroma"), counts


def test_retrieval_returns_cited_passages(store):
    retriever, counts = store
    assert counts["DOC"] == retriever.count() > 0
    passages = retriever.search("white faecal strands floating on the pond surface", k=3)
    assert passages[0].disease == "ehp"
    top = passages[0]
    assert top.source == "DOC" and top.title == "A Manual" and top.page >= 1
    assert top.id.startswith("DOC:p") and 0 < top.score <= 1
    assert passages == sorted(passages, key=lambda p: -p.score)


def test_disease_filter_restricts_results(store):
    retriever, _ = store
    passages = retriever.search("shrimp pond problem", k=5, disease="vibriosis")
    assert passages and all(p.disease == "vibriosis" for p in passages)


def test_missing_store_gives_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="pondguard ingest"):
        Retriever(HashEmbedder(), tmp_path / "empty")


@pytest.mark.skipif(
    not (DATA_DIR / "kb" / "raw" / "CIBA-EHP-2022.pdf").exists(),
    reason="KB PDFs are gitignored; only present on machines that downloaded them",
)
def test_real_pdf_chunks_have_pages_and_skip_contents():
    from pondguard.kb.ingest import read_pdf_pages

    pages = read_pdf_pages(DATA_DIR / "kb" / "raw" / "CIBA-EHP-2022.pdf")
    chunks = chunk_pages("CIBA-EHP-2022", "EHP manual", pages)
    assert len(chunks) > 50
    assert all(c.page > FRONT_MATTER_PAGES for c in chunks)
    assert any(c.disease == "ehp" for c in chunks)
