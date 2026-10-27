"""The Diagnostician's corrective loop, with the mock LLM (no model, offline)."""

import pytest
from pydantic import ValidationError

from pondguard import llm as llm_module
from pondguard.agents.diagnostician import (
    build_query,
    diagnose,
    follow_up_question,
    ranking_schema,
)
from pondguard.config import load_settings
from pondguard.kb.embeddings import HashEmbedder
from pondguard.kb.ingest import collection_name, open_store
from pondguard.kb.retriever import Retriever
from pondguard.llm import PassageGrade, PassageGrades, TokenBudget, get_llm
from pondguard.models import Anomaly

SETTINGS = load_settings(env={})
DOCS = {
    "D:p1:c0": ("White faecal strands floating on the pond surface indicate EHP infection.", "ehp"),
    "D:p2:c0": (
        "Low dissolved oxygen before dawn stresses shrimp and cuts feeding.",
        "water_quality_stress",
    ),
    "D:p3:c0": ("Vibriosis shows red discoloration and luminescence at night.", "vibriosis"),
}
DO_TREND = Anomaly(
    parameter="do_mg_l", kind="trend", severity="medium", value=3.2, window="3 dawn minima"
)


@pytest.fixture
def retriever(tmp_path):
    embedder = HashEmbedder()
    client = open_store(tmp_path / "chroma")
    collection = client.create_collection(
        collection_name(embedder), metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=list(DOCS),
        documents=[t for t, _ in DOCS.values()],
        embeddings=embedder.embed_documents([t for t, _ in DOCS.values()]),
        metadatas=[
            {"source": "D", "title": "Doc", "page": int(i[3]), "disease": d}
            for i, (_, d) in DOCS.items()
        ],
    )
    return Retriever(embedder, tmp_path / "chroma")


@pytest.fixture
def llm():
    return get_llm("agent", SETTINGS, TokenBudget(cap=100_000))


def test_query_puts_signs_and_conditions_before_water_quality():
    q = build_query([DO_TREND], ["white_faecal_strings", "reduced_feeding"], 42)
    assert q.startswith("signs: white faecal strings, reduced feeding")
    assert "white feces syndrome" in q and "EHP Enterocytozoon" in q
    assert q.index("possible conditions") < q.index("water quality: dissolved oxygen falling")
    assert q.endswith("day 42 of culture")


def test_query_is_general_not_scenario_specific():
    wssv = build_query([], ["white_spots", "red_discoloration"], 55)
    assert "white spot syndrome virus" in wssv and "water quality" not in wssv
    only_vague = build_query([], ["lethargy"], 50)  # non-specific sign: no condition guessed
    assert "possible conditions" not in only_vague and "lethargy" in only_vague
    assert build_query([DO_TREND], [], 30).startswith("water quality: dissolved oxygen")


def test_weak_evidence_requeries_twice_then_abstains(llm, retriever):
    result = diagnose(
        anomalies=[DO_TREND],
        symptoms=["lethargy"],
        day_of_culture=50,
        llm=llm,
        settings=SETTINGS,
        retriever=retriever,
    )  # mock grades 0.5 < 0.6
    assert [r["decision"] for r in result.rounds] == ["re-query", "re-query", "abstain"]
    assert result.diagnosis.abstained and result.diagnosis.retrieval_rounds == 3
    assert result.diagnosis.follow_up_question == "Do you see white spots on the shell?"
    assert result.rounds[1]["query"].endswith("(rewritten)")  # rewritten query was used
    assert not result.diagnosis.hypotheses


def test_good_evidence_ranks_with_citations_from_the_relevant_set(llm, retriever, monkeypatch):
    real = llm_module.mock_response

    def strong_grades(task, context):
        if task == "grade":
            grades = [
                PassageGrade(passage_id=p, grade=0.9, reason="r") for p in context["passage_ids"]
            ]
            return PassageGrades(grades=grades)
        return real(task, context)

    monkeypatch.setattr(llm_module, "mock_response", strong_grades)
    result = diagnose(
        anomalies=[DO_TREND],
        symptoms=["white_faecal_strings"],
        day_of_culture=42,
        llm=llm,
        settings=SETTINGS,
        retriever=retriever,
    )
    assert len(result.rounds) == 1 and result.rounds[0]["decision"] == "rank"
    d = result.diagnosis
    assert not d.abstained and d.retrieval_rounds == 1
    relevant_ids = {p.id for p in result.relevant}
    assert all(set(h.citations) <= relevant_ids for h in d.hypotheses)


def test_requery_limit_comes_from_settings(llm, retriever):
    settings = SETTINGS.model_copy(
        update={"loop_bounds": SETTINGS.loop_bounds.model_copy(update={"max_requeries": 0})}
    )
    result = diagnose(
        anomalies=[],
        symptoms=["lethargy"],
        day_of_culture=50,
        llm=llm,
        settings=settings,
        retriever=retriever,
    )
    assert len(result.rounds) == 1 and result.diagnosis.abstained


def test_follow_up_question_skips_signs_already_reported():
    assert (
        follow_up_question(["white_spots"])
        == "Have many shrimp died suddenly in the last day or two?"
    )
    every = [
        "white_spots",
        "sudden_mortality",
        "white_faecal_strings",
        "red_discoloration",
        "luminescence",
        "empty_gut",
        "slow_growth",
    ]
    assert "other changes" in follow_up_question(every)


def test_ranking_schema_only_allows_the_given_passage_ids():
    schema = ranking_schema(["A:p1:c0"])
    ok = {"hypotheses": [{"condition": "ehp", "confidence": 0.7, "citations": ["A:p1:c0"]}]}
    assert schema.model_validate(ok)
    bad = {"hypotheses": [{"condition": "ehp", "confidence": 0.7, "citations": ["B:p9:c0"]}]}
    with pytest.raises(ValidationError):
        schema.model_validate(bad)
