"""Tools: telemetry maths and anomaly rules, ladder lookup, guardrails, KB search and grading."""

import json

import pandas as pd
import pytest

from pondguard.config import load_settings
from pondguard.kb.embeddings import HashEmbedder
from pondguard.kb.ingest import collection_name, open_store
from pondguard.kb.retriever import Retriever
from pondguard.llm import TokenBudget, get_llm
from pondguard.models import Passage, Reading
from pondguard.paths import CONFIG_DIR
from pondguard.simulator.scenarios import simulate
from pondguard.simulator.telemetry import apply_events, baseline
from pondguard.tools import telemetry_tools
from pondguard.tools.action_tools import lookup_action_ladder
from pondguard.tools.guardrail_tools import check_guardrails, find_dosage, step_citations
from pondguard.tools.kb_tools import grade_passages, search_disease_kb
from pondguard.tools.telemetry_tools import (
    check_thresholds,
    compute_trends,
    detect_anomalies,
    get_telemetry,
)
from pondguard.tracing import RunTracer

SETTINGS = load_settings(env={})
END = pd.Timestamp("2026-10-20 07:00")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    simulate(CONFIG_DIR / "scenario_specs.yaml", root / "scenarios", root / "telemetry")
    return root


@pytest.fixture(autouse=True)
def _point_tools_at_test_data(data_dir, monkeypatch):
    monkeypatch.setattr(telemetry_tools, "DATA_DIR", data_dir)


def _readings(df: pd.DataFrame) -> list[Reading]:
    return [Reading(**row) for row in df.to_dict("records")]


# --- telemetry -------------------------------------------------------------------------


def test_get_telemetry_returns_the_last_72_hours_in_order():
    readings = get_telemetry("P-07", hours=72)
    assert len(readings) == 72 and all(r.pond_id == "P-07" for r in readings)
    stamps = [r.ts for r in readings]
    assert stamps == sorted(stamps) and stamps[-1] == END
    assert len(get_telemetry("P-07", hours=24)) == 24
    with pytest.raises(FileNotFoundError):
        get_telemetry("P-99")


def test_compute_trends_shows_falling_dawn_oxygen():
    trends = compute_trends(get_telemetry("P-07"))
    dawn = [m["do_mg_l"] for m in trends["dawn_min_do"]]
    assert len(dawn) == 3 and dawn[0] > dawn[1] > dawn[2]
    assert trends["do_mg_l"]["min"] < 4.0
    assert {"mean", "min", "max", "slope_per_day", "change_24h"} <= set(trends["temp_c"])
    healthy = compute_trends(get_telemetry("P-01"))
    assert healthy["do_mg_l"]["min"] > 4.0 and abs(healthy["temp_c"]["change_24h"]) < 1.5


def test_detect_anomalies_finds_the_dawn_oxygen_trend():
    found = detect_anomalies(get_telemetry("P-07"))
    trend = next(a for a in found if a.parameter == "do_mg_l" and a.kind == "trend")
    assert trend.value < 4.0 and "dawn minima" in trend.window


@pytest.mark.parametrize("seed", range(1, 11))
def test_healthy_ponds_raise_no_anomalies(seed):
    readings = _readings(baseline("P-1", END, 35, seed).tail(72))
    assert detect_anomalies(readings) == []
    assert check_thresholds(readings) == []


def test_sudden_temperature_drop_is_flagged():
    df = baseline("P-1", END, 35, 3)
    df.loc[df.index[-24:], "temp_c"] -= 4.0  # drops 4 C within the last 24 h
    found = detect_anomalies(_readings(df.tail(72)))
    drop = next(a for a in found if a.parameter == "temp_c" and a.kind == "drop")
    assert drop.severity == "high" and drop.limit == 3.0


def test_steady_ammonia_rise_is_flagged():
    df = baseline("P-1", END, 35, 3).tail(72).copy()
    df["tan_mg_l"] += pd.Series(range(72), index=df.index) * 0.01  # +0.24/day
    found = detect_anomalies(_readings(df))
    assert any(a.parameter == "tan_mg_l" and a.kind == "trend" for a in found)


def test_check_thresholds_flags_with_the_source_string():
    flags = check_thresholds(get_telemetry("P-07"))
    do = next(a for a in flags if a.parameter == "do_mg_l")
    assert do.kind == "below_limit" and do.limit == 4.0
    assert do.limit_source and do.limit_source.startswith("TODO-TEAM")  # not verified yet


def test_critical_oxygen_is_high_severity():
    df = apply_events(
        baseline("P-1", END, 35, 3),
        [{"type": "do_crash", "start": "2026-10-18T00:00", "nights": 3, "depth": 3.0}],
    )
    do = next(a for a in check_thresholds(_readings(df.tail(72))) if a.parameter == "do_mg_l")
    assert do.severity == "high" and do.value < 3.0


# --- ladder ------------------------------------------------------------------------------


def test_lookup_action_ladder():
    alert = lookup_action_ladder("alert", "white_faeces")
    assert alert["allowed"] == ["L1", "L2"] and alert["condition_default_risk"] == "alert"
    assert not alert["definitions"]["L2"]["requires_approval"]
    critical = lookup_action_ladder("critical", "wssv")
    assert critical["definitions"]["L3"]["requires_approval"] is True


# --- guardrails ------------------------------------------------------------------------


def _passage(pid="KB-1"):
    return Passage(id=pid, text="t", source="S", title="T", score=0.9, grade=0.8)


def _draft(**changes):
    draft = {
        "hypotheses": [{"condition": "ehp", "citations": ["KB-1"]}],
        "action": {
            "risk_level": "alert",
            "action_level": "L2",
            "steps": [
                "Send a shrimp sample for PCR [KB-1].",
                "Reduce feeding by one ration [KB-1].",
            ],
            "citations": ["KB-1"],
        },
        "disclaimer": "Advisory only.",
    }
    draft.update(changes)
    return draft


def test_a_good_draft_passes():
    assert check_guardrails(_draft(), [_passage()]).passed


def test_uncited_step_fails_and_goes_back_to_the_advisor():
    steps = ["Send a sample for PCR [KB-1].", "Increase aeration at night."]
    verdict = check_guardrails(_draft(action={**_draft()["action"], "steps": steps}), [_passage()])
    assert not verdict.passed and verdict.revise_target == "advisor"
    assert "action step 2 has no citation" in verdict.issues[0]


def test_unknown_citation_in_a_step_fails():
    steps = ["Send a sample for PCR [KB-9]."]
    verdict = check_guardrails(_draft(action={**_draft()["action"], "steps": steps}), [_passage()])
    assert not verdict.passed and "unknown passages" in verdict.issues[0]


def test_hypothesis_with_unretrieved_citation_goes_back_to_the_diagnostician():
    hyps = [{"condition": "ehp", "citations": ["KB-7"]}]
    verdict = check_guardrails(_draft(hypotheses=hyps), [_passage()])
    assert not verdict.passed and verdict.revise_target == "diagnostician"


def test_ladder_citation_is_valid_for_the_matching_level():
    action = {
        "risk_level": "normal",
        "action_level": "L0",
        "steps": ["Continue routine monitoring [ladder:L0]."],
        "citations": [],
    }
    assert check_guardrails(_draft(hypotheses=[], action=action), []).passed
    action["steps"] = ["Continue routine monitoring [ladder:L2]."]
    assert not check_guardrails(_draft(hypotheses=[], action=action), []).passed


@pytest.mark.parametrize("text", ["use chloramphenicol in the feed", "Furazolidone bath"])
def test_banned_substances_are_blocked(text):
    action = {**_draft()["action"], "steps": [f"{text} [KB-1]."]}
    verdict = check_guardrails(_draft(action=action), [_passage()])
    assert not verdict.passed and any("banned substance" in i for i in verdict.issues)


@pytest.mark.parametrize(
    "step,flagged",
    [
        ("Add 10 mg/kg of the product to feed", True),
        ("Apply 2 ml per litre of the solution", True),
        ("Treat with 5 ppm for one hour", True),
        ("Keep dissolved oxygen above 5 mg/L at dawn", False),
        ("Check ammonia below 1 mg/L each day", False),
    ],
)
def test_dosage_detection(step, flagged):
    assert (find_dosage(step) is not None) is flagged


def test_action_level_must_fit_the_risk_level_and_disclaimer_is_required():
    action = {**_draft()["action"], "risk_level": "watch"}  # L2 is not allowed for watch
    verdict = check_guardrails(_draft(action=action, disclaimer=""), [_passage()])
    assert not verdict.passed
    assert any("not allowed for risk" in i for i in verdict.issues)
    assert any("disclaimer" in i for i in verdict.issues)


def test_step_citation_parsing():
    assert step_citations("Do this [A:p1:c0, B:p2:c1]. Then [C].") == ["A:p1:c0", "B:p2:c1", "C"]


# --- knowledge-base tools --------------------------------------------------------------


@pytest.fixture
def retriever(tmp_path):
    embedder = HashEmbedder()
    client = open_store(tmp_path / "chroma")
    collection = client.create_collection(
        collection_name(embedder), metadata={"hnsw:space": "cosine"}
    )
    docs = {
        "D:p1:c0": ("White faecal strands floating on the pond surface indicate EHP.", "ehp"),
        "D:p2:c0": ("Low dissolved oxygen before dawn stresses shrimp.", "water_quality_stress"),
    }
    collection.add(
        ids=list(docs),
        documents=[t for t, _ in docs.values()],
        embeddings=embedder.embed_documents([t for t, _ in docs.values()]),
        metadatas=[
            {"source": "D", "title": "Doc", "page": int(i[3]), "disease": d}
            for i, (_, d) in docs.items()
        ],
    )
    return Retriever(embedder, tmp_path / "chroma")


def test_search_disease_kb_returns_cited_passages(retriever):
    found = search_disease_kb("white faecal strands on the surface", k=2, retriever=retriever)
    assert found[0].id == "D:p1:c0" and found[0].source == "D" and found[0].page == 1
    with pytest.raises(ValueError, match="only the disease_kb"):
        search_disease_kb("x", collection="sop_kb", retriever=retriever)


def test_grade_passages_fills_grades_and_missing_ids_get_zero(retriever):
    passages = search_disease_kb("white faecal strands", k=2, retriever=retriever)
    llm = get_llm("agent", SETTINGS, TokenBudget(cap=10_000))
    graded = grade_passages("white faecal strands", passages, llm)
    assert [p.grade for p in graded] == [0.5, 0.5]  # the mock grades 0.5 for every id
    assert all(p.grade is None for p in passages)  # inputs are not modified
    assert grade_passages("q", [], llm) == []


def test_tool_calls_appear_in_the_trace(tmp_path):
    with RunTracer("t1", settings=SETTINGS, runs_dir=tmp_path) as tracer:
        get_telemetry("P-07", hours=24)
        lookup_action_ladder("watch", "unknown")
    names = [json.loads(line).get("name") for line in tracer.path.read_text().splitlines()]
    assert "get_telemetry" in names and "lookup_action_ladder" in names


# --- robustness: empty and very short windows, dosages in words, incomplete drafts ------------


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_very_short_windows_do_not_crash(n):
    short = _readings(baseline("P-1", END, 35, 3).head(n))
    assert detect_anomalies(short) == [] and check_thresholds(short) == []
    assert compute_trends(short)["hours"] == n


def test_empty_telemetry_raises_a_clear_error():
    for tool in (compute_trends, detect_anomalies, check_thresholds):
        with pytest.raises(telemetry_tools.TelemetryError, match="no telemetry"):
            tool([])


@pytest.mark.parametrize(
    "step",
    [
        "Use 2 grams per kg of feed",
        "Add two grams per kilogram of feed to the ration",
        "Apply half a gram per litre of the product",
        "Mix 0.5 g/L into the water",
        "Treat with 3 milligrams per litre",
    ],
)
def test_spelled_out_dosages_are_caught(step):
    assert find_dosage(step) is not None


def test_a_draft_without_an_action_or_steps_does_not_pass():
    no_action = check_guardrails({"hypotheses": [], "disclaimer": "x"}, [])
    assert not no_action.passed and any("no action steps" in i for i in no_action.issues)
    empty = {
        "action": {"risk_level": "alert", "action_level": "L2", "steps": []},
        "disclaimer": "x",
    }
    assert not check_guardrails(empty, []).passed
