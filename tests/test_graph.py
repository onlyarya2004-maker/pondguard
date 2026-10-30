"""The full agent graph in mock mode: planning, skip-diagnosis, corrective RAG, the critic's
feedback loop, bounds, and the Advisor's deterministic rules. No model, no network."""

import json

import pytest

from pondguard import llm as llm_module
from pondguard.agents.advisor import decide_action, decide_risk
from pondguard.config import load_settings
from pondguard.graph import run_scenario
from pondguard.kb.embeddings import HashEmbedder
from pondguard.kb.ingest import collection_name, open_store
from pondguard.kb.retriever import Retriever
from pondguard.llm import (
    CauseHypothesis,
    PassageGrade,
    PassageGrades,
    RankedHypotheses,
    SupervisorPlan,
    TokenBudget,
    get_llm,
)
from pondguard.paths import CONFIG_DIR
from pondguard.simulator.scenarios import load_scenario, simulate
from pondguard.tools import telemetry_tools

SETTINGS = load_settings(env={})
DOCS = {
    "D:p1:c0": ("White faecal strands floating on the pond surface indicate EHP infection.", "ehp"),
    "D:p2:c0": (
        "Low dissolved oxygen before dawn stresses shrimp and cuts feeding.",
        "water_quality_stress",
    ),
    "D:p3:c0": ("Vibriosis shows red discoloration and luminescence at night.", "vibriosis"),
}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("graph_data")
    simulate(CONFIG_DIR / "scenario_specs.yaml", root / "scenarios", root / "telemetry")
    return root


@pytest.fixture(autouse=True)
def _telemetry_from_test_data(data_dir, monkeypatch):
    monkeypatch.setattr(telemetry_tools, "DATA_DIR", data_dir)


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
def strong(monkeypatch):
    """The mock LLM grades every passage 0.9 and ranks `condition` (default ehp)."""
    real = llm_module.mock_response
    state = {"condition": "ehp", "plan": None}

    def fn(task, context):
        ids = context.get("passage_ids", [])
        if task == "grade":
            return PassageGrades(
                grades=[PassageGrade(passage_id=p, grade=0.9, reason="r") for p in ids]
            )
        if task == "rank" and ids:
            hyp = CauseHypothesis(
                condition=state["condition"], confidence=0.8, evidence=["e"], citations=[ids[0]]
            )
            return RankedHypotheses(hypotheses=[hyp])
        if task == "plan" and state["plan"]:
            return SupervisorPlan(plan=state["plan"], reasoning="test")
        return real(task, context)

    monkeypatch.setattr(llm_module, "mock_response", fn)
    return state


def _run(scenario_id, tmp_path, retriever, settings=SETTINGS, data_dir=None, budget=None):
    from pondguard.paths import DATA_DIR  # noqa: F401  (scenario JSON comes from the test dir)

    scenario = load_scenario(scenario_id, telemetry_tools.DATA_DIR / "scenarios")
    llm = get_llm("agent", settings, budget or TokenBudget(cap=1_000_000))
    final, tracer = run_scenario(scenario, llm, settings, retriever, runs_dir=tmp_path / "runs")
    return final, tracer, llm


def _events(final):
    return [f"{e['node']}: {e['message']}" for e in final["events"]]


def _called(llm):
    return [c.task for c in llm.calls]


# --- the four behaviours -------------------------------------------------------------


def test_healthy_pond_skips_diagnosis_and_finishes_at_l0(tmp_path, retriever, strong):
    final, tracer, llm = _run("HEALTHY-01", tmp_path, retriever)
    report = final["report"]
    assert report["risk_level"] == "normal" and report["action"]["action_level"] == "L0"
    assert any("skipping diagnosis" in e for e in _events(final))
    assert "grade" not in _called(llm) and "rank" not in _called(llm)  # no retrieval at all
    assert report["action"]["steps"] == [
        "Continue routine checks; re-check at the next reading [ladder:L0]."
    ]
    assert report["needs_human"] is False and report["approval_status"] == "not_needed"
    kinds = [json.loads(line)["type"] for line in tracer.path.read_text().splitlines()]
    assert "report" in kinds and kinds[0] == "run_start" and kinds[-1] == "run_end"


def test_ehp_scenario_is_diagnosed_cited_and_advised_at_l2(tmp_path, retriever, strong):
    final, _, llm = _run("EHP-03", tmp_path, retriever)
    report = final["report"]
    assert report["risk_level"] == "alert" and report["action"]["action_level"] == "L2"
    assert report["hypotheses"][0]["condition"] == "ehp"
    steps = report["action"]["steps"]
    assert steps[0].startswith("Send a sample for PCR") and "[ladder:L2]" in steps[0]
    assert all("[" in s for s in steps)  # every step is cited
    assert {s["id"] for s in report["sources"]}  # sources listed
    assert report["issues"] == [] and final["revisions"] == 0
    assert any("passed every check" in e for e in _events(final))


def test_ambiguous_scenario_requeries_then_abstains_with_a_question(tmp_path, retriever):
    final, _, _ = _run("AMBIG-01", tmp_path, retriever)  # mock grades 0.5: always weak
    report = final["report"]
    assert report["hypotheses"] == [] and report["follow_up_question"]
    assert report["risk_level"] == "watch" and report["action"]["action_level"] == "L0"
    assert final["retries"] == 2 and final["diagnosis"]["retrieval_rounds"] == 3
    assert sum(e.startswith("diagnostician: re-query") for e in _events(final)) == 2


def test_critic_sends_an_uncited_draft_back_once_then_passes(tmp_path, retriever, strong):
    settings = load_settings(env={"PONDGUARD_INJECT_FAULT": "uncited_step"})
    strong["condition"] = "vibriosis"
    final, _, _ = _run("CRITIC-DEMO-01", tmp_path, retriever, settings)
    events = _events(final)
    assert any("FAULT INJECTED" in e for e in events)
    assert any("draft REJECTED" in e and "no citation" in e for e in events)
    assert any("sent the draft back to the advisor (revision 1)" in e for e in events)
    assert any("passed every check" in e for e in events)
    assert final["revisions"] == 1 and final["report"]["issues"] == []
    assert all("[" in s for s in final["report"]["action"]["steps"])  # the final draft is clean


def test_fault_injection_is_off_by_default(tmp_path, retriever, strong):
    final, _, _ = _run("CRITIC-DEMO-01", tmp_path, retriever)
    assert not any("FAULT INJECTED" in e for e in _events(final)) and final["revisions"] == 0


# --- bounds -----------------------------------------------------------------------------


def test_critic_failing_again_goes_to_a_human_instead_of_looping(tmp_path, retriever, strong):
    bounds = SETTINGS.loop_bounds.model_copy(update={"max_critic_revisions": 0})
    settings = load_settings(env={"PONDGUARD_INJECT_FAULT": "uncited_step"}).model_copy(
        update={"loop_bounds": bounds}
    )
    final, _, _ = _run("EHP-03", tmp_path, retriever, settings)
    report = final["report"]
    assert report["needs_human"] and report["approval_status"] == "pending"
    assert report["issues"] and final["revisions"] == 0
    assert any("flagging for a human" in e for e in _events(final))


def test_supervisor_step_bound_stops_the_run(tmp_path, retriever, strong):
    bounds = SETTINGS.loop_bounds.model_copy(update={"max_supervisor_steps": 2})
    final, _, _ = _run(
        "EHP-03", tmp_path, retriever, SETTINGS.model_copy(update={"loop_bounds": bounds})
    )
    assert final["loop_bound_hit"] == "max_supervisor_steps"
    assert final["report"]["action"]["steps"]  # still a report, built safely
    assert any("step bound reached" in e for e in _events(final))


def test_token_cap_ends_the_run_cleanly(tmp_path, retriever, strong):
    final, _, _ = _run("EHP-03", tmp_path, retriever, budget=TokenBudget(cap=1))
    assert final["loop_bound_hit"] == "token_cap" and final["report"]
    assert any("token cap reached" in e for e in _events(final))


def test_an_incomplete_plan_falls_back_to_the_default_plan(tmp_path, retriever, strong):
    strong["plan"] = ["monitor", "advise"]  # forgets diagnose and critique
    final, _, _ = _run("HEALTHY-01", tmp_path, retriever)
    assert final["plan_source"] == "default"
    assert final["plan"] == ["monitor", "diagnose", "advise", "critique"]


# --- Advisor rules ---------------------------------------------------------------------


def _state(hyp=None, anomalies=(), symptoms=(), mortality=0, skipped_diagnosis=False):
    diagnosis = (
        None
        if skipped_diagnosis
        else {"hypotheses": [{"condition": hyp}] if hyp else [], "abstained": not hyp}
    )
    return {
        "diagnosis": diagnosis,
        "anomalies": [{"severity": s, "parameter": f"p{i}"} for i, s in enumerate(anomalies)],
        "symptoms": list(symptoms),
        "farmer_log": {"mortality_count": mortality},
    }


@pytest.mark.parametrize(
    "state,risk,level",
    [
        (_state(skipped_diagnosis=True), "normal", "L0"),
        (_state(symptoms=["lethargy"]), "watch", "L0"),  # abstained, vague
        (_state(anomalies=["low"]), "watch", "L1"),
        (_state(anomalies=["high"]), "critical", "L2"),  # severe water quality, abstained
        (_state(hyp="ehp"), "alert", "L2"),
        (
            _state(hyp="ehp", anomalies=["high"]),
            "critical",
            "L2",
        ),  # raised; EHP is not fast-killing
        (_state(hyp="wssv"), "critical", "L3"),
        (_state(hyp="water_quality_stress", anomalies=["high"]), "critical", "L2"),
        (
            _state(hyp="water_quality_stress", anomalies=["medium"]),
            "watch",
            "L1",
        ),  # one mild signal
        (_state(hyp="water_quality_stress", anomalies=["medium", "medium"]), "alert", "L2"),
        (
            _state(
                hyp="water_quality_stress", anomalies=["medium"], symptoms=["surfacing_at_dawn"]
            ),
            "alert",
            "L2",
        ),
        (_state(hyp="water_quality_stress", anomalies=["low"]), "watch", "L1"),
        (
            _state(hyp="vibriosis", mortality=60),
            "critical",
            "L2",
        ),  # mortality raises, not a disease default
    ],
)
def test_risk_and_action_rules(state, risk, level):
    got_risk, _ = decide_risk(state, SETTINGS)
    assert got_risk == risk
    assert decide_action(got_risk, state, SETTINGS) == level


# --- robustness ---------------------------------------------------------------------------


def test_missing_telemetry_does_not_crash_and_is_not_called_healthy(tmp_path, retriever):
    scenario = load_scenario("HEALTHY-01", telemetry_tools.DATA_DIR / "scenarios")
    scenario = {**scenario, "scenario_id": "NO-DATA", "pond_id": "P-404"}
    llm = get_llm("agent", SETTINGS, TokenBudget(cap=1_000_000))
    final, _ = run_scenario(scenario, llm, SETTINGS, retriever, runs_dir=tmp_path / "runs")
    report = final["report"]
    assert report["risk_level"] == "watch"  # never "normal" without sensor data
    assert any("Telemetry unavailable" in e["message"] for e in final["events"])
    assert "no telemetry" in report["action"]["rationale"]


def test_critic_sending_the_draft_to_the_diagnostician_re_runs_diagnosis(
    tmp_path, retriever, strong, monkeypatch
):
    from pondguard.agents import critic as critic_module
    from pondguard.models import CriticVerdict

    real, calls = critic_module.check_guardrails, {"n": 0}

    def first_blames_the_diagnosis(draft, passages, settings=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return CriticVerdict(
                passed=False, issues=["hypothesis lacks evidence"], revise_target="diagnostician"
            )
        return real(draft, passages, settings)

    monkeypatch.setattr(critic_module, "check_guardrails", first_blames_the_diagnosis)
    final, _, llm = _run("EHP-03", tmp_path, retriever)
    assert final["revisions"] == 1 and final["report"]["issues"] == []
    assert _called(llm).count("rank") == 2  # the Diagnostician ran again
    assert any("back to the diagnostician" in e for e in _events(final))
