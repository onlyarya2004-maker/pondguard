"""The full agent graph in mock mode: planning, skip-diagnosis, corrective RAG, the critic's
feedback loop, bounds, and the Advisor's deterministic rules. No model, no network."""

import json

import pytest

from pondguard import llm as llm_module
from pondguard.agents.advisor import decide_action, decide_risk
from pondguard.config import load_settings
from pondguard.graph import resume_run, run_scenario
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
from pondguard.memory import Memory, open_checkpointer
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
    assert any("sending it to a human" in e for e in _events(final))


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


def test_variant_a_runs_the_whole_graph_with_a_single_search(tmp_path, retriever, strong):
    scenario = load_scenario("EHP-03", telemetry_tools.DATA_DIR / "scenarios")
    llm = get_llm("agent", SETTINGS, TokenBudget(cap=1_000_000))
    final, _ = run_scenario(
        scenario, llm, SETTINGS, retriever, runs_dir=tmp_path / "runs", variant="A"
    )
    assert final["report"]["variant"] == "A" and final["report"]["hypotheses"]
    assert "grade" not in _called(llm) and final["diagnosis"]["retrieval_rounds"] == 1


# --- human gate, checkpointing and pond memory (Review 4) ----------------------------------


def _wssv_run(tmp_path, retriever, *, history=None, **kwargs):
    scenario = load_scenario("WSSV-01", telemetry_tools.DATA_DIR / "scenarios")
    if history is not None:
        scenario = {**scenario, "history": history}
    llm = get_llm("agent", SETTINGS, TokenBudget(cap=1_000_000))
    return run_scenario(scenario, llm, SETTINGS, retriever, runs_dir=tmp_path / "runs", **kwargs)


def _restart(db):
    """A fresh checkpointer, memory and LLM, as in a new process."""
    return open_checkpointer(db), Memory(db), get_llm("agent", SETTINGS, TokenBudget(cap=1_000_000))


def test_wssv_pauses_at_the_gate_and_resumes_from_the_checkpoint_after_a_restart(
    tmp_path, retriever, strong
):
    strong["condition"] = "wssv"
    db = tmp_path / "pondguard.db"
    first = open_checkpointer(db)
    final, tracer = _wssv_run(tmp_path, retriever, checkpointer=first, memory=Memory(db))
    assert "report" not in final and final["__paused__"]["waiting_at"] == ["human_gate"]
    assert final["action_plan"]["action_level"] == "L3"  # critical, contained: needs approval
    thread = final["__paused__"]["thread_id"]
    assert [p["thread_id"] for p in Memory(db).list_pending()] == [thread]
    first.conn.close()

    checkpointer, memory, llm = _restart(db)
    done, _ = resume_run(
        thread,
        {"decision": "approve", "reviewer": "Asha"},
        llm,
        SETTINGS,
        retriever,
        checkpointer=checkpointer,
        memory=memory,
        runs_dir=tmp_path / "runs",
    )
    report = done["report"]
    assert report["approval_status"] == "approved" and report["action"]["action_level"] == "L3"
    assert memory.list_pending() == []
    assert [a["decision"] for a in memory.approvals_for(done["run_id"])] == ["approve"]
    rows = memory.get_pond_history("P-21", today="2026-10-21")
    assert [(r["risk_level"], r["action_level"], r["approval_status"]) for r in rows] == [
        ("critical", "L3", "approved")
    ]
    kinds = [json.loads(line) for line in tracer.path.read_text().splitlines()]
    assert any(r.get("status") == "paused" for r in kinds)  # a pause is not recorded as an error
    assert not any(r.get("status") == "error" for r in kinds)


def test_reject_downgrades_to_the_highest_level_that_needs_no_approval(tmp_path, retriever, strong):
    strong["condition"] = "wssv"
    db = tmp_path / "pondguard.db"
    final, _ = _wssv_run(tmp_path, retriever, checkpointer=open_checkpointer(db), memory=Memory(db))
    checkpointer, memory, llm = _restart(db)
    done, _ = resume_run(
        final["__paused__"]["thread_id"],
        {"decision": "reject", "reviewer": "Asha"},
        llm,
        SETTINGS,
        retriever,
        checkpointer=checkpointer,
        memory=memory,
        runs_dir=tmp_path / "runs",
    )
    report = done["report"]
    assert report["approval_status"] == "rejected" and report["action"]["action_level"] == "L2"
    assert (
        "rejected" in report["action"]["steps"][0] and "[ladder:L2]" in report["action"]["steps"][0]
    )


def test_edit_changes_the_level_within_what_the_risk_allows(tmp_path, retriever, strong):
    strong["condition"] = "wssv"
    db = tmp_path / "pondguard.db"
    final, _ = _wssv_run(tmp_path, retriever, checkpointer=open_checkpointer(db), memory=Memory(db))
    thread = final["__paused__"]["thread_id"]
    checkpointer, memory, llm = _restart(db)
    with pytest.raises(ValueError, match="not allowed"):  # L1 is not allowed for a critical pond
        resume_run(
            thread,
            {"decision": "edit", "edited_action": {"action_level": "L1"}},
            llm,
            SETTINGS,
            retriever,
            checkpointer=checkpointer,
            memory=memory,
            runs_dir=tmp_path / "runs",
        )
    assert [p["thread_id"] for p in memory.list_pending()] == [thread]  # still waiting
    checkpointer, memory, llm = _restart(db)
    done, _ = resume_run(
        thread,
        {
            "decision": "edit",
            "reviewer": "Asha",
            "edited_action": {"action_level": "L4", "note": "harvest now"},
        },
        llm,
        SETTINGS,
        retriever,
        checkpointer=checkpointer,
        memory=memory,
        runs_dir=tmp_path / "runs",
    )
    report = done["report"]
    assert report["approval_status"] == "edited" and report["action"]["action_level"] == "L4"
    assert "harvest now" in report["action"]["rationale"]


def test_evaluation_mode_records_that_the_gate_would_pause_and_approves(
    tmp_path, retriever, strong
):
    strong["condition"] = "wssv"
    final, _ = _wssv_run(tmp_path, retriever, auto_approve=True)
    assert final["gate_would_pause"] is True and "__paused__" not in final
    assert final["report"]["approval_status"] == "approved"
    assert any("auto-approved (evaluation mode)" in e for e in _events(final))


def test_without_a_checkpoint_database_the_report_is_flagged_pending(tmp_path, retriever, strong):
    strong["condition"] = "wssv"
    final, _ = _wssv_run(tmp_path, retriever)
    assert final["report"]["needs_human"] and final["report"]["approval_status"] == "pending"


def test_l0_to_l2_runs_never_reach_the_gate(tmp_path, retriever, strong):
    db = tmp_path / "pondguard.db"
    scenario = load_scenario("EHP-03", telemetry_tools.DATA_DIR / "scenarios")
    llm = get_llm("agent", SETTINGS, TokenBudget(cap=1_000_000))
    final, _ = run_scenario(
        scenario,
        llm,
        SETTINGS,
        retriever,
        runs_dir=tmp_path / "runs",
        checkpointer=open_checkpointer(db),
        memory=Memory(db),
    )
    assert "__paused__" not in final and final["report"]["approval_status"] == "not_needed"
    assert "gate_would_pause" not in final


def test_three_alerts_in_a_row_escalate_the_risk_one_level(tmp_path, retriever, strong):
    scenario = load_scenario("EHP-03", telemetry_tools.DATA_DIR / "scenarios")

    def risk_with(history):
        llm = get_llm("agent", SETTINGS, TokenBudget(cap=1_000_000))
        final, _ = run_scenario(
            {**scenario, "history": history}, llm, SETTINGS, retriever, runs_dir=tmp_path / "runs"
        )
        return final

    one = risk_with([{"date": "2026-10-19", "risk_level": "alert", "action_level": "L2"}])
    assert one["report"]["risk_level"] == "alert"
    two = risk_with(
        [
            {"date": "2026-10-19", "risk_level": "alert", "action_level": "L2"},
            {"date": "2026-10-18", "risk_level": "alert", "action_level": "L2"},
        ]
    )
    assert two["report"]["risk_level"] == "critical"
    assert any("escalated one level" in e["message"] for e in two["events"])
    gap = risk_with(
        [
            {"date": "2026-10-19", "risk_level": "alert", "action_level": "L2"},
            {"date": "2026-10-17", "risk_level": "alert", "action_level": "L2"},
        ]
    )
    assert gap["report"]["risk_level"] == "alert"  # not consecutive days


def test_memory_returns_past_runs_newest_first_and_only_before_today(tmp_path):
    memory = Memory(tmp_path / "m.db")
    for day, risk in [("2026-10-17", "watch"), ("2026-10-18", "alert"), ("2026-10-20", "critical")]:
        memory.add_history(
            pond_id="P-1",
            day=day,
            run_id=f"r{day}",
            risk_level=risk,
            top_condition=None,
            action_level="L1",
            approval_status="not_needed",
        )
    memory.add_history(
        pond_id="P-2",
        day="2026-10-19",
        run_id="x",
        risk_level="alert",
        top_condition=None,
        action_level="L2",
        approval_status="not_needed",
    )
    rows = memory.get_pond_history("P-1", days=14, today="2026-10-20")
    assert [r["date"] for r in rows] == [
        "2026-10-18",
        "2026-10-17",
    ]  # today excluded, other pond excluded
    assert memory.get_pond_history("P-1", days=1, today="2026-10-20") == []


def test_pending_queue_is_idempotent_and_resolves(tmp_path):
    memory = Memory(tmp_path / "m.db")
    memory.add_pending(
        "P-1:2026-10-20:r1",
        run_id="r1",
        pond_id="P-1",
        scenario_id="S",
        risk_level="critical",
        action_level="L3",
    )
    memory.add_pending("P-1:2026-10-20:r1", run_id="r1")  # a resumed node re-runs: no duplicate
    assert len(memory.list_pending()) == 1
    memory.resolve_pending("P-1:2026-10-20:r1")
    assert memory.list_pending() == []


def test_tokens_carry_across_a_pause_and_resume(tmp_path, retriever, strong):
    strong["condition"] = "wssv"
    db = tmp_path / "pondguard.db"
    final, _ = _wssv_run(tmp_path, retriever, checkpointer=open_checkpointer(db), memory=Memory(db))
    before = final["tokens_used"]
    assert before > 0
    checkpointer, memory, llm = _restart(db)  # a fresh budget, as in a new process
    done, _ = resume_run(
        final["__paused__"]["thread_id"],
        {"decision": "approve", "reviewer": "A"},
        llm,
        SETTINGS,
        retriever,
        checkpointer=checkpointer,
        memory=memory,
        runs_dir=tmp_path / "runs",
    )
    assert done["tokens_used"] == before  # nothing lost, nothing double-counted
