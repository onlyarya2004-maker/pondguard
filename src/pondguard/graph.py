"""The LangGraph that runs one PondGuard run (spec §6.10).

    START -> supervisor --next--> monitor | diagnostician | advisor | critic | human_gate | finish
    monitor / diagnostician / advisor / critic -> supervisor ;  human_gate -> finish -> END

The Supervisor is the only router. Every node runs inside a trace span; the events each
node records become the run's human-readable step log. Token-cap errors are caught and
turn into a finish with `loop_bound_hit`, so a runaway run ends cleanly.

Durable workflow: compiled with a SQLite checkpointer, a run that reaches the human gate
(L3 or L4, or an unresolved Critic) PAUSES with its whole state saved, and `resume_run`
continues it from the checkpoint after a person decides, even in a new process. Without a
checkpointer the gate cannot pause and the report is flagged "pending approval". In
evaluation mode (`auto_approve`) the gate records that it would have paused and approves.

Pond memory (SQLite) is optional: with it, each finished run is written to `pond_history`
and `runs`, and past runs are read back for the Advisor's repeat-alert rule.
"""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from pondguard.agents.advisor import advisor_node
from pondguard.agents.critic import critic_node
from pondguard.agents.diagnostician import diagnostician_node
from pondguard.agents.gate import apply_decision, human_gate_node
from pondguard.agents.monitor import monitor_node
from pondguard.agents.supervisor import supervisor_node
from pondguard.config import Settings, get_settings
from pondguard.kb.retriever import Retriever
from pondguard.llm import LLMClient, TokenCapExceeded
from pondguard.memory import Memory, thread_id_for
from pondguard.models import ActionPlan, CauseHypothesis, PondReport, SourceRef
from pondguard.state import GraphState
from pondguard.tracing import RunTracer, current_tracer

NODES = ("supervisor", "monitor", "diagnostician", "advisor", "critic", "human_gate", "finish")
APPROVAL_STATUS = {"approve": "approved", "edit": "edited", "reject": "rejected"}


def finish_node(
    state: GraphState, *, llm: LLMClient, settings: Settings, memory: Memory | None = None
) -> dict[str, Any]:
    """Build the PondReport, save it to pond memory and close the run."""
    tracer = current_tracer()
    plan = ActionPlan(**state["action_plan"]) if state.get("action_plan") else None
    if plan is None:  # the run was stopped before the Advisor ran
        plan = ActionPlan(
            risk_level="watch",
            action_level="L0",
            steps=["Run stopped early; check the pond manually and re-run [ladder:L0]."],
            rationale=f"Run stopped early ({state.get('loop_bound_hit')}).",
        )
    diagnosis = state.get("diagnosis") or {}
    hypotheses = [CauseHypothesis(**h) for h in diagnosis.get("hypotheses", [])]
    passages = {p["id"]: p for p in state.get("retrieved_passages", [])}
    cited = list(dict.fromkeys([c for h in hypotheses for c in h.citations] + plan.citations))
    sources = [
        SourceRef(
            id=i, title=passages[i]["title"], source=passages[i]["source"], page=passages[i]["page"]
        )
        for i in cited
        if i in passages
    ]
    verdict = state.get("critic_verdict") or {}
    unresolved = [] if verdict.get("passed", True) else list(verdict.get("issues", []))
    required = settings.action_ladder.requires_approval(plan.action_level) or bool(
        state.get("needs_human")
    )
    decision = (state.get("approval") or {}).get("decision")
    if decision in APPROVAL_STATUS:
        status = APPROVAL_STATUS[decision]
    else:
        status = "pending" if required else "not_needed"
    report = PondReport(
        run_id=state["run_id"],
        pond_id=state["pond_id"],
        date=date.fromisoformat(state["date"]),
        variant=state.get("variant", "B"),  # type: ignore[arg-type]
        risk_level=plan.risk_level,
        hypotheses=hypotheses,
        action=plan,
        sources=sources,
        follow_up_question=diagnosis.get("follow_up_question"),
        needs_human=required,
        approval_status=status,  # type: ignore[arg-type]
        issues=unresolved,
    )
    latency_s = sum(c.latency_s for c in llm.calls)
    stats = {
        "steps": state.get("step_count"),
        "retrieval_rounds": diagnosis.get("retrieval_rounds", 0),
        "revisions": state.get("revisions", 0),
        "tokens": state.get("tokens_used", llm.budget.used),
        "llm_calls": llm.budget.calls,
        "latency_s": round(latency_s, 3),
        "plan_source": state.get("plan_source"),
        "loop_bound_hit": state.get("loop_bound_hit"),
        "gate_would_pause": bool(state.get("gate_would_pause")),
        "schema_fallbacks": sum(1 for c in llm.calls if c.fallback),
    }
    if memory:
        top = hypotheses[0].condition if hypotheses else None
        memory.add_history(
            pond_id=report.pond_id,
            day=state["date"],
            run_id=report.run_id,
            risk_level=report.risk_level,
            top_condition=top,
            action_level=plan.action_level,
            approval_status=status,
        )
        memory.add_run(
            run_id=report.run_id,
            pond_id=report.pond_id,
            scenario_id=state.get("scenario_id"),
            variant=report.variant,
            started_at=state.get("started_at"),
            ended_at=datetime.now(UTC).isoformat(timespec="seconds"),
            steps=stats["steps"],
            retrieval_rounds=stats["retrieval_rounds"],
            tokens=stats["tokens"],
            latency_ms=int(latency_s * 1000),
            loop_bound_hit=stats["loop_bound_hit"],
        )
    if tracer:
        suffix = " (human approval required, still pending)" if status == "pending" else ""
        tracer.event(
            "finish",
            f"report ready: risk {report.risk_level}, action {plan.action_level}, "
            f"approval {status}{suffix}",
        )
        tracer.write({"type": "report", "report": report.model_dump(mode="json"), "stats": stats})
    return {"report": report.model_dump(mode="json")}


def _traced(name: str, fn: Callable[[GraphState], dict[str, Any]], llm: LLMClient):
    """Wrap a node: trace span, collect its events, catch the token cap."""

    @functools.wraps(fn)
    def wrapper(state: GraphState) -> dict[str, Any]:
        tracer = current_tracer()
        before = len(tracer.events) if tracer else 0
        spent_before = llm.budget.used
        try:
            if tracer:
                with tracer.span(
                    name, kind="node", inputs={"step": state.get("step_count", 0)}
                ) as sp:
                    out = fn(state)
                    sp.outputs = {
                        k: v for k, v in out.items() if k in ("next", "plan", "critic_verdict")
                    }
            else:
                out = fn(state)
        except TokenCapExceeded:
            if tracer:
                tracer.event(name, "token cap reached; stopping the run")
            out = {"loop_bound_hit": "token_cap", "next": "finish"}
        out = dict(out)
        # tokens carry across a pause: a resumed run has a fresh budget in a new process
        out["tokens_used"] = state.get("tokens_used", 0) + (llm.budget.used - spent_before)
        if tracer:
            out["events"] = tracer.events[before:]
        return out

    return wrapper


def build_graph(
    llm: LLMClient,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    *,
    checkpointer: Any = None,
    memory: Memory | None = None,
    auto_approve: bool = False,
):
    settings = settings or get_settings()
    nodes: dict[str, Callable[[GraphState], dict[str, Any]]] = {
        "supervisor": functools.partial(supervisor_node, llm=llm, settings=settings),
        "monitor": functools.partial(monitor_node, llm=llm, settings=settings),
        "diagnostician": functools.partial(
            diagnostician_node, llm=llm, settings=settings, retriever=retriever
        ),
        "advisor": functools.partial(advisor_node, llm=llm, settings=settings),
        "critic": functools.partial(critic_node, settings=settings),
        "human_gate": functools.partial(
            human_gate_node,
            settings=settings,
            auto_approve=auto_approve,
            can_pause=checkpointer is not None,
            memory=memory,
        ),
        "finish": functools.partial(finish_node, llm=llm, settings=settings, memory=memory),
    }
    graph = StateGraph(GraphState)
    for name, fn in nodes.items():
        graph.add_node(name, _traced(name, fn, llm))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", lambda s: s["next"], {n: n for n in NODES if n != "supervisor"}
    )
    for name in ("monitor", "diagnostician", "advisor", "critic"):
        graph.add_edge(name, "supervisor")
    graph.add_edge("human_gate", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(
    scenario: dict, run_id: str, variant: str = "B", history: list[dict] | None = None
) -> GraphState:
    log = scenario["farmer_logs"][0]
    return {
        "run_id": run_id,
        "scenario_id": scenario["scenario_id"],
        "pond_id": scenario["pond_id"],
        "date": scenario["date"],
        "day_of_culture": scenario["day_of_culture"],
        "variant": variant,
        "trigger": "farmer_log",
        "farmer_log": log,
        "pond_history": scenario.get("history", []) if history is None else history,
        "step_count": 0,
        "retries": 0,
        "revisions": 0,
        "tokens_used": 0,
        "needs_human": False,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "events": [],
    }


def _config(thread: str, settings: Settings) -> dict[str, Any]:
    return {
        "recursion_limit": settings.loop_bounds.recursion_limit,
        "configurable": {"thread_id": thread},
    }


def _final(graph, config: dict[str, Any], out: dict[str, Any], checkpointer: Any) -> dict[str, Any]:
    """The run's end state; if the run is paused at the gate, say so."""
    if checkpointer is not None:
        snap = graph.get_state(config)
        if snap.next:
            thread = config["configurable"]["thread_id"]
            paused = {"thread_id": thread, "waiting_at": list(snap.next)}
            return {**snap.values, "__paused__": paused}
    return out


def run_scenario(
    scenario: dict,
    llm: LLMClient,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    *,
    runs_dir: Path | None = None,
    echo: Callable[[dict], None] | None = None,
    run_id: str | None = None,
    variant: str = "B",
    checkpointer: Any = None,
    memory: Memory | None = None,
    auto_approve: bool = False,
) -> tuple[GraphState, RunTracer]:
    """Run the graph once for a scenario, fully traced. Returns the final state.

    If the run pauses at the human gate, the returned state has a `__paused__` entry with the
    thread id to resume it with (`resume_run`).
    """
    settings = settings or get_settings()
    run_id = run_id or f"{scenario['scenario_id']}-{uuid.uuid4().hex[:6]}"
    tags = {
        "pond_id": scenario["pond_id"],
        "scenario_id": scenario["scenario_id"],
        "variant": variant,
    }
    kwargs = {"runs_dir": runs_dir} if runs_dir else {}
    graph = build_graph(
        llm,
        settings,
        retriever,
        checkpointer=checkpointer,
        memory=memory,
        auto_approve=auto_approve,
    )
    history = None
    if memory:
        remembered = memory.get_pond_history(scenario["pond_id"], today=scenario["date"])
        history = [*remembered, *scenario.get("history", [])]
    thread = thread_id_for(scenario["pond_id"], scenario["date"], run_id)
    config = _config(thread, settings)
    with RunTracer(run_id, settings=settings, tags=tags, echo=echo, **kwargs) as tracer:
        out = graph.invoke(initial_state(scenario, run_id, variant, history), config)
        final = _final(graph, config, out, checkpointer)
        if "__paused__" in final:
            tracer.write({"type": "paused", **final["__paused__"]})
    return final, tracer


def resume_run(
    thread_id: str,
    decision: dict[str, Any],
    llm: LLMClient,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    *,
    checkpointer: Any,
    memory: Memory | None = None,
    runs_dir: Path | None = None,
    echo: Callable[[dict], None] | None = None,
) -> tuple[GraphState, RunTracer]:
    """Continue a run that is paused at the human gate with a person's decision."""
    settings = settings or get_settings()
    run_id = thread_id.split(":")[-1]
    graph = build_graph(llm, settings, retriever, checkpointer=checkpointer, memory=memory)
    config = _config(thread_id, settings)
    snapshot = graph.get_state(config)
    if not snapshot.next:
        raise ValueError(f"nothing is waiting for approval on thread {thread_id}")
    # Check the decision against the pending draft BEFORE resuming: LangGraph stores a resume
    # value once given, so an invalid edit sent through would be replayed on every later resume.
    draft = next((i.value["draft"] for t in snapshot.tasks for i in t.interrupts), None)
    if draft is not None:
        apply_decision(ActionPlan(**draft["action"]), decision, settings)
    kwargs = {"runs_dir": runs_dir} if runs_dir else {}
    tags = {"resumed": "true"}
    with RunTracer(run_id, settings=settings, tags=tags, echo=echo, **kwargs) as tracer:
        out = graph.invoke(Command(resume=decision), config)
        final = _final(graph, config, out, checkpointer)
    return final, tracer
