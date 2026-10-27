"""The LangGraph that runs one PondGuard run (spec §6.10, R3 scope).

    START -> supervisor --next--> monitor | diagnostician | advisor | critic | finish
    monitor / diagnostician / advisor / critic -> supervisor ;  finish -> END

The Supervisor is the only router. Every node runs inside a trace span; the events each
node records become the run's human-readable step log. Token-cap errors are caught and
turn into a finish with `loop_bound_hit`, so a runaway run ends cleanly.

Not in R3 scope (built for R4): the human gate (L3/L4 reports are marked "pending
approval" and flagged, but do not pause), SQLite checkpointing and pond memory.
"""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from pondguard.agents.advisor import advisor_node
from pondguard.agents.critic import critic_node
from pondguard.agents.diagnostician import diagnostician_node
from pondguard.agents.monitor import monitor_node
from pondguard.agents.supervisor import supervisor_node
from pondguard.config import Settings, get_settings
from pondguard.kb.retriever import Retriever
from pondguard.llm import LLMClient, TokenCapExceeded
from pondguard.models import ActionPlan, CauseHypothesis, PondReport, SourceRef
from pondguard.state import GraphState
from pondguard.tracing import RunTracer, current_tracer

NODES = ("supervisor", "monitor", "diagnostician", "advisor", "critic", "finish")


def finish_node(state: GraphState, *, llm: LLMClient, settings: Settings) -> dict[str, Any]:
    """Build the PondReport and close the run."""
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
    approval = plan.action_level in ("L3", "L4") or bool(state.get("needs_human"))
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
        needs_human=approval,
        approval_status="pending" if approval else "not_needed",
        issues=unresolved,
    )
    if tracer:
        suffix = " (human approval would be required; the gate is built for R4)" if approval else ""
        tracer.event(
            "finish", f"report ready: risk {report.risk_level}, action {plan.action_level}{suffix}"
        )
        tracer.write(
            {
                "type": "report",
                "report": report.model_dump(mode="json"),
                "stats": {
                    "steps": state.get("step_count"),
                    "retrieval_rounds": diagnosis.get("retrieval_rounds", 0),
                    "revisions": state.get("revisions", 0),
                    "tokens": llm.budget.used,
                    "llm_calls": llm.budget.calls,
                    "plan_source": state.get("plan_source"),
                    "loop_bound_hit": state.get("loop_bound_hit"),
                },
            }
        )
    return {"report": report.model_dump(mode="json")}


def _traced(name: str, fn: Callable[[GraphState], dict[str, Any]], llm: LLMClient):
    """Wrap a node: trace span, collect its events, catch the token cap."""

    @functools.wraps(fn)
    def wrapper(state: GraphState) -> dict[str, Any]:
        tracer = current_tracer()
        before = len(tracer.events) if tracer else 0
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
        out["tokens_used"] = llm.budget.used
        if tracer:
            out["events"] = tracer.events[before:]
        return out

    return wrapper


def build_graph(
    llm: LLMClient, settings: Settings | None = None, retriever: Retriever | None = None
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
        "finish": functools.partial(finish_node, llm=llm, settings=settings),
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
    graph.add_edge("finish", END)
    return graph.compile()


def initial_state(scenario: dict, run_id: str, variant: str = "B") -> GraphState:
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
        "pond_history": scenario.get("history", []),
        "step_count": 0,
        "retries": 0,
        "revisions": 0,
        "tokens_used": 0,
        "needs_human": False,
        "events": [],
    }


def run_scenario(
    scenario: dict,
    llm: LLMClient,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    *,
    runs_dir: Path | None = None,
    echo: Callable[[dict], None] | None = None,
    run_id: str | None = None,
) -> tuple[GraphState, RunTracer]:
    """Run the graph once for a scenario, fully traced. Returns the final state."""
    settings = settings or get_settings()
    run_id = run_id or f"{scenario['scenario_id']}-{uuid.uuid4().hex[:6]}"
    tags = {"pond_id": scenario["pond_id"], "scenario_id": scenario["scenario_id"], "variant": "B"}
    kwargs = {"runs_dir": runs_dir} if runs_dir else {}
    graph = build_graph(llm, settings, retriever)
    with RunTracer(run_id, settings=settings, tags=tags, echo=echo, **kwargs) as tracer:
        final = graph.invoke(
            initial_state(scenario, run_id),
            {"recursion_limit": settings.loop_bounds.recursion_limit},
        )
    return final, tracer
