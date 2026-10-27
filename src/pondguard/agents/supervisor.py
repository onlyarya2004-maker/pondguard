"""The Supervisor: plan-and-execute routing with hard bounds.

Agentic concept: planning. On its first turn the Supervisor asks the LLM for an ordered
plan from {monitor, diagnose, advise, critique}. After every specialist it decides what
runs next:

- the next plan step that has not run yet;
- `diagnose` is skipped when there are no anomalies and no symptoms (a healthy pond);
- a failing Critic verdict sends the draft back ONCE (to the Advisor, or to the
  Diagnostician for citation problems); a second failure goes to a human (flagged in the
  report) instead of looping;
- the run is forced to finish at max_supervisor_steps.

An invalid or incomplete plan falls back to the default plan, and the report can say how
often that happened (plan_source).
"""

from __future__ import annotations

from typing import Any

from pondguard.config import Settings
from pondguard.llm import DEFAULT_PLAN, LLMClient
from pondguard.state import GraphState
from pondguard.tracing import current_tracer

SYSTEM = (
    "You are the Supervisor of PondGuard, an advisory assistant for shrimp ponds. Plan the "
    "checks for one run as an ordered list drawn from: monitor, diagnose, advise, critique. "
    "Reply only with JSON that matches the schema."
)
STEP_TO_NODE = {
    "monitor": "monitor",
    "diagnose": "diagnostician",
    "advise": "advisor",
    "critique": "critic",
}


def make_plan(state: GraphState, llm: LLMClient) -> tuple[list[str], str]:
    log = state.get("farmer_log", {})
    user = (
        f"Trigger: {state.get('trigger', 'farmer_log')}. Day {state.get('day_of_culture')} of "
        f"culture. Farmer reports: symptoms {log.get('symptoms') or 'none'}, feed consumed "
        f"{log.get('feed_consumed_pct')}%, mortality {log.get('mortality_count')}. "
        "Plan the checks. Always read the telemetry (monitor) first."
    )
    result = llm.run("plan", [("system", SYSTEM), ("human", user)])
    plan = list(result.output.plan)
    if result.fallback:
        return list(DEFAULT_PLAN), "fallback"
    # A usable plan covers every step, reads telemetry first and critiques last. Skipping
    # diagnosis is decided later from the evidence, never by the planner alone.
    if set(plan) != set(DEFAULT_PLAN) or plan[0] != "monitor" or plan[-1] != "critique":
        return list(DEFAULT_PLAN), "default"
    return list(dict.fromkeys(plan)), "llm"


def _done(state: GraphState, step: str) -> bool:
    if step == "monitor":
        return bool(state.get("telemetry_summary"))
    if step == "diagnose":
        return bool(state.get("diagnosis")) or bool(state.get("diagnosis_skipped"))
    if step == "advise":
        return bool(state.get("action_plan"))
    verdict = state.get("critic_verdict")
    return bool(verdict) and verdict["passed"]


def supervisor_node(state: GraphState, *, llm: LLMClient, settings: Settings) -> dict[str, Any]:
    tracer = current_tracer()
    bounds = settings.loop_bounds
    updates: dict[str, Any] = {"step_count": state.get("step_count", 0) + 1}
    view: dict[str, Any] = {**state, **updates}

    def note(message: str) -> None:
        if tracer:
            tracer.event("supervisor", message)

    if state.get("loop_bound_hit"):  # a specialist hit a bound (e.g. the token cap): stop now
        note(f"{state['loop_bound_hit']} reached; finishing")
        return {**updates, "next": "finish"}

    if updates["step_count"] > bounds.max_supervisor_steps:
        note(f"step bound reached ({bounds.max_supervisor_steps}); finishing")
        return {**updates, "next": "finish", "loop_bound_hit": "max_supervisor_steps"}

    if not state.get("plan"):
        plan, source = make_plan(state, llm)
        updates.update(plan=plan, plan_source=source)
        view.update(updates)
        note(f"plan ({source}): {' -> '.join(plan)}")

    verdict = view.get("critic_verdict")
    if verdict and not verdict["passed"]:
        issues = "; ".join(verdict["issues"])
        if view.get("revisions", 0) < bounds.max_critic_revisions:
            target = verdict["revise_target"]
            updates.update(
                revisions=view.get("revisions", 0) + 1,
                critic_feedback=issues,
                critic_verdict=None,
                action_plan=None,
            )
            if target == "diagnostician":
                updates.update(diagnosis=None, diagnosis_skipped=False)
            view.update(updates)
            n = updates["revisions"]
            note(f"Critic sent the draft back to the {target} (revision {n}): {issues}")
        else:
            note("Critic still not satisfied after the allowed revision; flagging for a human")
            return {**updates, "next": "finish", "needs_human": True}

    for step in view["plan"]:
        if _done(view, step):
            continue
        if step == "diagnose" and not view.get("anomalies") and not view.get("symptoms"):
            updates["diagnosis_skipped"] = True
            note("no anomalies and no symptoms: skipping diagnosis")
            continue
        node = STEP_TO_NODE[step]
        note(f"next: {node}")
        return {**updates, "next": node}
    note("all planned steps done; finishing")
    return {**updates, "next": "finish"}
