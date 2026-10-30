"""The LangGraph state shared by every node in a run (spec §6.5).

Agentic concept: short-term working memory. The graph state is the agent's
scratchpad for one run: the plan, what each specialist found, loop counters
for the bounds, and a human-readable event log. `events` uses an append-only
reducer, so each node adds its own step to the log and nothing overwrites
earlier steps; the UI and traces show this log.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

Trigger = Literal["farmer_log", "scheduled_anomaly"]
NextNode = Literal["monitor", "diagnostician", "advisor", "critic", "human_gate", "finish"]


class GraphState(TypedDict, total=False):
    # Identity of the run
    run_id: str
    scenario_id: str
    pond_id: str
    date: str
    day_of_culture: int
    variant: str
    trigger: Trigger

    # Context loaded at the start
    farmer_log: dict
    pond_history: list[dict]

    # Supervisor routing (plan-and-execute)
    plan: list[str]
    plan_source: str  # "llm", "fallback" (model output invalid) or "default" (plan rejected)
    next: NextNode
    diagnosis_skipped: bool  # no anomalies and no symptoms: nothing to diagnose
    loop_bound_hit: str | None  # which bound stopped the run, if any

    # Monitor
    telemetry_summary: dict
    anomalies: list[dict]
    symptoms: list[str]
    telemetry_missing: bool  # no usable sensor data: the pond cannot be called healthy

    # Diagnostician (corrective RAG)
    queries: list[str]
    retrieved_passages: list[dict]
    relevant_ids: list[str]  # passages graded relevant in the last retrieval round
    diagnosis: dict | None

    # Advisor and Critic (reflection)
    action_plan: dict | None
    critic_verdict: dict | None
    critic_feedback: str | None
    follow_up_question: str | None

    # Loop counters checked against config/settings.yaml loop_bounds
    step_count: int
    retries: int
    revisions: int
    tokens_used: int

    # Human-in-the-loop
    needs_human: bool
    approval: dict | None
    gate_would_pause: bool  # the human gate was reached (paused, auto-approved or flagged)
    started_at: str

    # Output
    report: dict
    events: Annotated[list[dict], operator.add]


def event(node: str, message: str, **data: object) -> dict:
    """One entry for the human-readable step log, e.g. event("monitor", "3 anomalies")."""
    return {"node": node, "message": message, **data}
