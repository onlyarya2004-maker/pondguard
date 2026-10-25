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
    next: NextNode

    # Monitor
    telemetry_summary: dict
    anomalies: list[dict]
    symptoms: list[str]

    # Diagnostician (corrective RAG)
    queries: list[str]
    retrieved_passages: list[dict]
    diagnosis: dict

    # Advisor and Critic (reflection)
    action_plan: dict
    critic_verdict: dict
    critic_feedback: str
    follow_up_question: str | None

    # Loop counters checked against config/settings.yaml loop_bounds
    step_count: int
    retries: int
    revisions: int
    tokens_used: int

    # Human-in-the-loop
    needs_human: bool
    approval: dict | None

    # Output
    report: dict
    events: Annotated[list[dict], operator.add]


def event(node: str, message: str, **data: object) -> dict:
    """One entry for the human-readable step log, e.g. event("monitor", "3 anomalies")."""
    return {"node": node, "message": message, **data}
