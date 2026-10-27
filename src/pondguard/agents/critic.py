"""The Critic: reflection on the Advisor's draft, with deterministic checks.

Agentic concept: reflection / self-correction. The Critic does not write anything; it
runs `check_guardrails` on the draft (citations on every hypothesis and step, no banned
substance, no dosage, action level allowed for the risk, disclaimer present) and returns
a verdict. A failing verdict goes back to the Supervisor, which sends the draft back once.
"""

from __future__ import annotations

from typing import Any

from pondguard.config import Settings
from pondguard.models import DISCLAIMER, Passage
from pondguard.state import GraphState
from pondguard.tools.guardrail_tools import check_guardrails
from pondguard.tracing import current_tracer


def critic_node(state: GraphState, *, settings: Settings) -> dict[str, Any]:
    tracer = current_tracer()
    diagnosis = state.get("diagnosis") or {}
    draft = {
        "hypotheses": diagnosis.get("hypotheses", []),
        "action": state["action_plan"],
        "disclaimer": DISCLAIMER,
    }
    passages = [Passage(**p) for p in state.get("retrieved_passages", [])]
    verdict = check_guardrails(draft, passages, settings)
    if tracer:
        if verdict.passed:
            tracer.event("critic", "draft passed every check")
        else:
            tracer.event(
                "critic",
                f"draft REJECTED ({verdict.revise_target}): " + "; ".join(verdict.issues),
                issues=verdict.issues,
            )
    return {"critic_verdict": verdict.model_dump()}
