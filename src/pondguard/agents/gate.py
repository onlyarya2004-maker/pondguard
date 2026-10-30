"""The human gate: containment and emergency harvest need a person's approval.

Agentic concept: human-in-the-loop with a durable workflow. When the proposed action is L3 or
L4 (the ladder marks them `requires_approval`), or the Critic could not be satisfied, the run
PAUSES at this node. LangGraph saves the whole state in SQLite; a person answers later with
`approve` / `edit` / `reject`, even after the program was restarted, and the run resumes from
the checkpoint.

Decisions (spec §6.9):
- approve: the draft goes out unchanged;
- edit: the reviewer picks another action level allowed for the risk level, and may add a note;
- reject: the action is downgraded to the highest allowed level that needs no approval (L2 for
  alert or critical), with a note that a person rejected the proposal.

Two special modes:
- evaluation mode (`auto_approve`): the gate does not pause; it records that it WOULD have paused
  (`gate_would_pause`) and approves automatically, so the evaluation can measure the gate;
- no checkpointer: a pause is impossible, so the report is flagged "pending approval" instead.
"""

from __future__ import annotations

from typing import Any

from pondguard.config import Settings
from pondguard.memory import Memory, thread_id_for
from pondguard.models import ActionPlan
from pondguard.state import GraphState
from pondguard.tools.memory_tools import request_human_approval
from pondguard.tracing import current_tracer

DECISIONS = ("approve", "edit", "reject")


def needs_gate(state: GraphState, settings: Settings) -> bool:
    plan = state.get("action_plan")
    if not plan:
        return bool(state.get("needs_human"))
    return bool(state.get("needs_human")) or settings.action_ladder.requires_approval(
        plan["action_level"]
    )


def _ladder_step(settings: Settings, level: str, extra: str = "") -> str:
    meaning = settings.action_ladder.levels[level].meaning.rstrip(".")  # type: ignore[index]
    return f"{meaning}{extra} [ladder:{level}]."


def apply_decision(plan: ActionPlan, decision: dict[str, Any], settings: Settings) -> ActionPlan:
    """Return the plan after a human decision. Raises ValueError for an invalid edit."""
    kind = decision.get("decision")
    if kind not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {kind!r}")
    ladder = settings.action_ladder
    allowed = ladder.allowed_actions[plan.risk_level]
    if kind == "approve":
        return plan
    if kind == "reject":
        safe = [lv for lv in allowed if not ladder.requires_approval(lv)]
        level = safe[-1] if safe else allowed[0]
        note = f" A human reviewer rejected the proposed {plan.action_level}."
        return plan.model_copy(
            update={
                "action_level": level,
                "steps": [_ladder_step(settings, level, note)],
                "rationale": f"{plan.rationale}{note} Downgraded to {level}.",
                "citations": [],
            }
        )
    edit = decision.get("edited_action") or {}
    level = edit.get("action_level", plan.action_level)
    if level not in allowed:
        raise ValueError(f"edited action level {level} is not allowed for risk '{plan.risk_level}'")
    note = str(edit.get("note", "")).strip()
    steps = list(plan.steps)
    if level != plan.action_level:
        steps[0] = _ladder_step(settings, level)  # the first step is always the ladder's wording
    suffix = f" Reviewer edit: {note}" if note else ""
    return plan.model_copy(
        update={
            "action_level": level,
            "steps": steps,
            "rationale": f"{plan.rationale} Reviewer set the action to {level}.{suffix}",
        }
    )


def human_gate_node(
    state: GraphState,
    *,
    settings: Settings,
    auto_approve: bool = False,
    can_pause: bool = True,
    memory: Memory | None = None,
) -> dict[str, Any]:
    tracer = current_tracer()
    plan = ActionPlan(**state["action_plan"])
    reason = (
        "the Critic could not be satisfied"
        if state.get("needs_human")
        and not settings.action_ladder.requires_approval(plan.action_level)
        else f"{plan.action_level} needs human approval"
    )

    if auto_approve:
        if tracer:
            tracer.event("gate", f"{reason}: the gate WOULD pause; auto-approved (evaluation mode)")
        decision = {"decision": "approve", "reviewer": "auto-eval"}
        return {"approval": decision, "gate_would_pause": True}

    if not can_pause:
        if tracer:
            tracer.event(
                "gate", f"{reason}: no checkpoint database, so the report is flagged pending"
            )
        return {"approval": None, "gate_would_pause": True}

    thread = thread_id_for(state["pond_id"], state["date"], state["run_id"])
    if memory:
        memory.add_pending(
            thread,
            run_id=state["run_id"],
            pond_id=state["pond_id"],
            scenario_id=state.get("scenario_id"),
            risk_level=plan.risk_level,
            action_level=plan.action_level,
        )
    if tracer:
        tracer.event("gate", f"{reason}: PAUSED for human approval (thread {thread})")
    draft = {
        "risk_level": plan.risk_level,
        "action": plan.model_dump(),
        "hypotheses": (state.get("diagnosis") or {}).get("hypotheses", []),
    }
    answer = request_human_approval(draft)  # the run stops here until `approve` resumes it

    new_plan = apply_decision(plan, answer, settings)
    if memory:
        memory.add_approval(
            state["run_id"],
            answer["decision"],
            str(answer.get("edited_action") or ""),
            str(answer.get("reviewer", "")),
        )
        memory.resolve_pending(thread)
    if tracer:
        tracer.event(
            "gate",
            f"human decision: {answer['decision']} by {answer.get('reviewer', 'unknown')} "
            f"(action {plan.action_level} -> {new_plan.action_level})",
        )
    return {
        "approval": {k: answer.get(k) for k in ("decision", "edited_action", "reviewer")},
        "action_plan": new_plan.model_dump(),
        "gate_would_pause": True,
    }
