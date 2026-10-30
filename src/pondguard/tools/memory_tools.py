"""Memory and approval tools: `get_pond_history` and `request_human_approval`.

`request_human_approval` wraps LangGraph's `interrupt()`: the graph stops, its state is saved
in the checkpoint database, and the run resumes only when a person answers (see the `approve`
command). It needs a checkpointer; the gate node checks that before calling it.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from pondguard.memory import Memory
from pondguard.tracing import traced_tool


@traced_tool
def get_pond_history(pond_id: str, days: int = 14, memory: Memory | None = None) -> list[dict]:
    """Past runs for the pond (risk, cause, action, approval), newest first."""
    return (memory or Memory()).get_pond_history(pond_id, days)


@traced_tool
def request_human_approval(report_draft: dict[str, Any]) -> dict[str, Any]:
    """Pause for a person. Returns {decision, edited_action, reviewer} once resumed."""
    answer = interrupt({"type": "approval_request", "draft": report_draft})
    return dict(answer)
