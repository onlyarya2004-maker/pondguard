"""The L0-L4 action ladder as a tool (spec §3.4).

`lookup_action_ladder` tells the Advisor which action levels are allowed for a risk
level, what each means, and whether a human must approve it. The rules live in
config/action_ladder.yaml; this tool only reads them.
"""

from __future__ import annotations

from typing import Any

from pondguard.config import Settings, get_settings
from pondguard.models import Condition, RiskLevel
from pondguard.tracing import traced_tool


@traced_tool
def lookup_action_ladder(
    risk: RiskLevel, condition: Condition, settings: Settings | None = None
) -> dict[str, Any]:
    ladder = (settings or get_settings()).action_ladder
    allowed = ladder.allowed_actions[risk]
    return {
        "risk": risk,
        "condition": condition,
        "condition_default_risk": ladder.condition_default_risk[condition],
        "allowed": allowed,
        "definitions": {
            level: {
                "name": ladder.levels[level].name,
                "meaning": ladder.levels[level].meaning,
                "requires_approval": ladder.levels[level].requires_approval,
            }
            for level in allowed
        },
    }
