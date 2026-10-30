"""Output guardrails used by the Critic (reflection) -- deterministic, no LLM.

`check_guardrails` inspects a report draft and returns a CriticVerdict. Checks:
1. every hypothesis cites at least one retrieved passage, and only retrieved ones;
2. every action step ends with a citation tag `[id, id]` naming a retrieved passage
   (or the policy source `ladder:<level>` for fixed monitoring steps);
3. no banned substance appears anywhere in the draft;
4. no dosage appears in the action steps;
5. the action level is allowed for the risk level;
6. the advisory disclaimer is present.

A failure names the target to send the draft back to: the Diagnostician when
hypothesis citations are wrong, otherwise the Advisor.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pondguard.config import Settings, get_settings
from pondguard.models import CriticVerdict, Passage
from pondguard.tracing import traced_tool

CITATION_TAG = re.compile(r"\[([^\[\]]+)\]")
# A dose is "number + amount-per-something". Only flagged next to an action verb, so
# water-quality readings such as "keep oxygen above 5 mg/L" are not mistaken for doses.
_NUM = r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|half)"
_UNIT = r"(?:mg|milligrams?|g|grams?|kg|kilograms?|ml|millilit(?:re|er)s?|l|lit(?:re|er)s?|ppm)"
_PER = r"(?:kg|kilograms?|g|grams?|l|lit(?:re|er)s?|m3|m2|ha|hectares?|tons?|tonnes?|day|days)"
DOSE = re.compile(
    rf"\b{_NUM}\s*(?:an?\s+)?{_UNIT}\s*(?:/|per\s+)\s*{_PER}\b|\b{_NUM}\s*ppm\b",
    re.IGNORECASE,
)
DOSE_VERBS = re.compile(
    r"\b(add|apply|use|dose|dosage|treat|administer|mix|dissolve|spray|broadcast|sprinkle|"
    r"feed|supplement|dip|bath)\w*",
    re.IGNORECASE,
)


def step_citations(step: str) -> list[str]:
    ids: list[str] = []
    for tag in CITATION_TAG.findall(step):
        ids += [part.strip() for part in tag.split(",") if part.strip()]
    return ids


def strip_citations(step: str) -> str:
    text = CITATION_TAG.sub("", step)
    return re.sub(r"\s+([.,;:])", r"\1", text).strip()


def find_dosage(step: str) -> str | None:
    for match in DOSE.finditer(step):
        window = step[max(0, match.start() - 60) : match.end() + 20]
        window = re.sub(r"dissolved\s+oxygen", "oxygen", window, flags=re.IGNORECASE)
        if DOSE_VERBS.search(window):
            return match.group(0)
    return None


def banned_pattern(names: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)


@traced_tool
def check_guardrails(
    report_draft: dict[str, Any], passages: list[Passage], settings: Settings | None = None
) -> CriticVerdict:
    settings = settings or get_settings()
    known = {p.id for p in passages}
    issues: list[str] = []
    revise = "advisor"

    action = report_draft.get("action") or {}
    level = action.get("action_level")
    risk = action.get("risk_level")

    for hyp in report_draft.get("hypotheses", []):
        cites = hyp.get("citations", [])
        if not cites or not set(cites) <= known:
            unknown = sorted(set(cites) - known)
            issues.append(
                f"hypothesis '{hyp.get('condition')}' has missing or unknown citations {unknown}"
            )
            revise = "diagnostician"

    if not action.get("steps"):
        issues.append("the draft has no action steps")
    for number, step in enumerate(action.get("steps", []), start=1):
        cites = step_citations(step)
        valid = known | {f"ladder:{level}"}
        if not cites:
            issues.append(f"action step {number} has no citation: '{strip_citations(step)[:80]}'")
        elif not set(cites) <= valid:
            issues.append(
                f"action step {number} cites unknown passages {sorted(set(cites) - valid)}"
            )
        dose = find_dosage(strip_citations(step))
        if dose:
            issues.append(
                f"action step {number} contains a dosage ('{dose}'); dosages are not allowed"
            )

    if settings.banned.substances:
        found = banned_pattern(settings.banned.all_names()).findall(
            json.dumps(report_draft, default=str)
        )
        if found:
            issues.append(f"banned substance mentioned: {sorted({f.lower() for f in found})}")

    if risk and level and level not in settings.action_ladder.allowed_actions.get(risk, []):
        allowed = settings.action_ladder.allowed_actions[risk]
        issues.append(f"action level {level} is not allowed for risk '{risk}' (allowed: {allowed})")

    if not str(report_draft.get("disclaimer", "")).strip():
        issues.append("the advisory disclaimer is missing")

    if issues:
        return CriticVerdict(passed=False, issues=issues, revise_target=revise)
    return CriticVerdict(passed=True)
