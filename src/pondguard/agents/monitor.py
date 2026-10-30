"""The Monitor: turns raw telemetry into trends and anomaly flags (ReAct-style tool use).

Agentic concept: tool use. The Monitor calls four tools in a fixed order (visible as
spans in the trace) and then asks the LLM for a short plain-language summary. The
tools compute everything numeric; the LLM only describes what they found.
"""

from __future__ import annotations

from typing import Any

from pondguard.config import Settings
from pondguard.llm import LLMClient
from pondguard.state import GraphState
from pondguard.tools.telemetry_tools import (
    TelemetryError,
    check_thresholds,
    compute_trends,
    detect_anomalies,
    get_telemetry,
)
from pondguard.tracing import current_tracer

SYSTEM = (
    "You are the Monitor of PondGuard, an advisory assistant for shrimp ponds. Summarise "
    "the pond's water-quality situation in two short sentences using only the numbers given."
)


def monitor_node(state: GraphState, *, llm: LLMClient, settings: Settings) -> dict[str, Any]:
    tracer = current_tracer()
    pond_id = state["pond_id"]
    try:
        readings = get_telemetry(
            pond_id, hours=settings.telemetry.window_hours, scenario_id=state.get("scenario_id")
        )
        trends = compute_trends(readings)
        anomalies = detect_anomalies(readings, settings) + check_thresholds(readings, settings)
    except (TelemetryError, FileNotFoundError) as exc:
        # No sensor data must not crash the run, and must not look like a healthy pond: the
        # Advisor treats `telemetry_missing` as "cannot confirm the pond is healthy".
        text = f"Telemetry unavailable ({exc}); water quality could not be assessed."
        if tracer:
            tracer.event("monitor", text)
        return {
            "telemetry_summary": {"text": text, "dawn_min_do": []},
            "anomalies": [],
            "symptoms": list(state.get("farmer_log", {}).get("symptoms", [])),
            "telemetry_missing": True,
        }

    facts = [f"{len(readings)} hourly readings."]
    facts.append(
        "Dawn oxygen minima (mg/L): " + ", ".join(str(m["do_mg_l"]) for m in trends["dawn_min_do"])
    )
    for a in anomalies:
        facts.append(f"{a.parameter}: {a.kind}, {a.severity} ({a.window}), value {a.value}")
    if not anomalies:
        facts.append("No anomalies were detected.")
    summary = llm.run("summarise", [("system", SYSTEM), ("human", "\n".join(facts))]).output.summary
    if len(summary.split()) < 4:  # a bare label instead of a sentence: use a plain statement
        found = ", ".join(f"{a.parameter} {a.kind}" for a in anomalies) or "none"
        summary = f"{len(readings)} readings checked; anomalies: {found}."

    if tracer:
        tracer.event(
            "monitor",
            f"{len(readings)} readings, {len(anomalies)} anomalies"
            + (": " + "; ".join(f"{a.parameter} {a.kind}" for a in anomalies) if anomalies else ""),
        )
        tracer.event("monitor", summary)
    return {
        "telemetry_summary": {"text": summary, "dawn_min_do": trends["dawn_min_do"]},
        "anomalies": [a.model_dump() for a in anomalies],
        "symptoms": list(state.get("farmer_log", {}).get("symptoms", [])),
    }
