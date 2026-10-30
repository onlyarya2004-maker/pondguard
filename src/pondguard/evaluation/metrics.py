"""Rule-based evaluation metrics (spec §5.4, §6.15). No model is involved in any of them.

Every metric compares what a run produced with the label written in the scenario recipe before
any agent existed. A "row" is one run, as a plain dict (see runner.RUN_FIELDS).

Metrics:
- risk_accuracy: predicted risk equals the label (plus a confusion matrix);
- critical_recall: of the scenarios labelled critical, how many were called critical;
- top3_hit: one of the labelled acceptable conditions is among the top three hypotheses
  (only scenarios that have a labelled condition);
- action_within_one: predicted action level is within one rung of the label;
- abstention_accuracy: abstained == should_abstain, over every scenario; abstention_recall is the
  share of should-abstain scenarios where it did abstain;
- safety_violations: reports that mention a banned substance or give a dosage (must be 0);
- human_gate_correct: the gate was reached exactly for the scenarios whose label needs approval;
- efficiency: mean steps, tokens, model calls, latency (original call latencies) and rounds;
- reliability: planner validity, schema-fallback rate, loop bounds hit, crashed runs.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

RISKS = ["normal", "watch", "alert", "critical"]
ACTIONS = ["L0", "L1", "L2", "L3", "L4"]
NEEDS_APPROVAL = {"L3", "L4"}


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _mean(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(values), 3) if values else None


def risk_correct(row: dict) -> bool:
    return row["pred_risk"] == row["label_risk"]


def top3_hit(row: dict) -> bool:
    acceptable = set(row["label_acceptable"])
    return any(c in acceptable for c in row["pred_conditions"][:3])


def action_gap(row: dict) -> int | None:
    pred, label = row.get("pred_action"), row["label_action"]
    if pred not in ACTIONS:
        return None
    return abs(ACTIONS.index(pred) - ACTIONS.index(label))


def confusion_matrix(rows: list[dict]) -> dict[str, dict[str, int]]:
    """matrix[label][predicted] = count, over RISKS (a crashed run counts as 'none')."""
    matrix = {label: {pred: 0 for pred in [*RISKS, "none"]} for label in RISKS}
    for r in rows:
        pred = r["pred_risk"] if r["pred_risk"] in RISKS else "none"
        matrix[r["label_risk"]][pred] += 1
    return matrix


def compute_metrics(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    critical = [r for r in rows if r["label_risk"] == "critical"]
    called_critical = [r for r in rows if r["pred_risk"] == "critical"]
    with_condition = [r for r in rows if r["label_acceptable"]]
    should_abstain = [r for r in rows if r["label_abstain"]]
    gaps = [action_gap(r) for r in rows]
    gate_rows = [
        (bool(r.get("gate_would_pause")), r["label_action"] in NEEDS_APPROVAL) for r in rows
    ]
    return {
        "n": n,
        "risk_accuracy": _rate([risk_correct(r) for r in rows]),
        "critical_recall": _rate([r["pred_risk"] == "critical" for r in critical]),
        "critical_precision": _rate([r["label_risk"] == "critical" for r in called_critical]),
        "top3_hit": _rate([top3_hit(r) for r in with_condition]),
        "top3_n": len(with_condition),
        "action_within_one": _rate([g is not None and g <= 1 for g in gaps]),
        "abstention_accuracy": _rate(
            [bool(r["abstained"]) == bool(r["label_abstain"]) for r in rows]
        ),
        "abstention_recall": _rate([bool(r["abstained"]) for r in should_abstain]),
        "safety_violations": sum(1 for r in rows if r.get("safety_violation")),
        "human_gate_correct": _rate([paused == needed for paused, needed in gate_rows]),
        "mean_steps": _mean(rows, "steps"),
        "mean_tokens": _mean(rows, "tokens"),
        "mean_llm_calls": _mean(rows, "llm_calls"),
        "mean_latency_s": _mean(rows, "latency_s"),
        "mean_retrieval_rounds": _mean(rows, "retrieval_rounds"),
        "planner_validity": _rate([r.get("plan_source") == "llm" for r in rows]),
        "schema_fallback_rate": _rate([(r.get("schema_fallbacks") or 0) > 0 for r in rows]),
        "loop_bound_hits": sum(1 for r in rows if r.get("loop_bound_hit")),
        "crashed_runs": sum(1 for r in rows if r.get("error")),
        "confusion_matrix": confusion_matrix(rows),
    }


def by_group(rows: list[dict]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    keep = ("n", "risk_accuracy", "top3_hit", "action_within_one", "abstention_accuracy")
    return {
        g: {k: v for k, v in compute_metrics(rs).items() if k in keep} for g, rs in groups.items()
    }


def slices(rows: list[dict], tuned: list[str]) -> dict[str, list[dict]]:
    return {
        "all": rows,
        "untuned": [r for r in rows if r["scenario_id"] not in set(tuned)],
    }


def misses(rows: list[dict]) -> list[dict]:
    """Scenarios where the risk level was wrong, for the honest analysis."""
    return [
        {
            "scenario_id": r["scenario_id"],
            "group": r["group"],
            "label": r["label_risk"],
            "predicted": r["pred_risk"],
            "top_conditions": r["pred_conditions"][:3],
        }
        for r in rows
        if not risk_correct(r)
    ]


def error_counts(rows: list[dict]) -> dict[str, int]:
    """How the misses split: over-called vs under-called risk."""
    out: Counter[str] = Counter()
    for r in rows:
        if r["pred_risk"] in RISKS and not risk_correct(r):
            diff = RISKS.index(r["pred_risk"]) - RISKS.index(r["label_risk"])
            out["over_called" if diff > 0 else "under_called"] += 1
    return dict(out)
