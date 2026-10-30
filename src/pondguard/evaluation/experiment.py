"""The A/B experiment: single-pass RAG (Variant A) versus agentic RAG (Variant B).

Same model, same prompts, same 40 scenarios, same temperature; only the retrieval strategy
differs. Outputs:
    results/ab_summary.csv    slice, metric, A, B, delta  (for all 40 and the untuned scenarios)
    results/ab_decision.md    the decision rule applied to the numbers, and the paired comparison

Decision rule (spec §5.4): adopt B if its top-3 hit rate or its citation faithfulness rises by at
least 5 points AND its cost is at most twice A's (cost = mean tokens per run, taken from the
original model calls). Otherwise keep A and report it. The paired comparison lists the scenarios
where B fixed an answer A got wrong and where B broke one A got right, which matters more than
the averages with only 40 cases. The script refuses to compare runs from different models.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pondguard.evaluation import metrics
from pondguard.evaluation.runner import eval_config, read_rows
from pondguard.paths import RESULTS_DIR

COMPARED = [
    "risk_accuracy", "critical_recall", "top3_hit", "action_within_one", "abstention_accuracy",
    "abstention_recall", "safety_violations", "human_gate_correct", "mean_steps", "mean_tokens",
    "mean_llm_calls", "mean_latency_s", "mean_retrieval_rounds", "planner_validity",
    "schema_fallback_rate",
]  # fmt: skip
POINTS = 0.05  # "at least 5 points"
MAX_COST_RATIO = 2.0


class IncomparableRuns(ValueError):
    """The two runs used different models, so an A/B comparison would not be fair."""


def latest(variant: str, results_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    """Newest (csv, summary) pair for a variant."""
    summaries = sorted(results_dir.glob(f"eval_{variant}_*_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no evaluation run for variant {variant}; run `pondguard eval`")
    summary = summaries[-1]
    return summary.with_name(summary.name.replace("_summary.json", ".csv")), summary


def model_identity(summary: dict) -> tuple:
    m = summary["models"]["agent"]
    return (m["provider"], m["model_key"], m.get("quantization"), m.get("runtime"))


def check_comparable(a: dict, b: dict) -> None:
    if model_identity(a) != model_identity(b):
        raise IncomparableRuns(
            f"variant A used {model_identity(a)} but variant B used {model_identity(b)}; "
            "re-run both variants on the same model"
        )
    if a["temperature"] != b["temperature"]:
        raise IncomparableRuns("the two runs used different temperatures")


def _faithfulness(variant: str, stamp: str, results_dir: Path) -> float | None:
    path = results_dir / f"judge_{variant}_{stamp}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("faithfulness")


def paired(rows_a: list[dict], rows_b: list[dict]) -> dict[str, Any]:
    """Per-scenario comparison on three yes/no outcomes."""
    by_a = {r["scenario_id"]: r for r in rows_a}
    out: dict[str, Any] = {}
    outcomes = {
        "risk_correct": metrics.risk_correct,
        "top3_hit": metrics.top3_hit,
        "abstention_correct": lambda r: bool(r["abstained"]) == bool(r["label_abstain"]),
    }
    for name, fn in outcomes.items():
        both = neither = 0
        fixed: list[str] = []
        broke: list[str] = []
        for rb in rows_b:
            ra = by_a.get(rb["scenario_id"])
            if ra is None or (name == "top3_hit" and not rb["label_acceptable"]):
                continue
            a_ok, b_ok = fn(ra), fn(rb)
            if a_ok and b_ok:
                both += 1
            elif not a_ok and not b_ok:
                neither += 1
            elif b_ok:
                fixed.append(rb["scenario_id"])
            else:
                broke.append(rb["scenario_id"])
        out[name] = {
            "both_right": both,
            "both_wrong": neither,
            "b_fixed_a": fixed,
            "b_broke_a": broke,
        }
    return out


def decide(a: dict, b: dict, faith_a: float | None, faith_b: float | None) -> dict[str, Any]:
    """Apply the decision rule to the 'all 40' metrics."""
    top3_gain = (b["top3_hit"] or 0) - (a["top3_hit"] or 0)
    faith_gain = None if faith_a is None or faith_b is None else faith_b - faith_a
    cost_ratio = (b["mean_tokens"] / a["mean_tokens"]) if a["mean_tokens"] else None
    quality_up = top3_gain >= POINTS or (faith_gain is not None and faith_gain >= POINTS)
    affordable = cost_ratio is not None and cost_ratio <= MAX_COST_RATIO
    return {
        "adopt": "B" if quality_up and affordable else "A",
        "top3_gain_points": round(top3_gain * 100, 1),
        "faithfulness_gain_points": None if faith_gain is None else round(faith_gain * 100, 1),
        "cost_ratio": None if cost_ratio is None else round(cost_ratio, 2),
        "quality_rule_met": quality_up,
        "cost_rule_met": affordable,
    }


def run_experiment(results_dir: Path = RESULTS_DIR, a_files=None, b_files=None) -> dict[str, Any]:
    a_csv, a_sum = a_files or latest("A", results_dir)
    b_csv, b_sum = b_files or latest("B", results_dir)
    sa = json.loads(Path(a_sum).read_text(encoding="utf-8"))
    sb = json.loads(Path(b_sum).read_text(encoding="utf-8"))
    check_comparable(sa, sb)
    rows_a, rows_b = read_rows(a_csv), read_rows(b_csv)
    faith_a = _faithfulness("A", sa["started"], results_dir)
    faith_b = _faithfulness("B", sb["started"], results_dir)

    table = []
    for part in ("all", "untuned"):
        ma, mb = sa["metrics"][part], sb["metrics"][part]
        for key in COMPARED:
            va, vb = ma.get(key), mb.get(key)
            delta = None if va is None or vb is None else round(vb - va, 4)
            table.append({"slice": part, "metric": key, "A": va, "B": vb, "delta": delta})
    faith_delta = None if None in (faith_a, faith_b) else round(faith_b - faith_a, 4)
    table.append({"slice": "all", "metric": "citation_faithfulness (SLM judge)",
                  "A": faith_a, "B": faith_b, "delta": faith_delta})  # fmt: skip

    with (results_dir / "ab_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slice", "metric", "A", "B", "delta"])
        writer.writeheader()
        writer.writerows(table)

    decision = decide(sa["metrics"]["all"], sb["metrics"]["all"], faith_a, faith_b)
    decision_untuned = decide(sa["metrics"]["untuned"], sb["metrics"]["untuned"], faith_a, faith_b)
    pairs = paired(rows_a, rows_b)
    result = {
        "a_run": sa["started"], "b_run": sb["started"], "model": model_identity(sa),
        "decision": decision, "decision_untuned_only": decision_untuned, "paired": pairs,
        "a_commit": sa["git"]["commit"], "b_commit": sb["git"]["commit"],
    }  # fmt: skip
    (results_dir / "ab_decision.md").write_text(render_decision(result, sa, sb), encoding="utf-8")
    return result


def render_decision(result: dict, sa: dict, sb: dict) -> str:
    d, du = result["decision"], result["decision_untuned_only"]
    ma, mb = sa["metrics"]["all"], sb["metrics"]["all"]
    gain = d["faithfulness_gain_points"]
    lines = [
        "# A/B decision: single-pass RAG (A) vs agentic RAG (B)",
        "",
        f"Model: {result['model'][1]} ({result['model'][2]}), runtime {result['model'][3]}. "
        f"Run A `{result['a_run']}` (commit {result['a_commit'][:7]}), "
        f"run B `{result['b_run']}` (commit {result['b_commit'][:7]}).",
        "",
        "## Decision rule",
        "Adopt B if its top-3 hit rate or citation faithfulness rises by at least 5 points and "
        "its cost (mean tokens per run) is at most 2x A's. Otherwise keep A.",
        "",
        "## Result on all 40 scenarios",
        f"- Top-3 hit: A {ma['top3_hit']}, B {mb['top3_hit']} ({d['top3_gain_points']:+} points)",
        f"- Faithfulness gain: {'not measured' if gain is None else f'{gain:+} points'}",
        f"- Cost ratio (B tokens / A tokens): {d['cost_ratio']}x "
        f"(A {ma['mean_tokens']}, B {mb['mean_tokens']})",
        f"- Quality rule met: {d['quality_rule_met']}. Cost rule met: {d['cost_rule_met']}.",
        f"- **Decision: adopt {d['adopt']}.**",
        "",
        "## Same rule on the untuned scenarios only",
        f"- Top-3 gain {du['top3_gain_points']:+} points, cost ratio {du['cost_ratio']}x, "
        f"decision: adopt {du['adopt']}.",
        "",
        "## Paired comparison (same scenarios under both variants)",
    ]
    for name, p in result["paired"].items():
        lines.append(
            f"- {name}: both right {p['both_right']}, both wrong {p['both_wrong']}, "
            f"B fixed A on {len(p['b_fixed_a'])} ({', '.join(p['b_fixed_a']) or 'none'}), "
            f"B broke A on {len(p['b_broke_a'])} ({', '.join(p['b_broke_a']) or 'none'})"
        )
    lines += [
        "",
        "## Caveats",
        "- Only 40 scenarios (about 30 with a labelled condition): one scenario is 2.5 to 3.3 "
        "points, so a 5-point difference is one or two scenarios. Read the paired lists, not "
        "just the averages.",
        "- The scenarios are simulated, and prompts and rules were tuned on EHP-03, AMBIG-01 and "
        "HEALTHY-01 (see docs/tuning_log.md), which is why the untuned slice is reported too.",
        "- One run per variant at temperature 0; run-to-run variation was not measured.",
        "",
    ]
    return "\n".join(lines)


def _targets_note(summary: dict) -> dict[str, bool | None]:
    """Which Review 2 targets were met (for the report)."""
    cfg = eval_config()["targets"]
    m = summary["metrics"]["all"]
    return {k: (None if m.get(k) is None else m[k] >= v) for k, v in cfg.items() if k in m}
