"""Run the golden set for one variant and save per-scenario rows plus a summary.

Evaluation mode: temperature 0 (config), the human gate auto-approves but records that it WOULD
have paused, pond memory is off (each scenario is independent), and the LLM response cache is on
so an interrupted run can resume without redoing finished calls. Cost and latency come from the
ORIGINAL model calls stored with each cache entry, never from a cache hit.

Outputs (results/, committed):
    eval_<variant>_<timestamp>.csv           one row per scenario
    eval_<variant>_<timestamp>_summary.json  metrics (all and untuned), commit, model identity

Metrics are computed from the CSV rows, never typed in.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pondguard.config import Settings, load_settings
from pondguard.evaluation import metrics
from pondguard.graph import run_scenario
from pondguard.llm import get_llm
from pondguard.paths import CONFIG_DIR, DATA_DIR, REPO_ROOT, RESULTS_DIR
from pondguard.simulator.scenarios import simulate
from pondguard.tools.guardrail_tools import banned_pattern, find_dosage, strip_citations
from pondguard.tools.kb_tools import get_retriever
from pondguard.tracing import run_models

LIST_FIELDS = ("pred_conditions", "label_acceptable")
RUN_FIELDS = [
    "scenario_id", "group", "variant",
    "label_risk", "label_action", "label_top", "label_acceptable", "label_abstain",
    "pred_risk", "pred_action", "pred_conditions", "abstained", "approval_status",
    "safety_violation", "critic_passed", "gate_would_pause",
    "steps", "tokens", "llm_calls", "latency_s", "retrieval_rounds", "revisions",
    "loop_bound_hit", "plan_source", "schema_fallbacks", "wall_s", "error",
]  # fmt: skip


def eval_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "eval.yaml").read_text(encoding="utf-8"))


def load_golden(scenarios_dir: Path | None = None) -> list[dict]:
    folder = scenarios_dir or DATA_DIR / "scenarios"
    found = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
    return [s for s in found if s.get("golden", True)]


def ensure_scenarios() -> None:
    """Regenerate the (deterministic) scenario files and telemetry if any are missing."""
    have = {p.stem for p in (DATA_DIR / "scenarios").glob("*.json")}
    csvs = {p.stem for p in (DATA_DIR / "telemetry").glob("*.csv")}
    if not have or have - csvs:
        simulate(CONFIG_DIR / "scenario_specs.yaml", DATA_DIR / "scenarios")


def safety_violation(report: dict, settings: Settings) -> bool:
    """A banned substance anywhere in the report, or a dosage in any action step."""
    names = settings.banned.all_names()
    text = json.dumps(
        {k: report.get(k) for k in ("hypotheses", "action", "follow_up_question")}, default=str
    )
    if names and banned_pattern(names).search(text):
        return True
    return any(find_dosage(strip_citations(s)) for s in report["action"]["steps"])


def to_row(scenario: dict, variant: str, final: dict, llm, settings: Settings, wall: float) -> dict:
    report = final["report"]
    label = scenario["label"]
    diagnosis = final.get("diagnosis") or {}
    verdict = final.get("critic_verdict") or {}
    return {
        "scenario_id": scenario["scenario_id"],
        "group": scenario["group"],
        "variant": variant,
        "label_risk": label["risk_level"],
        "label_action": label["action_level"],
        "label_top": label["top_condition"],
        "label_acceptable": list(label["acceptable_conditions"]),
        "label_abstain": bool(label["should_abstain"]),
        "pred_risk": report["risk_level"],
        "pred_action": report["action"]["action_level"],
        "pred_conditions": [h["condition"] for h in report["hypotheses"]][:3],
        "abstained": bool(diagnosis.get("abstained")),
        "approval_status": report["approval_status"],
        "safety_violation": safety_violation(report, settings),
        "critic_passed": bool(verdict.get("passed")),
        "gate_would_pause": bool(final.get("gate_would_pause")),
        "steps": final.get("step_count"),
        "tokens": final.get("tokens_used"),
        "llm_calls": llm.budget.calls,
        "latency_s": round(sum(c.latency_s for c in llm.calls), 3),
        "retrieval_rounds": diagnosis.get("retrieval_rounds", 0),
        "revisions": final.get("revisions", 0),
        "loop_bound_hit": final.get("loop_bound_hit") or "",
        "plan_source": final.get("plan_source"),
        "schema_fallbacks": sum(1 for c in llm.calls if c.fallback),
        "wall_s": round(wall, 1),
        "error": "",
    }


def failed_row(scenario: dict, variant: str, error: Exception, wall: float) -> dict:
    """A run that crashed still counts: it is a wrong answer, not a missing one."""
    label = scenario["label"]
    return {
        **{k: None for k in RUN_FIELDS},
        "scenario_id": scenario["scenario_id"],
        "group": scenario["group"],
        "variant": variant,
        "label_risk": label["risk_level"],
        "label_action": label["action_level"],
        "label_top": label["top_condition"],
        "label_acceptable": list(label["acceptable_conditions"]),
        "label_abstain": bool(label["should_abstain"]),
        "pred_conditions": [],
        "abstained": False,
        "safety_violation": False,
        "wall_s": round(wall, 1),
        "error": f"{type(error).__name__}: {error}"[:200],
    }


def write_row(writer: csv.DictWriter, row: dict) -> None:
    out = {k: ("|".join(row[k]) if k in LIST_FIELDS else row[k]) for k in RUN_FIELDS}
    writer.writerow(out)


def read_rows(path: Path) -> list[dict]:
    """Load a results CSV back into the row dicts that the metrics expect."""
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = dict(raw)
            for k in LIST_FIELDS:
                row[k] = [x for x in raw[k].split("|") if x]
            for k in ("label_abstain", "abstained", "safety_violation", "critic_passed",
                      "gate_would_pause"):  # fmt: skip
                row[k] = raw[k] == "True"
            for k in ("steps", "tokens", "llm_calls", "retrieval_rounds", "revisions",
                      "schema_fallbacks"):  # fmt: skip
                row[k] = int(float(raw[k])) if raw[k] not in ("", None) else None
            row["latency_s"] = float(raw["latency_s"]) if raw["latency_s"] else None
            row["label_top"] = raw["label_top"] or None
            row["pred_risk"] = raw["pred_risk"] or None
            rows.append(row)
    return rows


def git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return out.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(git("status", "--porcelain")),
    }


def summarise(
    rows: list[dict], variant: str, settings: Settings, started: str, wall: float
) -> dict:
    cfg = eval_config()
    parts = metrics.slices(rows, cfg["tuned_scenarios"])
    return {
        "variant": variant,
        "started": started,
        "wall_clock_s": round(wall, 1),
        "git": git_info(),
        "provider": settings.runtime.provider,
        "models": run_models(settings),
        "temperature": settings.llm.temperature,
        "tuned_scenarios": cfg["tuned_scenarios"],
        "targets": cfg["targets"],
        "metrics": {name: metrics.compute_metrics(part) for name, part in parts.items()},
        "by_group": metrics.by_group(rows),
        "risk_misses": metrics.misses(rows),
        "miss_direction": metrics.error_counts(rows),
    }


def run_eval(
    variant: str,
    *,
    only: list[str] | None = None,
    use_cache: bool = True,
    out_dir: Path | None = None,
    echo=print,
) -> tuple[Path, Path]:
    """Run the golden set (or `only` some of it) for one variant. Returns (csv, summary) paths."""
    ensure_scenarios()
    settings = load_settings()
    retriever = get_retriever(settings)
    scenarios = [s for s in load_golden() if not only or s["scenario_id"] in only]
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"eval_{variant}_{stamp}.csv"
    rows: list[dict] = []
    t_start = time.perf_counter()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for i, scenario in enumerate(scenarios, start=1):
            llm = get_llm("agent", settings, use_cache=use_cache)
            t0 = time.perf_counter()
            try:
                final, _ = run_scenario(
                    scenario, llm, settings, retriever, variant=variant, auto_approve=True,
                    run_id=f"eval{variant}-{scenario['scenario_id']}-{stamp}",
                )  # fmt: skip
                row = to_row(scenario, variant, final, llm, settings, time.perf_counter() - t0)
            except Exception as exc:  # a crash is a result: record it and carry on
                row = failed_row(scenario, variant, exc, time.perf_counter() - t0)
            rows.append(row)
            write_row(writer, row)
            f.flush()
            mark = "ok " if metrics.risk_correct(row) else "MISS"
            echo(
                f"[{i:2}/{len(scenarios)}] {row['scenario_id']:14} {mark} "
                f"label {row['label_risk']:8} "
                f"pred {str(row['pred_risk']):8} action {row['pred_action']} "
                f"{row['wall_s']}s {row['error']}"
            )
    summary = summarise(rows, variant, settings, stamp, time.perf_counter() - t_start)
    summary_path = out_dir / f"eval_{variant}_{stamp}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return csv_path, summary_path
