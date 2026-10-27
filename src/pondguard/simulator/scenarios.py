"""Scenario builder: recipes in config/scenario_specs.yaml -> JSON + telemetry CSV.

Agentic concept: evaluation data with ground truth. Each scenario JSON holds the
telemetry location, the farmer log, pond history, the injected events and the
LABEL, written from the recipe before any agent exists. The CSV is regenerable
(data/telemetry/ is gitignored); the JSON is committed (data/scenarios/).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from pondguard.models import FarmerLog
from pondguard.paths import DATA_DIR, REPO_ROOT
from pondguard.simulator.telemetry import apply_events, baseline

MORNING_CHECK_HOUR = 7  # telemetry ends at 07:00 on the evaluation date


def load_specs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["scenarios"]


def build_scenario(spec: dict, telemetry_dir: Path) -> dict:
    """Generate one scenario: write its CSV and return the JSON-ready dict."""
    end = pd.Timestamp(spec["date"]) + pd.Timedelta(hours=MORNING_CHECK_HOUR)
    df = baseline(spec["pond_id"], end, spec["day_of_culture"], spec["seed"])
    df = apply_events(df, spec.get("events", []))
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    csv_path = telemetry_dir / f"{spec['scenario_id']}.csv"
    numeric = df.columns.difference(["pond_id", "ts"])
    df.assign(**{c: df[c].round(4) for c in numeric}).to_csv(csv_path, index=False)

    log = FarmerLog(
        pond_id=spec["pond_id"],
        date=spec["date"],
        day_of_culture=spec["day_of_culture"],
        **spec["farmer_log"],
    )
    try:
        shown_path = csv_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        shown_path = csv_path.as_posix()
    return {
        "scenario_id": spec["scenario_id"],
        "group": spec["group"],
        "pond_id": spec["pond_id"],
        "date": spec["date"],
        "day_of_culture": spec["day_of_culture"],
        "seed": spec["seed"],
        "telemetry_csv": shown_path,
        "farmer_logs": [log.model_dump(mode="json")],
        "history": spec.get("history", []),
        "injected_events": spec.get("events", []),
        "label": spec["label"],
    }


def simulate(specs_path: Path, out_dir: Path, telemetry_dir: Path | None = None) -> list[Path]:
    """Build every scenario in `specs_path`; return the JSON paths written."""
    telemetry_dir = telemetry_dir or DATA_DIR / "telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in load_specs(specs_path):
        scenario = build_scenario(spec, telemetry_dir)
        path = out_dir / f"{scenario['scenario_id']}.json"
        path.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def load_scenario(scenario_id: str, scenarios_dir: Path | None = None) -> dict:
    path = (scenarios_dir or DATA_DIR / "scenarios") / f"{scenario_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))
