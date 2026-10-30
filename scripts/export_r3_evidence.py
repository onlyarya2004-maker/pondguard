"""Collect Review 3 evidence into docs/r3/ without a person at the screen.

Copies one finished real-model JSONL trace per scenario, exports the matching Phoenix traces from
its SQLite database (read-only, Phoenix need not be running), and writes a short step summary per
scenario. The raw traces contain knowledge-base passage text (copyrighted manuals), so they go to
docs/r3/traces/ and docs/r3/phoenix/, which are gitignored. The step summaries contain no passage
text and are committed.

Run from pondguard/:  .venv\\Scripts\\python scripts\\export_r3_evidence.py
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "r3"
RUNS = ROOT / "results" / "runs"
PHOENIX_DB = ROOT.parent / "cache" / "phoenix" / "phoenix.db"

# One finished real-model run per scenario (chosen after reading every trace's run_start/report).
JSONL = {
    "HEALTHY-01": "HEALTHY-01-9158f6",
    "EHP-03": "EHP-03-c870e2",
    "AMBIG-01": "AMBIG-01-2c5a55",
    "CRITIC-DEMO-01": "CRITIC-DEMO-01-b4f62c",
    "WSSV-01": "WSSV-01-5236c0",
}


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def summary(scenario: str, run_id: str, records: list[dict]) -> str:
    start = records[0]
    report = next((r for r in records if r["type"] == "report"), None)
    lines = [f"# {scenario}  (run {run_id})", ""]
    models = (start.get("models") or {}).get("agent", {})
    lines.append(
        f"Model: {models.get('model_key')} ({models.get('quantization')}), "
        f"provider {models.get('provider')}"
    )
    lines.append(
        f"Spans: {sum(r['type'] == 'span' for r in records)}, "
        f"events: {sum(r['type'] == 'event' for r in records)}"
    )
    lines += ["", "## Steps as the agents reported them", ""]
    for r in records:
        if r["type"] == "event":
            lines.append(f"- [{r.get('node')}] {r.get('message')}")
    tools = [r["name"] for r in records if r["type"] == "span" and r.get("kind") == "tool"]
    lines += ["", f"Tool calls (in order): {', '.join(tools) if tools else 'none'}", ""]
    if report:
        rep = report["report"]
        action = rep["action"]
        lines += ["## Final report", ""]
        lines.append(
            f"- Risk: {rep['risk_level']}; action {action['action_level']}; "
            f"approval {rep.get('approval_status')}"
        )
        for h in rep["hypotheses"]:
            lines.append(
                f"- Hypothesis: {h['condition']} ({h['confidence']:.2f}), "
                f"cites {', '.join(h['citations'])}"
            )
        if rep.get("follow_up_question"):
            lines.append(f"- Follow-up question: {rep['follow_up_question']}")
        for step in action["steps"]:
            lines.append(f"- Step: {step}")
        lines.append(f"- Run stats: {json.dumps(report.get('stats'), default=str)}")
    return "\n".join(lines) + "\n"


def export_phoenix() -> list[str]:
    if not PHOENIX_DB.exists():
        return []
    db = sqlite3.connect(f"file:{PHOENIX_DB.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    folder = OUT / "phoenix"
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for trace in db.execute("select id, trace_id, start_time, end_time from traces order by id"):
        spans = [
            {
                **dict(s),
                "attributes": json.loads(s["attributes"]),
                "events": json.loads(s["events"] or "[]"),
            }
            for s in db.execute(
                "select * from spans where trace_rowid=? order by start_time", (trace["id"],)
            )
        ]
        blob = " ".join(json.dumps(s["attributes"]) for s in spans)
        found = re.findall(r'"run_id": ?"([^"]+)"', blob)
        run_id = found[0] if found else f"trace{trace['id']}"
        if len(spans) < 20 and not run_id.startswith("trace-demo"):
            continue  # an interrupted run: not useful as evidence
        name = f"phoenix_{run_id}.json"
        payload = {
            "trace_id": trace["trace_id"],
            "start": trace["start_time"],
            "end": trace["end_time"],
            "spans": spans,
        }
        (folder / name).write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        names.append(name)
    return names


def main() -> None:
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    for scenario, run_id in JSONL.items():
        src = RUNS / f"{run_id}.jsonl"
        shutil.copy(src, OUT / "traces" / src.name)
        text = summary(scenario, run_id, read(src))
        (OUT / f"steps_{scenario}.md").write_text(text, encoding="utf-8")
        print("wrote", f"steps_{scenario}.md")
    print("phoenix:", export_phoenix())


if __name__ == "__main__":
    main()
