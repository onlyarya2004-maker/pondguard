"""Long-term memory and the approval queue (SQLite, data/pondguard.db).

Agentic concepts: persistent memory across runs, and a durable, resumable workflow.

Tables (spec §6.12):
- pond_history: one row per finished run (risk, top condition, action, approval status),
  read back by the Advisor to spot an alert that repeats day after day;
- runs: statistics for every run (steps, retrieval rounds, tokens, latency, bound hit);
- approvals: every human decision on a paused L3/L4 (or unresolved) draft.
- pending_approvals: paused threads waiting for a person (used by `pondguard pending`).

The same file also holds the LangGraph checkpoints (see `open_checkpointer`), so a paused
run survives a restart of the program. The LLM response cache is a different file
(data/llm_cache.db), so clearing the cache never touches pond history.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pondguard.paths import DATA_DIR

DB_PATH = DATA_DIR / "pondguard.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS pond_history (
    id INTEGER PRIMARY KEY, pond_id TEXT, date TEXT, run_id TEXT, risk_level TEXT,
    top_condition TEXT, action_level TEXT, approval_status TEXT, outcome_note TEXT,
    created_at TEXT);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, pond_id TEXT, scenario_id TEXT, variant TEXT, started_at TEXT,
    ended_at TEXT, steps INTEGER, retrieval_rounds INTEGER, tokens INTEGER, latency_ms INTEGER,
    loop_bound_hit TEXT, trace_url TEXT);
CREATE TABLE IF NOT EXISTS approvals (
    run_id TEXT, decision TEXT, edited_action TEXT, reviewer TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS pending_approvals (
    thread_id TEXT PRIMARY KEY, run_id TEXT, pond_id TEXT, scenario_id TEXT, risk_level TEXT,
    action_level TEXT, created_at TEXT, status TEXT);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Memory:
    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()  # Windows keeps files locked while a connection is open

    # --- pond history ---------------------------------------------------------------------

    def add_history(
        self,
        *,
        pond_id: str,
        day: str,
        run_id: str,
        risk_level: str,
        top_condition: str | None,
        action_level: str,
        approval_status: str,
        outcome_note: str = "",
    ) -> None:
        with self._db() as db:
            db.execute(
                "DELETE FROM pond_history WHERE pond_id = ? AND run_id = ?", (pond_id, run_id)
            )  # a resumed run writes its final row once
            db.execute(
                "INSERT INTO pond_history (pond_id, date, run_id, risk_level, top_condition,"
                " action_level, approval_status, outcome_note, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    pond_id,
                    day,
                    run_id,
                    risk_level,
                    top_condition,
                    action_level,
                    approval_status,
                    outcome_note,
                    _now(),
                ),
            )

    def get_pond_history(
        self, pond_id: str, days: int = 14, today: str | None = None
    ) -> list[dict]:
        """Rows for the last `days` days before `today` (default: today), newest first."""
        end = date.fromisoformat(today) if today else date.today()
        start = (end - timedelta(days=days)).isoformat()
        with self._db() as db:
            rows = db.execute(
                "SELECT date, run_id, risk_level, top_condition, action_level, approval_status"
                " FROM pond_history WHERE pond_id = ? AND date >= ? AND date < ?"
                " ORDER BY date DESC, id DESC",
                (pond_id, start, end.isoformat()),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- runs and approvals -----------------------------------------------------------------

    def add_run(self, **row: Any) -> None:
        cols = (
            "run_id",
            "pond_id",
            "scenario_id",
            "variant",
            "started_at",
            "ended_at",
            "steps",
            "retrieval_rounds",
            "tokens",
            "latency_ms",
            "loop_bound_hit",
            "trace_url",
        )
        with self._db() as db:
            marks = ", ".join("?" * len(cols))
            db.execute(
                f"INSERT OR REPLACE INTO runs ({', '.join(cols)}) VALUES ({marks})",
                [row.get(c) for c in cols],
            )

    def add_approval(self, run_id: str, decision: str, edited_action: str, reviewer: str) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO approvals (run_id, decision, edited_action, reviewer, ts)"
                " VALUES (?,?,?,?,?)",
                (run_id, decision, edited_action, reviewer, _now()),
            )

    def approvals_for(self, run_id: str) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM approvals WHERE run_id = ?", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    # --- pending approvals --------------------------------------------------------------------

    def add_pending(self, thread_id: str, **info: Any) -> None:
        """Idempotent: a resumed node re-runs from its start."""
        with self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO pending_approvals (thread_id, run_id, pond_id, scenario_id,"
                " risk_level, action_level, created_at, status) VALUES (?,?,?,?,?,?,?,'pending')",
                (
                    thread_id,
                    info.get("run_id"),
                    info.get("pond_id"),
                    info.get("scenario_id"),
                    info.get("risk_level"),
                    info.get("action_level"),
                    _now(),
                ),
            )

    def resolve_pending(self, thread_id: str, status: str = "resolved") -> None:
        with self._db() as db:
            db.execute(
                "UPDATE pending_approvals SET status = ? WHERE thread_id = ?", (status, thread_id)
            )

    def list_pending(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM pending_approvals WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]


def open_checkpointer(path: Path = DB_PATH):
    """LangGraph's SQLite checkpointer on the same database file (durable, resumable runs)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    saver.setup()
    return saver


def thread_id_for(pond_id: str, day: str, run_id: str) -> str:
    return f"{pond_id}:{day}:{run_id}"
