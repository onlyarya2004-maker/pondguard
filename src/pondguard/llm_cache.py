"""SQLite response cache for LLM calls (plan Phase 2c).

Agentic concept: reproducibility and cost accounting. A cached answer makes a
re-run identical and lets an interrupted evaluation resume without redoing
finished calls. Each entry stores the ORIGINAL call's tokens and latency, and a
cache hit reports those stored values, so cost metrics never look free (plan
rule 9).

The key covers provider, model, task, the full prompt, the temperature and a
`repeat` index. Stability repeats use a new repeat index (or no cache), so they
make fresh calls instead of replaying the first answer. Only valid outputs are
cached; schema fallbacks never are.

Lives in data/llm_cache.db, separate from pond memory and checkpoints
(data/pondguard.db), so clearing the cache never touches pond history.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def cache_key(
    *, provider: str, model: str, task: str, prompt: Any, temperature: float, repeat: int
) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "task": task,
        "prompt": _jsonable(prompt),
        "temperature": temperature,
        "repeat": repeat,
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedCall:
    output_json: str
    input_tokens: int
    output_tokens: int
    estimated: bool
    latency_s: float
    attempts: int
    repaired: bool


class LLMCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    provider TEXT, model TEXT, task TEXT,
                    output_json TEXT NOT NULL,
                    input_tokens INTEGER, output_tokens INTEGER, estimated INTEGER,
                    latency_s REAL, attempts INTEGER, repaired INTEGER,
                    created_at TEXT
                )"""
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open, commit and always close (Windows keeps files locked while open)."""
        db = sqlite3.connect(self.path)
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def get(self, key: str) -> CachedCall | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT output_json, input_tokens, output_tokens, estimated, latency_s,"
                " attempts, repaired FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CachedCall(row[0], row[1], row[2], bool(row[3]), row[4], row[5], bool(row[6]))

    def put(self, key: str, *, provider: str, model: str, task: str, call: CachedCall) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    provider,
                    model,
                    task,
                    call.output_json,
                    call.input_tokens,
                    call.output_tokens,
                    int(call.estimated),
                    call.latency_s,
                    call.attempts,
                    int(call.repaired),
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )

    def __len__(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
