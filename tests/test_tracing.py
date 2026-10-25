"""Tracing: JSONL is always written, tool calls are recorded, tracing never breaks a run.

All tests run in mock mode with Phoenix off (or deliberately unreachable).
"""

import json

import pytest

from pondguard.config import load_settings
from pondguard.tracing import RunTracer, current_tracer, to_jsonable, traced_tool


@pytest.fixture
def settings():
    return load_settings(env={})  # mock provider, TRACING=jsonl


def _records(tracer: RunTracer) -> list[dict]:
    return [json.loads(line) for line in tracer.path.read_text(encoding="utf-8").splitlines()]


@traced_tool
def add_flag(pond_id: str, do_mg_l: float, limit: float = 4.0) -> dict:
    return {"pond_id": pond_id, "flag": do_mg_l < limit}


@traced_tool
def broken_tool(x: int) -> int:
    raise ValueError(f"bad input {x}")


def test_tool_outside_a_run_is_not_traced():
    assert current_tracer() is None
    assert add_flag("P-1", 3.0) == {"pond_id": "P-1", "flag": True}


def test_tool_call_is_recorded_with_inputs_outputs_and_duration(settings, tmp_path):
    with RunTracer("r1", settings=settings, runs_dir=tmp_path, tags={"pond_id": "P-07"}) as t:
        with t.span("monitor", kind="node"):
            assert add_flag("P-07", do_mg_l=3.2) == {"pond_id": "P-07", "flag": True}
    records = _records(t)
    assert [r["type"] for r in records] == ["run_start", "span", "span", "span", "run_end"]
    tool = next(r for r in records if r.get("kind") == "tool")
    assert tool["name"] == "add_flag"
    assert tool["inputs"] == {"pond_id": "P-07", "do_mg_l": 3.2}
    assert tool["outputs"] == {"pond_id": "P-07", "flag": True}
    assert tool["status"] == "ok" and tool["duration_ms"] >= 0
    assert all(r["run_id"] == "r1" for r in records)
    assert not t.phoenix_active


def test_every_span_carries_agent_and_judge_identity(settings, tmp_path):
    with RunTracer("r2", settings=settings, runs_dir=tmp_path) as t:
        add_flag("P-1", 5.0)
    start = _records(t)[0]
    assert start["models"]["agent"]["model_key"] == "mock-agent"
    assert start["models"]["judge"]["model_key"] == "mock-judge"
    for span in (r for r in _records(t) if r["type"] == "span"):
        assert set(span["models"]) == {"agent", "judge"}


def test_tool_errors_are_recorded_and_re_raised(settings, tmp_path):
    with pytest.raises(ValueError, match="bad input 7"):
        with RunTracer("r3", settings=settings, runs_dir=tmp_path) as t:
            broken_tool(7)
    records = _records(t)
    tool = next(r for r in records if r.get("kind") == "tool")
    assert tool["status"] == "error" and "bad input 7" in tool["error"]
    assert records[-1] == {**records[-1], "type": "run_end", "status": "error"}


def test_events_feed_the_graph_state_log(settings, tmp_path):
    with RunTracer("r4", settings=settings, runs_dir=tmp_path) as t:
        entry = t.event("monitor", "2 anomalies", count=2)
    assert entry == {"node": "monitor", "message": "2 anomalies", "count": 2}
    assert t.events == [entry]
    assert any(r["type"] == "event" and r["count"] == 2 for r in _records(t))


def test_phoenix_down_falls_back_to_jsonl(tmp_path):
    settings = load_settings(
        env={"TRACING": "phoenix", "PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:9"}
    )
    with RunTracer("r5", settings=settings, runs_dir=tmp_path) as t:
        assert add_flag("P-1", 3.0)["flag"]
    assert not t.phoenix_active
    assert _records(t)[0]["phoenix"] is False
    assert any(r.get("kind") == "tool" for r in _records(t))


def test_unwritable_trace_file_never_breaks_the_run(settings, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")  # runs_dir is a file, so every JSONL write fails
    with RunTracer("r6", settings=settings, runs_dir=blocker) as t:
        assert add_flag("P-1", 3.0)["flag"]  # the tool still works
        t.event("monitor", "still running")
    assert t.events  # in-memory event log still kept


def test_large_values_are_truncated():
    big = {"text": "x" * 5000, "items": list(range(120))}
    out = to_jsonable(big)
    assert len(out["text"]) <= 2001
    assert len(out["items"]) == 51 and out["items"][-1] == "… 70 more"
