"""Evaluation metrics on small fixed inputs, and the runner end to end in mock mode."""

import json

import pytest

from pondguard.evaluation import metrics
from pondguard.evaluation import runner as runner_module
from pondguard.evaluation.runner import read_rows, run_eval

BASE = dict(
    group="ehp", label_risk="alert", label_action="L2", label_acceptable=["ehp"],
    label_abstain=False,
    pred_risk="alert", pred_action="L2", pred_conditions=["ehp"], abstained=False,
    safety_violation=False, gate_would_pause=False, steps=5, tokens=5000, llm_calls=5,
    latency_s=30.0, retrieval_rounds=1, plan_source="llm", schema_fallbacks=0, loop_bound_hit="",
    error="",
)  # fmt: skip


def row(sid="S", **changes):
    return {**BASE, "scenario_id": sid, **changes}


def test_risk_accuracy_and_the_confusion_matrix():
    rows = [
        row("a"),
        row("b", pred_risk="watch"),
        row("c", label_risk="critical", pred_risk="critical"),
        row("d", pred_risk=None, error="boom"),
    ]
    m = metrics.compute_metrics(rows)
    assert m["risk_accuracy"] == 0.5  # a and c are right; b and the crashed run are wrong
    cm = m["confusion_matrix"]
    assert cm["alert"] == {"normal": 0, "watch": 1, "alert": 1, "critical": 0, "none": 1}
    assert cm["critical"]["critical"] == 1 and m["crashed_runs"] == 1


def test_critical_recall_and_precision():
    rows = [
        row("a", label_risk="critical", pred_risk="critical"),
        row("b", label_risk="critical", pred_risk="alert"),
        row("c", label_risk="alert", pred_risk="critical"),
    ]
    m = metrics.compute_metrics(rows)
    assert m["critical_recall"] == 0.5 and m["critical_precision"] == 0.5


def test_top3_hit_counts_only_scenarios_with_a_labelled_condition():
    rows = [
        row("a", pred_conditions=["x", "y", "ehp"]),
        row("b", pred_conditions=["x", "y", "z", "ehp"]),  # fourth place: not in the top three
        row("c", label_acceptable=[], pred_conditions=["ehp"], label_risk="normal"),
    ]
    m = metrics.compute_metrics(rows)
    assert m["top3_hit"] == 0.5 and m["top3_n"] == 2


def test_action_within_one_level():
    rows = [
        row("a", pred_action="L1"),
        row("b", pred_action="L3"),
        row("c", pred_action="L0"),
        row("d", pred_action=None),
    ]
    assert (
        metrics.compute_metrics(rows)["action_within_one"] == 0.5
    )  # L1 and L3 are within one of L2


def test_abstention_accuracy_and_recall():
    rows = [
        row("a", label_abstain=True, abstained=True),
        row("b", label_abstain=True, abstained=False),
        row("c", abstained=True),
        row("d"),
    ]
    m = metrics.compute_metrics(rows)
    assert m["abstention_accuracy"] == 0.5 and m["abstention_recall"] == 0.5


def test_human_gate_must_pause_exactly_when_the_label_needs_approval():
    rows = [
        row("a", label_action="L3", gate_would_pause=True),
        row("b", label_action="L3", gate_would_pause=False),
        row("c", gate_would_pause=True),
        row("d"),
    ]
    assert metrics.compute_metrics(rows)["human_gate_correct"] == 0.5


def test_safety_efficiency_and_reliability_counts():
    rows = [
        row("a", safety_violation=True),
        row("b", plan_source="default", loop_bound_hit="token_cap"),
        row("c", schema_fallbacks=2, tokens=7000),
    ]
    m = metrics.compute_metrics(rows)
    assert m["safety_violations"] == 1 and m["loop_bound_hits"] == 1
    assert m["planner_validity"] == pytest.approx(0.6667, abs=1e-4)
    assert m["schema_fallback_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert m["mean_tokens"] == pytest.approx(5666.667, abs=0.01)


def test_untuned_slice_excludes_the_tuned_scenarios():
    rows = [row("EHP-03"), row("X"), row("Y", pred_risk="watch")]
    parts = metrics.slices(rows, ["EHP-03"])
    assert len(parts["all"]) == 3 and [r["scenario_id"] for r in parts["untuned"]] == ["X", "Y"]
    assert metrics.error_counts(rows) == {"under_called": 1}
    assert [m["scenario_id"] for m in metrics.misses(rows)] == ["Y"]


def test_empty_input_does_not_crash():
    assert metrics.compute_metrics([]) == {"n": 0}


def test_the_runner_writes_rows_and_a_summary_in_mock_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "get_retriever", lambda settings: _FakeRetriever())
    csv_path, summary_path = run_eval(
        "B", only=["HEALTHY-01", "AMBIG-01"], out_dir=tmp_path, echo=lambda *_: None
    )
    rows = read_rows(csv_path)
    assert [r["scenario_id"] for r in rows] == ["AMBIG-01", "HEALTHY-01"]
    healthy = next(r for r in rows if r["scenario_id"] == "HEALTHY-01")
    assert (
        healthy["pred_risk"] == "normal" and healthy["pred_action"] == "L0" and not healthy["error"]
    )
    summary = json.loads(summary_path.read_text())
    assert summary["variant"] == "B" and summary["metrics"]["all"]["n"] == 2
    assert summary["models"]["agent"]["model_key"] == "mock-agent"
    assert summary["metrics"]["untuned"]["n"] == 0  # both scenarios are in the tuned list


class _FakeRetriever:
    def search(self, query, k=5, disease=None):
        from pondguard.models import Passage

        return [
            Passage(
                id="D:p1:c0",
                text="white faecal strands",
                source="D",
                title="Doc",
                page=1,
                score=0.9,
            )
        ]
