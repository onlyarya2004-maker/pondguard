"""The judge's claim handling, the A/B decision rule and the figures, on synthetic results."""

import csv
import json

import pytest

from pondguard.config import load_settings
from pondguard.evaluation import experiment, judge, plots
from pondguard.evaluation import runner as runner_module
from pondguard.llm import TokenBudget, get_llm

SETTINGS = load_settings(env={})

REPORT = {
    "hypotheses": [{"condition": "ehp", "confidence": 0.9, "citations": ["K1"]}],
    "action": {
        "steps": [
            "Send a sample for PCR/lab testing [ladder:L2].",
            "Monitor growth and white gut [K1].",
            "Increase night aeration.",  # no citation at all
        ]
    },
}


# --- judge ------------------------------------------------------------------------------


def test_claims_come_from_hypotheses_and_cited_steps_but_not_ladder_wording():
    claims = judge.claims_from_report("S1", REPORT, {"K1"})
    assert [(c.kind, c.passage_id) for c in claims] == [
        ("hypothesis", "K1"), ("step", "K1"), ("step", None),
    ]  # fmt: skip
    assert "EHP" in claims[0].text and claims[1].text == "Monitor growth and white gut."


def test_a_claim_without_a_valid_citation_is_unsupported_without_asking_the_model():
    llm = get_llm("judge", SETTINGS, TokenBudget(cap=100_000))
    claims = judge.claims_from_report("S1", REPORT, {"K1"})
    results = judge.judge_claims(claims, {"K1": "EHP passage text"}, llm)
    assert [r["supported"] for r in results] == [True, True, False]  # the mock says supported
    assert [r["judged"] for r in results] == [True, True, False]
    assert len(llm.calls) == 2  # the uncited claim cost no model call
    assert judge.faithfulness(results) == pytest.approx(0.6667, abs=1e-4)
    assert judge.faithfulness([]) is None


def test_a_citation_to_an_unknown_passage_is_unsupported():
    claims = judge.claims_from_report("S1", REPORT, set())  # nothing was retrievable
    assert all(c.passage_id is None for c in claims)


def test_reports_are_read_from_the_run_traces(tmp_path):
    for sid in ("EHP-03", "WQ-01"):
        path = tmp_path / f"evalB-{sid}-20261030_223213.jsonl"
        path.write_text(
            json.dumps({"type": "run_start"})
            + "\n"
            + json.dumps({"type": "report", "report": REPORT})
            + "\n",
            encoding="utf-8",
        )
    (tmp_path / "evalA-EHP-03-20261030_223213.jsonl").write_text(
        "{}", encoding="utf-8"
    )  # other variant
    got = judge.load_reports("B", "20261030_223213", tmp_path)
    assert sorted(got) == ["EHP-03", "WQ-01"]


def _judged_claims(n):
    return [{"scenario_id": f"S{i:02}", "kind": "step", "text": f"claim {i}", "passage_id": f"P{i}",
             "supported": i % 2 == 0, "reason": "", "judged": True} for i in range(n)]  # fmt: skip


def test_the_validation_sheet_has_ten_claims_half_and_half_and_the_key_is_separate(
    tmp_path, monkeypatch
):
    class Fake:
        def texts(self, ids):
            return {i: f"text of {i}" for i in ids}

    monkeypatch.setattr(judge, "get_retriever", lambda settings: Fake())
    monkeypatch.setattr(judge, "SHEET_DIR", tmp_path / "private")
    path = tmp_path / "judge_B.json"
    path.write_text(json.dumps({"claims": _judged_claims(40)}), encoding="utf-8")
    sheet = judge.build_sheet(path, SETTINGS)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    assert len(rows) == 10 and all(r["human_supported"] == "" for r in rows)
    assert "supported" not in rows[0] and "judge" not in " ".join(
        rows[0]
    )  # the verdicts are not on the sheet
    key = json.loads((tmp_path / "private" / "judge_verdicts_key.json").read_text())
    assert sum(key.values()) == 5

    with pytest.raises(ValueError, match="yes or no"):
        judge.validate_judge(tmp_path / "out.json")
    for r, verdict in zip(rows, key.values(), strict=True):
        r["human_supported"] = "yes" if verdict else "no"
    rows[0]["human_supported"] = "no" if key[rows[0]["claim_id"]] else "yes"  # one disagreement
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=judge.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result = judge.validate_judge(tmp_path / "out.json")
    assert result["agreement"] == 0.9 and len(result["disagreements"]) == 1


# --- experiment -----------------------------------------------------------------------------


def _row(sid, label_cond="ehp", pred=("ehp",), tokens=5000, **kw):
    base = dict(
        scenario_id=sid, group="ehp", variant="B", label_risk="alert", label_action="L2",
        label_top=label_cond, label_acceptable=[label_cond] if label_cond else [],
        label_abstain=False, pred_risk="alert", pred_action="L2", pred_conditions=list(pred),
        abstained=False, approval_status="not_needed", safety_violation=False, critic_passed=True,
        gate_would_pause=False, steps=5, tokens=tokens, llm_calls=5, latency_s=20.0,
        retrieval_rounds=1, revisions=0, loop_bound_hit="", plan_source="llm", schema_fallbacks=0,
        wall_s=20.0, error="",
    )  # fmt: skip
    return {**base, **kw}


def _write_run(folder, variant, rows, model="llama"):
    stamp = f"2026103{'0' if variant == 'A' else '1'}_000000"
    csv_path = folder / f"eval_{variant}_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runner_module.RUN_FIELDS)
        writer.writeheader()
        for r in rows:
            runner_module.write_row(writer, r)
    summary = runner_module.summarise(rows, variant, SETTINGS, stamp, 1.0)
    summary["models"]["agent"]["model_key"] = model
    path = folder / f"eval_{variant}_{stamp}_summary.json"
    path.write_text(json.dumps(summary, default=str), encoding="utf-8")
    return csv_path, path


def _ten(good_top3, tokens):
    rows = []
    for i in range(10):
        hit = i < good_top3
        rows.append(_row(f"X{i}", pred=("ehp",) if hit else ("vibriosis",), tokens=tokens))
    return rows


def test_adopt_b_when_top3_rises_5_points_at_no_more_than_twice_the_cost(tmp_path):
    a = _write_run(tmp_path, "A", _ten(6, 4000))
    b = _write_run(tmp_path, "B", _ten(8, 6000))  # +20 points at 1.5x the tokens
    result = experiment.run_experiment(tmp_path, a, b)
    assert result["decision"]["adopt"] == "B" and result["decision"]["cost_ratio"] == 1.5
    text = (tmp_path / "ab_decision.md").read_text(encoding="utf-8")
    assert "Decision: adopt B" in text and "fixed A on 2" in text
    assert {r["metric"] for r in csv.DictReader((tmp_path / "ab_summary.csv").open())} >= {
        "top3_hit"
    }


def test_keep_a_when_b_is_too_expensive_or_not_better(tmp_path):
    a = _write_run(tmp_path, "A", _ten(6, 4000))
    pricey = _write_run(tmp_path, "B", _ten(9, 9000))  # better, but 2.25x the cost
    assert experiment.run_experiment(tmp_path, a, pricey)["decision"]["adopt"] == "A"
    same = _write_run(tmp_path, "B", _ten(6, 5000))  # no gain
    assert experiment.run_experiment(tmp_path, a, same)["decision"]["adopt"] == "A"


def test_the_paired_comparison_names_the_scenarios_b_fixed_and_broke(tmp_path):
    rows_a = [_row("S1", pred=("x",)), _row("S2"), _row("S3", pred=("x",)), _row("S4")]
    rows_b = [_row("S1"), _row("S2", pred=("x",)), _row("S3", pred=("x",)), _row("S4")]
    pairs = experiment.paired(rows_a, rows_b)["top3_hit"]
    assert pairs["b_fixed_a"] == ["S1"] and pairs["b_broke_a"] == ["S2"]
    assert (pairs["both_right"], pairs["both_wrong"]) == (1, 1)


def test_runs_from_different_models_are_not_compared(tmp_path):
    a = _write_run(tmp_path, "A", _ten(5, 4000), model="llama")
    b = _write_run(tmp_path, "B", _ten(8, 4000), model="qwen")
    with pytest.raises(experiment.IncomparableRuns, match="same model"):
        experiment.run_experiment(tmp_path, a, b)


def test_figures_are_written_from_the_saved_results(tmp_path):
    _write_run(tmp_path, "A", _ten(6, 4000))
    _write_run(tmp_path, "B", _ten(8, 6000))
    paths = plots.make_figures(tmp_path)
    assert [p.name for p in paths] == [
        "confusion_A.png", "confusion_B.png", "metrics_ab.png", "cost_quality.png"
    ]  # fmt: skip
    assert all(p.exists() and p.stat().st_size > 5000 for p in paths)
