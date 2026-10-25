"""Data contracts reject bad input and enforce grounding rules."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from pondguard.models import (
    DISCLAIMER,
    ActionPlan,
    CauseHypothesis,
    CriticVerdict,
    Diagnosis,
    FarmerLog,
    PondReport,
    Reading,
)

GOOD_READING = dict(
    pond_id="P-07",
    ts=datetime(2026, 10, 20, 5, 0),
    do_mg_l=4.2,
    ph=7.9,
    temp_c=28.5,
    salinity_ppt=18,
    tan_mg_l=0.4,
    nitrite_mg_l=0.1,
    alkalinity_mg_l=120,
)


def _hyp(condition="white_faeces", confidence=0.7, citations=("kb-1",)):
    return CauseHypothesis(
        condition=condition,
        confidence=confidence,
        evidence=["feed drop"],
        citations=list(citations),
    )


def _plan(risk="alert", action="L2"):
    return ActionPlan(
        risk_level=risk, action_level=action, steps=["Send a sample for PCR."], rationale="r"
    )


def test_valid_reading():
    assert Reading(**GOOD_READING).do_mg_l == 4.2


@pytest.mark.parametrize(
    "field,value",
    [("do_mg_l", -1), ("do_mg_l", 25), ("ph", 3.5), ("temp_c", 45), ("salinity_ppt", 70)],
)
def test_impossible_sensor_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Reading(**{**GOOD_READING, field: value})


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        Reading(**GOOD_READING, turbidity=3)


def test_farmer_log_validates_symptoms_and_ranges():
    log = FarmerLog(
        pond_id="P-07",
        date=date(2026, 10, 20),
        day_of_culture=42,
        feed_consumed_pct=68,
        symptoms=["white_faecal_strings", "reduced_feeding"],
    )
    assert log.mortality_count == 0
    with pytest.raises(ValidationError):
        FarmerLog(**{**log.model_dump(), "symptoms": ["purple_tail"]})
    with pytest.raises(ValidationError):
        FarmerLog(**{**log.model_dump(), "feed_consumed_pct": 120})


def test_hypothesis_without_citations_is_rejected():
    with pytest.raises(ValidationError):
        _hyp(citations=())


def test_diagnosis_sorts_by_confidence_and_caps_at_three():
    d = Diagnosis(
        hypotheses=[_hyp("ehp", 0.4), _hyp("white_faeces", 0.8), _hyp("vibriosis", 0.6)],
        abstained=False,
        retrieval_rounds=2,
    )
    assert [h.condition for h in d.hypotheses] == ["white_faeces", "vibriosis", "ehp"]
    with pytest.raises(ValidationError):
        Diagnosis(hypotheses=[_hyp()] * 4, abstained=False, retrieval_rounds=1)


def test_abstention_needs_a_follow_up_question():
    with pytest.raises(ValidationError, match="follow-up"):
        Diagnosis(abstained=True, retrieval_rounds=3)
    d = Diagnosis(
        abstained=True,
        follow_up_question="Do you see white spots on the shell?",
        retrieval_rounds=3,
    )
    assert d.hypotheses == []


def test_non_abstaining_diagnosis_needs_a_hypothesis():
    with pytest.raises(ValidationError):
        Diagnosis(abstained=False, retrieval_rounds=1)


def test_critic_verdict_consistency():
    assert CriticVerdict(passed=True).revise_target == "none"
    with pytest.raises(ValidationError):
        CriticVerdict(passed=False)
    with pytest.raises(ValidationError):
        CriticVerdict(passed=True, revise_target="advisor")


def test_report_carries_disclaimer_and_matching_risk():
    report = PondReport(
        run_id="r1",
        pond_id="P-07",
        date=date(2026, 10, 20),
        variant="B",
        risk_level="alert",
        hypotheses=[_hyp()],
        action=_plan(),
        needs_human=False,
        approval_status="not_needed",
    )
    assert report.disclaimer == DISCLAIMER
    with pytest.raises(ValidationError, match="risk level"):
        PondReport(**{**report.model_dump(), "risk_level": "watch"})
    with pytest.raises(ValidationError, match="disclaimer"):
        PondReport(**{**report.model_dump(), "disclaimer": " "})


def test_action_plan_needs_at_least_one_step():
    with pytest.raises(ValidationError):
        ActionPlan(risk_level="normal", action_level="L0", steps=[], rationale="r")
