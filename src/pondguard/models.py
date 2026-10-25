"""Data contracts shared by every part of PondGuard (spec §6.4).

Agentic concept: input/output validation (guardrails). Every tool input, agent
output and final report is a Pydantic model, so bad sensor values, uncited
hypotheses or malformed LLM output are rejected at the boundary instead of
flowing silently through the graph.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- Shared vocabularies ---------------------------------------------------

Symptom = Literal[
    "white_spots",
    "red_discoloration",
    "lethargy",
    "surfacing_at_dawn",
    "gathering_at_edges",
    "reduced_feeding",
    "white_faecal_strings",
    "empty_gut",
    "pale_hepatopancreas",
    "slow_growth",
    "size_variation",
    "loose_shell",
    "luminescence",
    "sudden_mortality",
]
Condition = Literal[
    "wssv", "ehp", "white_faeces", "ahpnd", "vibriosis", "water_quality_stress", "unknown"
]
RiskLevel = Literal["normal", "watch", "alert", "critical"]
ActionLevel = Literal["L0", "L1", "L2", "L3", "L4"]

RISK_LEVELS: tuple[RiskLevel, ...] = ("normal", "watch", "alert", "critical")
ACTION_LEVELS: tuple[ActionLevel, ...] = ("L0", "L1", "L2", "L3", "L4")

DISCLAIMER = "Advisory only. Confirm with a qualified aquaculture professional or lab test."


class _Strict(BaseModel):
    """Base for all contracts: unknown fields are an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


# --- Inputs ------------------------------------------------------------------


class Reading(_Strict):
    """One hourly sensor reading. Ranges reject physically impossible values."""

    pond_id: str = Field(min_length=1)
    ts: datetime
    do_mg_l: float = Field(ge=0, le=20)
    ph: float = Field(ge=4, le=11)
    temp_c: float = Field(ge=10, le=40)
    salinity_ppt: float = Field(ge=0, le=60)
    tan_mg_l: float = Field(ge=0, le=20)
    nitrite_mg_l: float = Field(ge=0, le=20)
    alkalinity_mg_l: float = Field(ge=0, le=400)


class FarmerLog(_Strict):
    """The farmer's daily observation, including check-tray feeding."""

    pond_id: str = Field(min_length=1)
    date: Date
    day_of_culture: int = Field(ge=0, le=150)
    feed_consumed_pct: float = Field(ge=0, le=100)
    mortality_count: int = Field(default=0, ge=0)
    symptoms: list[Symptom] = Field(default_factory=list)
    notes: str = ""


# --- Intermediate results ----------------------------------------------------


class Anomaly(_Strict):
    parameter: str
    kind: Literal["below_limit", "above_limit", "trend", "spike", "drop"]
    severity: Literal["low", "medium", "high"]
    value: float
    limit: float | None = None
    window: str
    limit_source: str | None = None


class Passage(_Strict):
    """A retrieved knowledge-base chunk; `grade` is filled in by grade_passages."""

    id: str = Field(min_length=1)
    text: str
    source: str
    title: str
    disease: str | None = None
    section: str | None = None
    page: int | None = None
    score: float
    grade: float | None = Field(default=None, ge=0, le=1)


class CauseHypothesis(_Strict):
    """A candidate cause. Grounding rule: it must cite at least one passage."""

    condition: Condition
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    citations: list[str] = Field(min_length=1)


class Diagnosis(_Strict):
    """Up to three ranked hypotheses, or an abstention with one follow-up question."""

    hypotheses: list[CauseHypothesis] = Field(default_factory=list, max_length=3)
    abstained: bool
    follow_up_question: str | None = None
    retrieval_rounds: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> Diagnosis:
        if self.abstained and not self.follow_up_question:
            raise ValueError("an abstained diagnosis must ask a follow-up question")
        if not self.abstained and not self.hypotheses:
            raise ValueError("a diagnosis that does not abstain needs at least one hypothesis")
        self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return self


class ActionPlan(_Strict):
    risk_level: RiskLevel
    action_level: ActionLevel
    steps: list[str] = Field(min_length=1)
    rationale: str
    citations: list[str] = Field(default_factory=list)


class CriticVerdict(_Strict):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    revise_target: Literal["advisor", "diagnostician", "none"] = "none"

    @model_validator(mode="after")
    def _check(self) -> CriticVerdict:
        if self.passed and self.revise_target != "none":
            raise ValueError("a passing verdict cannot ask for a revision")
        if not self.passed and not self.issues:
            raise ValueError("a failing verdict must list at least one issue")
        return self


# --- Final output ------------------------------------------------------------


class SourceRef(_Strict):
    id: str
    title: str
    source: str
    page: int | None = None


class PondReport(_Strict):
    """The report shown to the farmer and saved to pond memory."""

    run_id: str
    pond_id: str
    date: Date
    variant: Literal["A", "B"]
    risk_level: RiskLevel
    hypotheses: list[CauseHypothesis] = Field(default_factory=list, max_length=3)
    action: ActionPlan
    sources: list[SourceRef] = Field(default_factory=list)
    follow_up_question: str | None = None
    needs_human: bool
    approval_status: Literal["not_needed", "pending", "approved", "edited", "rejected"]
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def _check(self) -> PondReport:
        if self.risk_level != self.action.risk_level:
            raise ValueError("report risk level must match the action plan's risk level")
        if not self.disclaimer.strip():
            raise ValueError("the advisory disclaimer is mandatory")
        return self
