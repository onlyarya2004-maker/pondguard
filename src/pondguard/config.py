"""Typed configuration: the four YAML files in config/ plus environment settings.

Agentic concept: bounded loops and guardrails as data. Loop limits, retrieval
thresholds, water-quality limits, the action ladder and the banned-substance list
all live in version-controlled YAML, validated here, so every agent reads the same
limits and a reviewer can see them without reading code.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pondguard.models import (
    ACTION_LEVELS,
    RISK_LEVELS,
    ActionLevel,
    Condition,
    RiskLevel,
    Symptom,
)
from pondguard.paths import CONFIG_DIR, REPO_ROOT

Provider = Literal["mock", "lmstudio", "ollama"]
FaultInjection = Literal["none", "uncited_step"]
APPROVAL_LEVELS: tuple[ActionLevel, ...] = ("L3", "L4")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- settings.yaml -----------------------------------------------------------


class LoopBounds(_Strict):
    max_supervisor_steps: int = Field(ge=1)
    max_requeries: int = Field(ge=0)
    max_critic_revisions: int = Field(ge=0)
    recursion_limit: int = Field(ge=1)
    tool_timeout_s: int = Field(ge=1)
    token_cap_per_run: int = Field(ge=1)


class RetrievalSettings(_Strict):
    k: int = Field(ge=1)
    grade_threshold: float = Field(ge=0, le=1)
    min_relevant_passages: int = Field(ge=1)


class TelemetrySettings(_Strict):
    window_hours: int = Field(ge=1)


class LLMSettings(_Strict):
    temperature: float = Field(ge=0, le=2)
    request_timeout_s: int = Field(ge=1)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_s: float = Field(default=1.0, ge=0)
    schema_repair_attempts: int = Field(default=1, ge=0, le=1)
    cache: bool = True


# --- thresholds.yaml ---------------------------------------------------------


class ThresholdSpec(_Strict):
    min: float | None = None
    max: float | None = None
    critical_min: float | None = None
    max_drop_24h: float | None = None
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ThresholdSpec:
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError("min must be below max")
        if self.critical_min is not None and self.min is not None and self.critical_min > self.min:
            raise ValueError("critical_min must not be above min")
        return self

    @property
    def verified(self) -> bool:
        return not self.source.startswith("TODO")


# --- action_ladder.yaml ------------------------------------------------------


class LevelSpec(_Strict):
    name: str
    meaning: str
    requires_approval: bool


class EscalationRule(_Strict):
    repeat_risk: RiskLevel
    consecutive_days: int = Field(ge=2)
    max_levels_up: int = Field(ge=0, le=1)


class ActionLadder(_Strict):
    levels: dict[ActionLevel, LevelSpec]
    allowed_actions: dict[RiskLevel, list[ActionLevel]]
    condition_default_risk: dict[Condition, RiskLevel]
    escalation: EscalationRule

    @model_validator(mode="after")
    def _check(self) -> ActionLadder:
        if set(self.levels) != set(ACTION_LEVELS):
            raise ValueError(f"levels must define exactly {ACTION_LEVELS}")
        if set(self.allowed_actions) != set(RISK_LEVELS):
            raise ValueError(f"allowed_actions must cover exactly {RISK_LEVELS}")
        if any(not acts for acts in self.allowed_actions.values()):
            raise ValueError("every risk level needs at least one allowed action")
        # Safety invariant: containment and emergency harvest always need a human.
        for level in APPROVAL_LEVELS:
            if not self.levels[level].requires_approval:
                raise ValueError(f"{level} must require human approval")
        return self

    def requires_approval(self, level: ActionLevel) -> bool:
        return self.levels[level].requires_approval


# --- banned_substances.yaml --------------------------------------------------


class BannedSubstance(_Strict):
    name: str = Field(min_length=1)
    synonyms: list[str] = Field(default_factory=list)
    verified: bool = False


class BannedSubstances(_Strict):
    source: str
    substances: list[BannedSubstance] = Field(min_length=1)

    def all_names(self) -> list[str]:
        """Every name and synonym, lower-cased, for the guardrail regex."""
        return sorted({n.lower() for s in self.substances for n in (s.name, *s.synonyms)})


# --- sign_sets.yaml ----------------------------------------------------------


class QuestionEntry(_Strict):
    sign: Symptom
    question: str = Field(min_length=1)


class SignSets(_Strict):
    sign_to_conditions: dict[Symptom, list[Condition]]
    anomaly_to_conditions: dict[str, list[Condition]]
    condition_terms: dict[Condition, str]
    water_quality_symptoms: list[Symptom] = Field(default_factory=list)
    question_bank: list[QuestionEntry]
    generic_question: str

    @model_validator(mode="after")
    def _check(self) -> SignSets:
        for sign, conditions in self.sign_to_conditions.items():
            if not conditions:
                raise ValueError(f"{sign}: list at least one condition, or omit the sign")
        mapped = {c for cs in self.sign_to_conditions.values() for c in cs}
        mapped |= {c for cs in self.anomaly_to_conditions.values() for c in cs}
        missing = mapped - set(self.condition_terms)
        if missing:
            raise ValueError(f"condition_terms is missing {sorted(missing)}")
        return self

    def is_specific(self, sign: str) -> bool:
        """A sign is specific if it maps to at least one condition."""
        return bool(self.sign_to_conditions.get(sign))  # type: ignore[call-overload]

    def conditions_for_signs(self, signs: list[str]) -> list[str]:
        """Conditions pointed to by the signs, most-supported first."""
        votes: dict[str, int] = {}
        for sign in signs:
            for condition in self.sign_to_conditions.get(sign, []):  # type: ignore[call-overload]
                votes[condition] = votes.get(condition, 0) + 1
        return sorted(votes, key=lambda c: -votes[c])

    def conditions_for_parameter(self, parameter: str) -> list[str]:
        return list(self.anomaly_to_conditions.get(parameter, []))


# --- environment (.env) ------------------------------------------------------


class RuntimeEnv(_Strict):
    provider: Provider = "mock"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_agent_model: str | None = None
    lmstudio_judge_model: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_agent_model: str | None = None
    ollama_judge_model: str | None = None
    tracing: Literal["phoenix", "jsonl"] = "jsonl"
    phoenix_collector_endpoint: str = "http://localhost:6006"
    variant: Literal["A", "B"] = "B"
    data_dir: Path = REPO_ROOT / "data"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> RuntimeEnv:
        def get(name: str) -> str | None:
            value = env.get(name, "").strip()
            return value or None

        raw = {
            "provider": get("LLM_PROVIDER"),
            "lmstudio_base_url": get("LMSTUDIO_BASE_URL"),
            "lmstudio_agent_model": get("LMSTUDIO_AGENT_MODEL"),
            "lmstudio_judge_model": get("LMSTUDIO_JUDGE_MODEL"),
            "ollama_base_url": get("OLLAMA_BASE_URL"),
            "ollama_agent_model": get("OLLAMA_AGENT_MODEL"),
            "ollama_judge_model": get("OLLAMA_JUDGE_MODEL"),
            "tracing": get("TRACING"),
            "phoenix_collector_endpoint": get("PHOENIX_COLLECTOR_ENDPOINT"),
            "variant": get("PONDGUARD_VARIANT"),
            "data_dir": get("DATA_DIR"),
        }
        values = {k: v for k, v in raw.items() if v is not None}
        if "data_dir" in values:
            path = Path(values["data_dir"])
            values["data_dir"] = path if path.is_absolute() else (REPO_ROOT / path).resolve()
        return cls(**values)


# --- everything together -----------------------------------------------------


class Settings(_Strict):
    loop_bounds: LoopBounds
    retrieval: RetrievalSettings
    telemetry: TelemetrySettings
    llm: LLMSettings
    fault_injection: FaultInjection = "none"
    thresholds: dict[str, ThresholdSpec]
    action_ladder: ActionLadder
    banned: BannedSubstances
    sign_sets: SignSets
    runtime: RuntimeEnv


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a mapping at the top level")
    return data


def load_settings(config_dir: Path = CONFIG_DIR, env: Mapping[str, str] | None = None) -> Settings:
    """Read and validate all config files. `env` defaults to os.environ."""
    env = os.environ if env is None else env
    base = _read_yaml(config_dir / "settings.yaml")
    fault = env.get("PONDGUARD_INJECT_FAULT", "").strip()
    if fault:
        base["fault_injection"] = fault
    return Settings(
        **base,
        thresholds=_read_yaml(config_dir / "thresholds.yaml"),
        action_ladder=_read_yaml(config_dir / "action_ladder.yaml"),
        banned=_read_yaml(config_dir / "banned_substances.yaml"),
        sign_sets=_read_yaml(config_dir / "sign_sets.yaml"),
        runtime=RuntimeEnv.from_env(env),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, loaded once. Tests call load_settings() directly."""
    return load_settings()
