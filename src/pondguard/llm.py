"""The LLM layer: one interface for every model call, local SLMs only.

Agentic concept: structured tool-like reasoning steps with bounded cost. Every
LLM call is one named *task* (plan, grade, rank, ...) with a Pydantic output
schema, so the graph only ever receives validated objects. Every call is charged
to a per-run token budget; once the cap is reached the next call is refused,
which bounds cost even if a loop misbehaves.

Two roles, two models (plan rule 10):
- `agent` runs the agents' tasks;
- `judge` runs the evaluation tasks and must be a different model.

Providers:
- `mock` (default; tests and CI): fixed, schema-valid responses, no model, offline.
- `lmstudio`: LM Studio's OpenAI-compatible server via langchain-openai, with
  structured output through the `json_schema` response format.
- `ollama`: optional fallback for teammates without LM Studio.

Not here yet (Phase 2c): response cache, retry with backoff, schema repair and
fallback, model-identity capture. Until then an invalid model output raises
LLMOutputError.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pondguard.config import Provider, Settings, get_settings
from pondguard.models import CauseHypothesis, Condition

log = logging.getLogger("pondguard.llm")

Role = Literal["agent", "judge"]
TaskType = Literal[
    "plan", "summarise", "grade", "rewrite", "rank", "advise", "extract_claims", "check_claim"
]
JUDGE_TASKS: frozenset[TaskType] = frozenset({"extract_claims", "check_claim"})
PlanStep = Literal["monitor", "diagnose", "advise", "critique"]
DEFAULT_PLAN: list[PlanStep] = ["monitor", "diagnose", "advise", "critique"]


# --- Output schema for each task ---------------------------------------------


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SupervisorPlan(_Out):
    plan: list[PlanStep] = Field(min_length=1, max_length=8)
    reasoning: str


class TelemetrySummary(_Out):
    summary: str


class PassageGrade(_Out):
    passage_id: str
    grade: float = Field(ge=0, le=1)
    reason: str


class PassageGrades(_Out):
    grades: list[PassageGrade]


class RewrittenQuery(_Out):
    query: str = Field(min_length=1)
    disease_filter: Condition | None = None


class RankedHypotheses(_Out):
    hypotheses: list[CauseHypothesis] = Field(max_length=3)


class AdviceStep(_Out):
    text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class AdviceDraft(_Out):
    steps: list[AdviceStep] = Field(min_length=1)
    rationale: str


class Claim(_Out):
    text: str = Field(min_length=1)
    citation: str | None = None


class ClaimList(_Out):
    claims: list[Claim]


class ClaimVerdict(_Out):
    supported: bool
    reason: str


TASK_SCHEMAS: dict[TaskType, type[_Out]] = {
    "plan": SupervisorPlan,
    "summarise": TelemetrySummary,
    "grade": PassageGrades,
    "rewrite": RewrittenQuery,
    "rank": RankedHypotheses,
    "advise": AdviceDraft,
    "extract_claims": ClaimList,
    "check_claim": ClaimVerdict,
}


# --- Errors ------------------------------------------------------------------


class LLMConfigError(ValueError):
    """The provider or model settings are unusable (e.g. judge == agent)."""


class LLMOutputError(RuntimeError):
    """The model returned output that does not match the task schema."""


class TokenCapExceeded(RuntimeError):
    """The run's token budget is spent; the graph must stop and report it."""


# --- Token accounting --------------------------------------------------------


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class TokenBudget:
    """Per-run token cap (config: loop_bounds.token_cap_per_run)."""

    cap: int
    used: int = 0
    calls: int = 0
    hit: bool = False

    @property
    def remaining(self) -> int:
        return max(self.cap - self.used, 0)

    def check(self, task: str) -> None:
        if self.used >= self.cap:
            self.hit = True
            log.warning("token cap reached (%d/%d); refusing task %s", self.used, self.cap, task)
            raise TokenCapExceeded(f"token cap of {self.cap} reached before task '{task}'")

    def charge(self, usage: TokenUsage) -> None:
        self.used += usage.total
        self.calls += 1


def estimate_tokens(text: str) -> int:
    """Rough count (about 4 characters per token) for mock mode."""
    return max(1, len(text) // 4)


# --- Mock model --------------------------------------------------------------


def mock_response(task: TaskType, context: dict[str, Any]) -> _Out:
    """Fixed, schema-valid output for each task. No reasoning happens here.

    `context` lets callers pass ids the response must reference (passage ids,
    the query to rewrite). The scripted grader used for the EHP-03 demo
    replaces the `grade` branch in Phase 6.
    """
    passage_ids: list[str] = list(context.get("passage_ids", []))
    if task == "plan":
        return SupervisorPlan(plan=list(DEFAULT_PLAN), reasoning="Mock: default plan.")
    if task == "summarise":
        return TelemetrySummary(summary="Mock summary: no model was called.")
    if task == "grade":
        return PassageGrades(
            grades=[
                PassageGrade(passage_id=p, grade=0.5, reason="Mock grade.") for p in passage_ids
            ]
        )
    if task == "rewrite":
        query = str(context.get("query", "")).strip() or "shrimp pond disease signs"
        return RewrittenQuery(query=f"{query} (rewritten)")
    if task == "rank":
        if not passage_ids:
            return RankedHypotheses(hypotheses=[])
        hypothesis = CauseHypothesis(
            condition=context.get("condition", "unknown"),
            confidence=0.5,
            evidence=["mock evidence"],
            citations=[passage_ids[0]],
        )
        return RankedHypotheses(hypotheses=[hypothesis])
    if task == "advise":
        cites = passage_ids[:1]
        step = AdviceStep(text="Mock step: continue routine monitoring.", citations=cites)
        return AdviceDraft(steps=[step], rationale="Mock rationale.")
    if task == "extract_claims":
        cite = passage_ids[0] if passage_ids else None
        return ClaimList(claims=[Claim(text="Mock claim.", citation=cite)])
    if task == "check_claim":
        return ClaimVerdict(supported=True, reason="Mock verdict.")
    raise ValueError(f"unknown task: {task}")


# --- Client ------------------------------------------------------------------


@dataclass
class LLMResult:
    task: TaskType
    output: _Out
    usage: TokenUsage
    latency_s: float
    provider: Provider
    model: str


@dataclass
class LLMClient:
    """One role's model. Use get_llm() to build it from settings."""

    role: Role
    provider: Provider
    model: str
    budget: TokenBudget
    chat_model: Any = None  # a LangChain chat model; None in mock mode
    calls: list[LLMResult] = field(default_factory=list)

    def run(
        self, task: TaskType, prompt: Any, *, context: dict[str, Any] | None = None
    ) -> LLMResult:
        """Run one task. `prompt` is a string or a list of (role, text) messages."""
        self._check_role(task)
        self.budget.check(task)
        schema = TASK_SCHEMAS[task]
        start = time.perf_counter()
        if self.provider == "mock":
            output = mock_response(task, context or {})
            usage = TokenUsage(
                input_tokens=estimate_tokens(str(prompt)),
                output_tokens=estimate_tokens(output.model_dump_json()),
                estimated=True,
            )
        else:
            output, usage = self._invoke_chat(task, schema, prompt)
        result = LLMResult(
            task=task,
            output=output,
            usage=usage,
            latency_s=time.perf_counter() - start,
            provider=self.provider,
            model=self.model,
        )
        self.budget.charge(usage)
        self.calls.append(result)
        return result

    def _check_role(self, task: TaskType) -> None:
        if task not in TASK_SCHEMAS:
            raise ValueError(f"unknown task: {task}")
        is_judge_task = task in JUDGE_TASKS
        if is_judge_task and self.role != "judge":
            raise LLMConfigError(f"task '{task}' must run on the judge model")
        if not is_judge_task and self.role != "agent":
            raise LLMConfigError(f"task '{task}' must run on the agent model")

    def _invoke_chat(
        self, task: TaskType, schema: type[_Out], prompt: Any
    ) -> tuple[_Out, TokenUsage]:
        structured = self.chat_model.with_structured_output(
            schema, method="json_schema", include_raw=True
        )
        response = structured.invoke(prompt)
        parsed, error = response.get("parsed"), response.get("parsing_error")
        raw = response.get("raw")
        if error is not None or not isinstance(parsed, schema):
            text = getattr(raw, "content", "")
            raise LLMOutputError(f"task '{task}': output did not match schema: {error or text!r}")
        meta = getattr(raw, "usage_metadata", None) or {}
        if meta:
            usage = TokenUsage(meta.get("input_tokens", 0), meta.get("output_tokens", 0))
        else:  # some local servers omit usage; estimate rather than charge nothing
            usage = TokenUsage(
                input_tokens=estimate_tokens(json.dumps(prompt, default=str)),
                output_tokens=estimate_tokens(parsed.model_dump_json()),
                estimated=True,
            )
        return parsed, usage


def _role_model(settings: Settings, role: Role) -> str | None:
    env = settings.runtime
    if env.provider == "lmstudio":
        return env.lmstudio_agent_model if role == "agent" else env.lmstudio_judge_model
    if env.provider == "ollama":
        return env.ollama_agent_model if role == "agent" else env.ollama_judge_model
    return f"mock-{role}"


def _build_chat_model(settings: Settings, model: str) -> Any:
    env, llm = settings.runtime, settings.llm
    if env.provider == "lmstudio":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=env.lmstudio_base_url,
            api_key="lm-studio",  # dummy; LM Studio ignores it
            model=model,
            temperature=llm.temperature,
            timeout=llm.request_timeout_s,
            max_retries=0,  # retries arrive with Phase 2c
        )
    if env.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(base_url=env.ollama_base_url, model=model, temperature=llm.temperature)
    raise LLMConfigError(f"no chat model for provider '{env.provider}'")


def get_llm(
    role: Role, settings: Settings | None = None, budget: TokenBudget | None = None
) -> LLMClient:
    """Build the client for one role. Without `budget`, a fresh per-run budget is created."""
    settings = settings or get_settings()
    provider = settings.runtime.provider
    model = _role_model(settings, role)
    if not model:
        var = f"{provider.upper()}_{role.upper()}_MODEL"
        raise LLMConfigError(f"set {var} in .env to use provider '{provider}'")
    if provider != "mock" and role == "judge" and model == _role_model(settings, "agent"):
        raise LLMConfigError("the judge model must differ from the agent model")
    budget = budget or TokenBudget(cap=settings.loop_bounds.token_cap_per_run)
    chat = None if provider == "mock" else _build_chat_model(settings, model)
    return LLMClient(role=role, provider=provider, model=model, budget=budget, chat_model=chat)
