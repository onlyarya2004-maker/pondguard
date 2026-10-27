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

Robustness (Phase 2c), in the order a call meets it:
1. Cache: a stored valid answer for the same provider, model, task, prompt and
   repeat index is returned with its ORIGINAL tokens and latency (rule 9).
2. Retry with backoff on connection errors and timeouts (LM Studio not started,
   model still loading). Other errors are not retried.
3. Schema repair: if the output fails validation, the model is asked once more,
   shown its own output and the validation error.
4. Fallback: if it is still invalid, a conservative default is returned (e.g. all
   passages graded 0, no hypotheses, claim not supported), logged as
   `schema_fallback`, and never cached. Downstream this leads to abstention or
   human review rather than a confident wrong answer.
Every attempt's tokens are charged to the run budget.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pondguard.config import LLMSettings, Provider, Settings, get_settings
from pondguard.llm_cache import CachedCall, LLMCache, cache_key
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


FALLBACK_NOTE = "Model output was invalid after repair; conservative default used."


def fallback_response(task: TaskType, context: dict[str, Any]) -> _Out:
    """The safe default when a model cannot produce valid output.

    Each default pushes the graph toward caution: no evidence counts as relevant,
    no hypothesis is offered (so the Diagnostician abstains), the advice step is
    uncited (so the Critic flags it for a human), and no claim counts as supported.
    """
    passage_ids: list[str] = list(context.get("passage_ids", []))
    if task == "plan":
        return SupervisorPlan(plan=list(DEFAULT_PLAN), reasoning=FALLBACK_NOTE)
    if task == "summarise":
        return TelemetrySummary(summary=f"Summary unavailable. {FALLBACK_NOTE}")
    if task == "grade":
        return PassageGrades(
            grades=[
                PassageGrade(passage_id=p, grade=0.0, reason=FALLBACK_NOTE) for p in passage_ids
            ]
        )
    if task == "rewrite":
        query = str(context.get("query", "")).strip() or "shrimp pond disease signs"
        return RewrittenQuery(query=query)
    if task == "rank":
        return RankedHypotheses(hypotheses=[])
    if task == "advise":
        step = AdviceStep(text="Continue routine monitoring and consult an aquaculture expert.")
        return AdviceDraft(steps=[step], rationale=FALLBACK_NOTE)
    if task == "extract_claims":
        return ClaimList(claims=[])
    if task == "check_claim":
        return ClaimVerdict(supported=False, reason=FALLBACK_NOTE)
    raise ValueError(f"unknown task: {task}")


def _transient_errors() -> tuple[type[BaseException], ...]:
    """Errors worth retrying: the server is down, slow, or still loading a model."""
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import openai

        errors += [openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError]
    except ImportError:
        pass
    try:
        import httpx

        errors += [httpx.ConnectError, httpx.TimeoutException]
    except ImportError:
        pass
    return tuple(errors)


TRANSIENT_ERRORS = _transient_errors()


# --- Client ------------------------------------------------------------------


@dataclass
class LLMResult:
    task: TaskType
    output: _Out
    usage: TokenUsage
    latency_s: float
    provider: Provider
    model: str
    attempts: int = 1  # model calls that returned something (not counting connection retries)
    repaired: bool = False  # valid only after the schema-repair re-ask
    fallback: bool = False  # still invalid after repair; conservative default used
    cache_hit: bool = False  # tokens and latency are the original call's
    error: str | None = None  # last validation error, if any


class _InvalidOutput(Exception):
    def __init__(self, message: str, raw_text: str, usage: TokenUsage) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.usage = usage


def _as_messages(prompt: Any) -> list[Any]:
    return [("human", prompt)] if isinstance(prompt, str) else list(prompt)


@dataclass
class LLMClient:
    """One role's model. Use get_llm() to build it from settings."""

    role: Role
    provider: Provider
    model: str
    budget: TokenBudget
    chat_model: Any = None  # a LangChain chat model; None in mock mode
    settings: LLMSettings | None = None
    cache: LLMCache | None = None  # None = no caching (mock mode, or --no-cache)
    repeat: int = 0  # stability repeats use 1, 2, ... so they never replay repeat 0
    calls: list[LLMResult] = field(default_factory=list)

    def run(
        self,
        task: TaskType,
        prompt: Any,
        *,
        context: dict[str, Any] | None = None,
        schema: type[_Out] | None = None,
    ) -> LLMResult:
        """Run one task. `prompt` is a string or a list of (role, text) messages.

        `schema` replaces the task's default output schema for this call, e.g. a
        model whose `citations` field only allows the passage ids that were retrieved,
        so constrained decoding cannot produce any other id.
        """
        self._check_role(task)
        self.budget.check(task)
        context = context or {}
        if self.provider == "mock":
            result = self._run_mock(task, prompt, context, schema)
        else:
            result = self._run_model(task, prompt, context, schema)
        self.budget.charge(result.usage)
        self.calls.append(result)
        return result

    def _run_mock(
        self, task: TaskType, prompt: Any, context: dict[str, Any], schema: type[_Out] | None
    ) -> LLMResult:
        start = time.perf_counter()
        output = mock_response(task, context)
        if schema is not None:
            output = schema.model_validate(output.model_dump())
        usage = TokenUsage(
            input_tokens=estimate_tokens(str(prompt)),
            output_tokens=estimate_tokens(output.model_dump_json()),
            estimated=True,
        )
        return LLMResult(task, output, usage, time.perf_counter() - start, "mock", self.model)

    def _run_model(
        self, task: TaskType, prompt: Any, context: dict[str, Any], schema: type[_Out] | None
    ) -> LLMResult:
        schema = schema or TASK_SCHEMAS[task]
        settings = self.settings or LLMSettings(temperature=0, request_timeout_s=120)
        key = cache_key(
            provider=self.provider,
            model=self.model,
            task=task,
            prompt=prompt,
            temperature=settings.temperature,
            repeat=self.repeat,
        )
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return LLMResult(
                task=task,
                output=schema.model_validate_json(hit.output_json),
                usage=TokenUsage(hit.input_tokens, hit.output_tokens, hit.estimated),
                latency_s=hit.latency_s,
                provider=self.provider,
                model=self.model,
                attempts=hit.attempts,
                repaired=hit.repaired,
                cache_hit=True,
            )

        messages = _as_messages(prompt)
        start = time.perf_counter()
        spent = TokenUsage()
        error: str | None = None
        for attempt in range(1, settings.schema_repair_attempts + 2):
            try:
                output, usage = self._invoke_with_retry(task, schema, messages, settings)
            except _InvalidOutput as bad:
                spent = _add(spent, bad.usage)
                error = str(bad)
                log.warning("task %s attempt %d invalid: %s", task, attempt, error)
                messages = [
                    *messages,
                    ("ai", bad.raw_text or "(empty)"),
                    (
                        "human",
                        "That reply did not match the required JSON schema. Error: "
                        f"{error}. Reply again with only valid JSON for the same task.",
                    ),
                ]
                continue
            spent = _add(spent, usage)
            result = LLMResult(
                task=task,
                output=output,
                usage=spent,
                latency_s=time.perf_counter() - start,
                provider=self.provider,
                model=self.model,
                attempts=attempt,
                repaired=attempt > 1,
                error=error,
            )
            if self.cache is not None:
                call = CachedCall(
                    output.model_dump_json(),
                    spent.input_tokens,
                    spent.output_tokens,
                    spent.estimated,
                    result.latency_s,
                    attempt,
                    attempt > 1,
                )
                self.cache.put(key, provider=self.provider, model=self.model, task=task, call=call)
            return result

        log.warning("schema_fallback: task %s on %s after %d attempts", task, self.model, attempt)
        return LLMResult(
            task=task,
            output=fallback_response(task, context),
            usage=spent,
            latency_s=time.perf_counter() - start,
            provider=self.provider,
            model=self.model,
            attempts=attempt,
            fallback=True,
            error=error,
        )

    def _invoke_with_retry(
        self, task: TaskType, schema: type[_Out], messages: list[Any], settings: LLMSettings
    ) -> tuple[_Out, TokenUsage]:
        for retry in range(settings.max_retries + 1):
            try:
                return self._invoke_chat(task, schema, messages)
            except TRANSIENT_ERRORS as exc:
                if retry == settings.max_retries:
                    raise
                wait = settings.retry_backoff_s * (2**retry)
                log.warning("task %s: %s; retrying in %.1f s", task, type(exc).__name__, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

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
        try:
            response = structured.invoke(prompt)
        except (ValidationError, OutputParserException) as exc:
            # The OpenAI client validates json_schema output itself and raises before
            # LangChain can return it as a parsing_error; treat it as invalid output.
            usage = TokenUsage(
                input_tokens=estimate_tokens(json.dumps(prompt, default=str)),
                estimated=True,
            )
            message = f"task '{task}': output did not match schema: {exc}"
            raise _InvalidOutput(message[:600], _invalid_input_text(exc), usage) from exc
        parsed, error = response.get("parsed"), response.get("parsing_error")
        raw = response.get("raw")
        text = str(getattr(raw, "content", "") or "")
        meta = getattr(raw, "usage_metadata", None) or {}
        if meta:
            usage = TokenUsage(meta.get("input_tokens", 0), meta.get("output_tokens", 0))
        else:  # some local servers omit usage; estimate rather than charge nothing
            usage = TokenUsage(
                input_tokens=estimate_tokens(json.dumps(prompt, default=str)),
                output_tokens=estimate_tokens(text),
                estimated=True,
            )
        if error is not None or not isinstance(parsed, schema):
            message = f"task '{task}': output did not match schema: {error or 'no parsed output'}"
            raise _InvalidOutput(message[:600], text, usage)
        return parsed, usage


def _invalid_input_text(exc: Exception) -> str:
    """Best effort at the model's rejected output, to show it back during repair."""
    if isinstance(exc, ValidationError):
        values = [str(e.get("input")) for e in exc.errors()[:3] if "input" in e]
        return "; ".join(values)
    return str(getattr(exc, "llm_output", "") or "")


def _add(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return TokenUsage(
        a.input_tokens + b.input_tokens,
        a.output_tokens + b.output_tokens,
        a.estimated or b.estimated,
    )


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
            max_retries=0,  # PondGuard retries itself, with logging and backoff
        )
    if env.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(base_url=env.ollama_base_url, model=model, temperature=llm.temperature)
    raise LLMConfigError(f"no chat model for provider '{env.provider}'")


def get_llm(
    role: Role,
    settings: Settings | None = None,
    budget: TokenBudget | None = None,
    *,
    use_cache: bool | None = None,
    repeat: int = 0,
) -> LLMClient:
    """Build the client for one role.

    Without `budget`, a fresh per-run budget is created. `use_cache=False` is the
    --no-cache switch; `repeat` > 0 gives stability repeats their own cache slots.
    """
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
    cache_on = settings.llm.cache if use_cache is None else use_cache
    cache = LLMCache(settings.runtime.data_dir / "llm_cache.db") if cache_on else None
    return LLMClient(
        role=role,
        provider=provider,
        model=model,
        budget=budget,
        chat_model=chat,
        settings=settings.llm,
        cache=None if provider == "mock" else cache,
        repeat=repeat,
    )
