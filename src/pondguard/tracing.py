"""Tracing: every run, node and tool call is recorded twice.

Agentic concept: observability. A reviewer must be able to see the agent plan,
call tools, retrieve, get critiqued and stop. Every step is written to:

1. a JSONL file, results/runs/{run_id}.jsonl, which is ALWAYS written (mock mode,
   CI, Phoenix down), so evaluation never depends on a tracing server; and
2. the local Arize Phoenix UI (http://localhost:6006), when TRACING=phoenix and
   the server answers. Spans are sent with plain OpenTelemetry (OTLP over HTTP);
   PondGuard never imports the Phoenix server package itself. LangChain/LangGraph
   calls are traced too, through the OpenInference LangChain instrumentor.

Tracing must never break a run: every tracing operation is wrapped, and a
failure is logged once and skipped. Errors raised by the traced code itself are
recorded and then re-raised unchanged.

Each span and each JSONL record carries the agent and judge model identity
(model key, quantization, LM Studio version, runtime) for plan rule 8.

Usage:
    with RunTracer("run-1") as tracer:
        with tracer.span("monitor", kind="node"):
            check_thresholds(readings)   # a @traced_tool function: recorded automatically
        state_update = {"events": [tracer.event("monitor", "2 anomalies")]}
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from pondguard.config import Settings, get_settings
from pondguard.identity import model_identity
from pondguard.paths import RESULTS_DIR
from pondguard.state import event as make_event

log = logging.getLogger("pondguard.tracing")

SpanKind = Literal["run", "node", "tool", "llm", "retrieval", "critic", "human"]
_OPENINFERENCE_KIND = {
    "run": "AGENT",
    "node": "CHAIN",
    "tool": "TOOL",
    "llm": "LLM",
    "retrieval": "RETRIEVER",
    "critic": "EVALUATOR",
    "human": "CHAIN",
}
MAX_FIELD_CHARS = 2000
PAUSE_EXCEPTIONS = ("GraphInterrupt", "NodeInterrupt")  # LangGraph interrupt() waiting for a human
PROJECT_NAME = "pondguard"

_current: contextvars.ContextVar[RunTracer | None] = contextvars.ContextVar(
    "pondguard_tracer", default=None
)


# --- Serialisation -----------------------------------------------------------


def to_jsonable(value: Any, _depth: int = 0) -> Any:
    """Make tool inputs/outputs JSON-safe and small: Pydantic -> dict, long text cut."""
    if _depth > 6:
        return "…"
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in list(value.items())[:50]}
    if isinstance(value, list | tuple | set):
        items = [to_jsonable(v, _depth + 1) for v in list(value)[:50]]
        return items + [f"… {len(value) - 50} more"] if len(value) > 50 else items
    if isinstance(value, str):
        return value if len(value) <= MAX_FIELD_CHARS else value[:MAX_FIELD_CHARS] + "…"
    if value is None or isinstance(value, bool | int | float):
        return value
    return to_jsonable(str(value), _depth + 1)


def _json(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, default=str)


# --- Phoenix / OpenTelemetry -------------------------------------------------

_provider: Any = None
_provider_endpoint: str | None = None


def phoenix_is_up(endpoint: str, timeout_s: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout_s) as resp:
            return resp.status < 500
    except Exception:
        return False


def setup_phoenix(endpoint: str) -> Any:
    """Create (once per process) an OpenTelemetry provider that exports to Phoenix.

    Returns the TracerProvider, or None if Phoenix is not reachable or setup
    fails. Also instruments LangChain/LangGraph so model calls appear as spans.
    """
    global _provider, _provider_endpoint
    if _provider is not None and _provider_endpoint == endpoint:
        return _provider
    if not phoenix_is_up(endpoint):
        log.warning("Phoenix not reachable at %s; tracing to JSONL only", endpoint)
        return None
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from openinference.semconv.resource import ResourceAttributes
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({ResourceAttributes.PROJECT_NAME: PROJECT_NAME})
        )
        exporter = OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces", timeout=5)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        LangChainInstrumentor().instrument(tracer_provider=provider)
    except Exception as exc:  # tracing must never break a run
        log.warning("Phoenix tracing setup failed (%s); tracing to JSONL only", exc)
        return None
    _provider, _provider_endpoint = provider, endpoint
    return provider


def flush_phoenix(timeout_ms: int = 5000) -> None:
    if _provider is not None:
        try:
            _provider.force_flush(timeout_ms)
        except Exception as exc:
            log.warning("Phoenix flush failed: %s", exc)


# --- Model identity on every span ---------------------------------------------


def run_models(settings: Settings) -> dict[str, dict[str, Any]]:
    """Agent and judge identity for this run (plan rule 8)."""
    env = settings.runtime
    keys = {
        "lmstudio": (env.lmstudio_agent_model, env.lmstudio_judge_model),
        "ollama": (env.ollama_agent_model, env.ollama_judge_model),
    }.get(env.provider, ("mock-agent", "mock-judge"))
    models = {}
    for role, key in zip(("agent", "judge"), keys, strict=True):
        models[role] = model_identity(env.provider, key or "unset").as_dict()
    return models


def _model_attributes(models: dict[str, dict[str, Any]]) -> dict[str, str]:
    attrs = {}
    for role, ident in models.items():
        for name in ("provider", "model_key", "quantization", "lmstudio_version", "runtime"):
            attrs[f"pondguard.{role}.{name}"] = str(ident.get(name))
    return attrs


# --- The run tracer ------------------------------------------------------------


@dataclass
class SpanRecord:
    """What a span sees while it is open; set `.outputs` to record a result."""

    name: str
    kind: SpanKind
    inputs: Any = None
    outputs: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunTracer:
    """Traces one PondGuard run. Use as a context manager."""

    run_id: str
    settings: Settings | None = None
    runs_dir: Path = RESULTS_DIR / "runs"
    tags: dict[str, str] = field(default_factory=dict)  # pond_id, scenario_id, variant
    models: dict[str, dict[str, Any]] | None = None  # filled from settings if None
    phoenix: bool | None = None  # None = follow settings.runtime.tracing
    echo: Callable[[dict], None] | None = None  # called with each event, e.g. to print live
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._warned: set[str] = set()
        self.settings = self.settings or get_settings()
        if self.models is None:
            self.models = self._safe(lambda: run_models(self.settings), default={})
        self.path = self.runs_dir / f"{self.run_id}.jsonl"
        self._otel = None
        use_phoenix = (
            self.settings.runtime.tracing == "phoenix" if self.phoenix is None else self.phoenix
        )
        if use_phoenix:
            provider = self._safe(
                lambda: setup_phoenix(self.settings.runtime.phoenix_collector_endpoint)
            )
            if provider is not None:
                self._otel = provider.get_tracer("pondguard")
        self.phoenix_active = self._otel is not None
        self._run_span_cm = None

    # -- resilience ---------------------------------------------------------

    def _safe(self, fn: Callable[[], Any], default: Any = None) -> Any:
        """Run a tracing operation; on failure log once per error type and carry on."""
        try:
            return fn()
        except Exception as exc:
            key = type(exc).__name__
            if key not in self._warned:
                self._warned.add(key)
                log.warning("tracing problem ignored (%s): %s", key, exc)
            return default

    # -- JSONL ----------------------------------------------------------------

    def write(self, record: dict) -> None:
        """Append one record to results/runs/{run_id}.jsonl. Never raises."""

        def _write() -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "run_id": self.run_id,
                **record,
            }
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(to_jsonable(line), ensure_ascii=False) + "\n")

        self._safe(_write)

    # -- events (human-readable step log for the UI and GraphState.events) ----

    def event(self, node: str, message: str, **data: Any) -> dict:
        entry = make_event(node, message, **data)
        self.events.append(entry)
        self.write({"type": "event", **entry})
        if self.echo is not None:
            self._safe(lambda: self.echo(entry))
        return entry

    # -- spans ------------------------------------------------------------------

    @contextmanager
    def span(
        self, name: str, kind: SpanKind = "node", inputs: Any = None, **attributes: Any
    ) -> Iterator[SpanRecord]:
        """Time a step; record inputs, outputs, errors to JSONL and Phoenix."""
        record = SpanRecord(name=name, kind=kind, inputs=inputs, attributes=attributes)
        otel_cm = self._safe(lambda: self._start_otel(record)) if self._otel else None
        otel_span = self._safe(otel_cm.__enter__) if otel_cm is not None else None
        start = time.perf_counter()
        error: BaseException | None = None
        paused = False
        try:
            yield record
        except BaseException as exc:
            paused = type(exc).__name__ in PAUSE_EXCEPTIONS  # waiting for a person, not a failure
            error = None if paused else exc
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.write(
                {
                    "type": "span",
                    "kind": kind,
                    "name": name,
                    "status": "paused" if paused else ("error" if error else "ok"),
                    "duration_ms": duration_ms,
                    "inputs": record.inputs,
                    "outputs": record.outputs,
                    "error": f"{type(error).__name__}: {error}" if error else None,
                    "attributes": record.attributes,
                    "models": self._compact_models(),
                }
            )
            if otel_span is not None:
                self._safe(lambda: self._end_otel(otel_span, otel_cm, record, error))

    def _compact_models(self) -> dict[str, str]:
        return {
            role: f"{m.get('model_key')}|{m.get('quantization')}|{m.get('runtime')}"
            for role, m in (self.models or {}).items()
        }

    def _start_otel(self, record: SpanRecord) -> Any:
        attrs = {
            "openinference.span.kind": _OPENINFERENCE_KIND[record.kind],
            "pondguard.run_id": self.run_id,
            "pondguard.span_kind": record.kind,
            **{f"pondguard.{k}": str(v) for k, v in self.tags.items()},
            **_model_attributes(self.models or {}),
        }
        if record.inputs is not None:
            attrs["input.value"] = _json(record.inputs)
            attrs["input.mime_type"] = "application/json"
        return self._otel.start_as_current_span(record.name, attributes=attrs)

    def _end_otel(self, span: Any, cm: Any, record: SpanRecord, error: BaseException | None):
        from opentelemetry.trace import Status, StatusCode

        if record.outputs is not None:
            span.set_attribute("output.value", _json(record.outputs))
            span.set_attribute("output.mime_type", "application/json")
        for k, v in record.attributes.items():
            span.set_attribute(
                f"pondguard.{k}", v if isinstance(v, bool | int | float) else _json(v)
            )
        if error is not None:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)[:200]))
        else:
            span.set_status(Status(StatusCode.OK))
        cm.__exit__(None, None, None)

    # -- run lifecycle ------------------------------------------------------------

    def __enter__(self) -> RunTracer:
        self.write(
            {
                "type": "run_start",
                "tags": self.tags,
                "models": self.models,
                "phoenix": self.phoenix_active,
            }
        )
        self._token = _current.set(self)
        self._run_span_cm = self.span("pondguard-run", kind="run", inputs=self.tags)
        self._run_span_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._run_span_cm is not None:
            self._safe(lambda: self._run_span_cm.__exit__(exc_type, exc, tb))
        _current.reset(self._token)
        self.write(
            {"type": "run_end", "status": "error" if exc else "ok", "events": len(self.events)}
        )
        flush_phoenix()


def current_tracer() -> RunTracer | None:
    return _current.get()


# --- @traced_tool ------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def traced_tool(func: F) -> F:
    """Record a tool call (inputs, outputs, duration, errors) in the active run.

    Outside a run (unit tests, notebooks) the tool runs untraced. Errors from the
    tool are recorded and re-raised; tracing failures are swallowed.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tracer = current_tracer()
        if tracer is None:
            return func(*args, **kwargs)
        try:
            bound = signature.bind_partial(*args, **kwargs)
            inputs = dict(bound.arguments)
        except TypeError:
            inputs = {"args": args, "kwargs": kwargs}
        with tracer.span(func.__name__, kind="tool", inputs=inputs) as span:
            result = func(*args, **kwargs)
            span.outputs = result
            return result

    return wrapper  # type: ignore[return-value]
