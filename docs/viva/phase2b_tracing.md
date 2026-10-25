# Viva notes — Phase 2b: tracing

Owner: M4. Files:
- `src/pondguard/tracing.py`
- `src/pondguard/cli.py` (`phoenix`, `trace-demo`)
- `tests/test_tracing.py`

## What it does

Every run is recorded in two places:

1. **JSONL file:** `results/runs/{run_id}.jsonl`. **Always written**: in mock mode, in CI, and when Phoenix is down. Evaluation and the report never depend on a tracing server.
2. **Local Arize Phoenix:** `http://localhost:6006`, when `TRACING=phoenix` and the server answers.

### How spans reach Phoenix

PondGuard sends spans with plain **OpenTelemetry** (OTLP over HTTP) to `localhost:6006/v1/traces`. It does **not** import the Phoenix package in the agent process: `import phoenix.otel` pulls in the whole Phoenix web server, which is slow, and failed in one of my test setups. Phoenix runs separately, through `pondguard phoenix`, with its data in `AI team project/cache/phoenix`.

LangChain/LangGraph calls are traced automatically through the **OpenInference LangChain instrumentor**, so LLM calls appear as spans once the agents exist (Phase 6).

### The building blocks

- **`RunTracer`:** one per run. Writes `run_start` (tags plus full model identity), the spans and events, and `run_end`. It also opens a top-level `pondguard-run` span.
- **`tracer.span(name, kind, inputs)`:** a context manager for graph nodes (and later for retrieval rounds, critic verdicts and human decisions). It records duration, inputs, outputs and status. An error inside it is recorded and re-raised unchanged.
- **`@traced_tool`:** a decorator for every tool (Phase 5). It records the tool's arguments by name, its result, duration and any error. Outside a run, for example in unit tests, the tool simply runs untraced.
- **`tracer.event(node, message)`:** the human-readable step log. It returns the dict that goes into `GraphState.events`, so the UI and the trace show the same steps.
- **Model identity on every span** (rule 8): each Phoenix span carries `pondguard.agent.*` and `pondguard.judge.*` attributes (model key, quantization, LM Studio version, runtime). Each JSONL record carries a compact form, and `run_start` the full one.

### Safety rules

- **Tracing never breaks a run.** Every tracing operation (JSONL write, Phoenix setup, span start and end, identity lookup) is wrapped. A failure is logged once per error type and skipped.
  - Phoenix down → JSONL only.
  - Disk not writable → events stay in memory.
- **Large values are truncated** (2,000 characters, 50 list items), so traces stay small and long KB text isn't copied wholesale into logs.

## Check done for this phase

- `pondguard phoenix` started Phoenix 20.26.0 with no Smart App Control blocks (none in the Code Integrity log).
- `pondguard trace-demo` with `TRACING=phoenix` and the chosen models wrote `results/runs/trace-demo-490bfe29.jsonl`.
- Phoenix's project `pondguard` showed 1 trace with 3 spans: `pondguard-run` (AGENT), `monitor` (CHAIN) and `dummy_check_thresholds` (TOOL). Each span carried agent `meta-llama-3.1-8b-instruct` Q4_K_M, judge `qwen2.5-7b-instruct`, and runtime `cuda12-avx2@2.64.0`.

## Agentic concept shown

**Observability.** Plans, tool calls, retrieval rounds, critic verdicts and stops will all be visible as spans and events. This covers the R3 rubric item "tracing + at least one feedback loop" (2 marks) once the agents use it.

## Likely viva questions

1. **Why keep a JSONL file when you have Phoenix?**
   Evaluation must never depend on a server being up. The JSONL file is written in every mode, it is what the eval scripts read, and it can be checked into results as evidence. Phoenix is for inspecting a run visually.
2. **What happens if Phoenix crashes in the middle of a demo?**
   The OpenTelemetry exporter fails in the background. PondGuard logs one warning and the run continues and finishes. The JSONL trace is complete either way.
3. **Why did the R2 design say Langfuse but you use Phoenix?**
   The project became local-only (no cloud services). Phoenix runs entirely on the laptop, stores its data in the project folder, and accepts standard OpenTelemetry spans. The report states this as a deviation from the R2 design.
