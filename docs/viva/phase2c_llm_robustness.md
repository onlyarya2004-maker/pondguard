# Viva notes — Phase 2c: LLM robustness and the SLM smoke test

Owner: M1.

Files:
- `src/pondguard/llm.py` (the run flow)
- `src/pondguard/llm_cache.py`
- `src/pondguard/identity.py`
- `scripts/smoke_test.py`
- `tests/test_llm.py`
- `results/smoke/`
- `docs/compute_budget.md`

## What it does

Every real model call goes through four layers, in this order:

1. **Cache** (`data/llm_cache.db`, kept separate from pond memory).
   - **Key:** provider + model + task + full prompt + temperature + repeat index.
   - **On a hit:** the stored output is returned with the original call's tokens and latency, so cost metrics never treat a cached answer as free (rule 9).
   - **What is stored:** only valid outputs; fallbacks are never cached.
   - **Repeats:** stability repeats use `repeat=1, 2, …` (or `use_cache=False`), so they make fresh calls.
2. **Retry with backoff** on connection errors and timeouts (LM Studio starting up, model still loading): up to 3 retries, waiting 1 s, 2 s, then 4 s. Real bugs, such as a `KeyError`, are not retried.
3. **Schema repair.** If the output fails validation, the model is asked once more and shown its own output plus the exact validation error.
4. **Fallback.** If it is still invalid, a conservative default is used and `schema_fallback` is logged. Each default makes the system more careful, not more confident:

| Task | Safe default | Effect downstream |
|---|---|---|
| `grade` | Every passage graded 0 | No evidence counts as relevant |
| `rank` | No hypotheses | The Diagnostician abstains |
| `advise` | One uncited step | The Critic sends it to a human |
| `check_claim` | Not supported | The claim counts against faithfulness |

Every attempt's tokens, including failed ones, are charged to the run's token budget.

**Model identity** (`identity.py`) reads the model key, GGUF file, quantization and parameter count from `lms ls --json`, the selected runtime from `lms runtime ls`, and the LM Studio app version. Unknown values are recorded as `None`, never guessed. One Windows-specific detail: `lms` prints UTF-8, so the output is decoded as UTF-8 to keep the ✓ that marks the selected runtime.

## Smoke test result (25 Oct)

240 real calls: 10 per task × 8 tasks × 3 models.

- **Agent: `meta-llama-3.1-8b-instruct`.** 100% valid on the first try, best grounding (76.7%), 1.35 s per call, 5.6 GB GPU.
- **Judge: `qwen2.5-7b-instruct`.** Another family; 9/10 on known-answer claim checks.
- **Rejected: `llama-3.2-3b-instruct`.** 5/10 on claim checks (chance level).

Full tables are in `docs/compute_budget.md`.

## What the smoke test found (good viva material)

1. **A real bug, found and fixed.** When structured output used `json_schema`, the OpenAI client raised pydantic's `ValidationError` itself, before LangChain could return it as a parsing error. The repair step never ran, and all of Qwen's `rank` calls crashed.
   - Fix: catch the error and route it to repair.
   - A test (`test_validation_error_raised_by_the_client_is_repaired_too`) replays the exact failure.
   - The first run's results are kept but marked as superseded.
2. **Constrained decoding does not enforce number ranges.** Qwen returned `confidence: 85` (a percentage) even though the schema says at most 1. Pydantic validation plus repair caught it on every call.
3. **Small models put quotes where ids belong.** Asked to cite passages, models often wrote the quoted sentence into `citations` instead of the passage id. This shapes the Phase 6 design: the schema for each call will restrict `citations` to the retrieved ids (a JSON-schema `enum`), so the model cannot produce anything else.

## Agentic concepts shown

- **Bounded loops:** retry and repair are both capped.
- **Fail-safe defaults:** fallbacks lead to abstention or human review, never a confident guess.
- **Reproducible evaluation:** the cache and the recorded model identity.
- **Evidence-based model choice:** every number in the choice comes from a saved run file.

## Likely viva questions

1. **Doesn't caching make your cost numbers wrong?**
   No. Each cache entry stores the original call's tokens and latency, and a hit reports those, so a cached run costs the same on paper as a fresh one. The cache only saves wall-clock time on reruns.
2. **What happens if the model returns garbage twice?**
   The task falls back to a conservative default and logs `schema_fallback`. For example, `rank` returns no hypotheses, so the Diagnostician abstains and asks a follow-up question. The fallback rate is reported as a metric in Phase 10.
3. **Why is the judge a different model from the agent?**
   A model checking its own claims tends to agree with itself. A different family (Qwen judging Llama) gives a more independent check. The judge is still a small model, so it's validated on 10 hand-checked cases, and every other metric is rule-based.
