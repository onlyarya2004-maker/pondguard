# Compute budget (local only)

Owner: M1 (models) · M4 (eval timing). Everything runs locally; there are no quotas. This file tracks **calls per run, seconds per call and total eval time**. Fill the empty cells from real runs only.

## Machine and runtime

Checked on M1's laptop, 25 Oct 2026:

| Item | Value |
|---|---|
| LM Studio app | 0.4.23 |
| `lms` CLI | `%USERPROFILE%\.lmstudio\bin\lms.exe` (commit 07b7252) |
| Selected runtime | `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.33.0` (GGUF) |
| Hardware | Intel Core Ultra 9 275HX · 31.4 GB RAM · NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Server | `http://localhost:1234/v1` |
| Tracing | Arize Phoenix on `http://localhost:6006` (not installed yet) · JSONL always on |
| Docker | 29.7.2 installed (only for a self-hosted Langfuse fallback; engine not checked) |

**SLM rule:** every LLM role uses a model with ≤ 8B parameters, 4-bit GGUF (Q4_K_M), run in LM Studio. Windows + NVIDIA, so GGUF, not MLX.

**Headless start** (flags checked against this `lms` version):

```bash
lms server start --port 1234
lms load <model-key> --identifier <model-key> --gpu max -y
lms ps
lms server status
```

`lms load --estimate-only <model-key>` shows the memory a model needs without loading it. Load only one model at a time; the agent pass and the judge pass run separately.

## Candidate SLMs

All three are downloaded and stored in **`AI team project/models/`**, outside the repo. LM Studio's models folder (`downloadsFolder` in `~/.lmstudio/settings.json`) points there; the original settings file is backed up as `settings.json.bak-before-pondguard`.

| Model key (`lms ls`) | File (Q4_K_M GGUF) | Size | Load estimate |
|---|---|---|---|
| `qwen2.5-7b-instruct` | `lmstudio-community/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4.68 GB | |
| `meta-llama-3.1-8b-instruct` | `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.92 GB | |
| `llama-3.2-3b-instruct` | `lmstudio-community/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf` | 2.02 GB | ~1.9 GiB GPU (`--estimate-only`, low confidence) |

These model keys go into `.env` and every results file.

## Smoke test (fill on Tue 27 Oct)

About 10 calls per task type per model. "First try" = valid structured output with no repair; "after repair" counts the one schema-repair retry.

| Model key | Quant | Pass rate (first try) | Pass rate (after repair) | Fallbacks | Median s/call | RAM (GB) | GPU memory (GB) |
|---|---|---|---|---|---|---|---|
| `qwen2.5-7b-instruct` | Q4_K_M | | | | | | |
| Llama 3.1 8B Instruct | Q4_K_M | | | | | | |
| Llama 3.2 3B Instruct | Q4_K_M | | | | | | |

Per task type, for the chosen agent model:

| Task type | First try valid? | After repair? | Median s/call |
|---|---|---|---|
| plan | | | |
| summarise | | | |
| grade | | | |
| rewrite | | | |
| rank | | | |
| advise | | | |

Judge model on its two tasks (about 10 calls each):

| Task | Valid output rate | Median s/call |
|---|---|---|
| extract claims (1 call per report) | | |
| check claim, yes/no (1 call per claim) | | |

**Agent model** (used for development, the demo and all reported runs): ____ · Why: ____

**Judge model** (must differ from the agent, ideally another family): ____ · Why: ____

**Judge validation:** agreement with 10 hand-checked cases: ____ / 10

## Calls per run (measure, then replace the estimates)

| Run type | Estimated calls | Measured calls |
|---|---|---|
| Variant B (agentic RAG): plan 1 · summary 1 · grade ≤3 · rewrite ≤2 · rank 1 · advise 1–2 · LLM critic check 1 | ~8–11 | |
| Variant A (single-pass RAG) | ~4–5 | |
| Judge per report: 1 extract + 1 per claim | ~1 + 6–10 | |

## Total eval time (estimate only)

| Item | Runs | Calls (estimate) |
|---|---|---|
| Variant B, 40 + 10 blind | 50 | ~500 |
| Variant A, 40 + 10 blind | 50 | ~250 |
| Stability check (3 repeats × 10, cache off) | 30 | ~300 |
| Judge, both variants (≈1 extract + ~8 claim checks per report) | 100 reports | ~900 |
| Judge validation (10 hand-checked cases) | 10 | ~90 |
| **Total** | | **~2,000** |

Total time ≈ calls × median seconds per call. For example, at 3 s per call that is about 1.7 hours, and at 8 s per call about 4.5 hours. Replace this with the measured figure before the Tue 3 Nov eval runs. The response cache lets an interrupted run resume without redoing finished calls.

| | Agent pass | Judge pass | Total |
|---|---|---|---|
| Measured median s/call | | | |
| Measured total calls | | | |
| **Measured total eval time** | | | |
