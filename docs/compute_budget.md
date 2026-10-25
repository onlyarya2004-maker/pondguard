# Compute budget (local only)

Owner: M1 (models) · M4 (eval timing). Everything runs locally; there are no quotas. This file tracks **calls per run, seconds per call and total eval time**. Fill the empty cells from real runs only.

## Machine and runtime

Checked on M1's laptop, 25 Oct 2026:

| Item | Value |
|---|---|
| LM Studio app | 0.4.23 |
| `lms` CLI | `%USERPROFILE%\.lmstudio\bin\lms.exe` (commit 07b7252) |
| Selected runtime | `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.64.0` (GGUF). It was 2.33.0 on 25 Oct morning; LM Studio switched it after a service restart that day. Every results file records the runtime used. |
| Runtime auto-updates | **Off** (set in LM Studio by M1 on 25 Oct, after the smoke test), so the runtime stays at 2.64.0 for all reported runs. Changing it means re-running everything reported (rule 8). |
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

## Smoke test (run 25 Oct 2026)

**Source:**
- `results/smoke/smoke_20261025_194720_summary.json`
- `results/smoke/smoke_20261025_194720_calls.csv` (one row per call)

**Setup:** `scripts/smoke_test.py`, 10 calls per task × 8 tasks × 3 models = 240 calls. Settings: cache off, temperature 0, 8K context, full GPU offload, runtime `cuda12-avx2@2.64.0`, LM Studio 0.4.23. Inputs were 10 varied pond cases and 7 excerpts from the ICAR-CIBA PDFs. The prompts are smoke-test prompts, not the final agent prompts (Phase 6).

**An earlier run is superseded.** `smoke_20261025_191102_*` was run before a bug fix: validation errors raised inside the OpenAI client skipped schema repair, so all 10 of Qwen's `rank` calls crashed instead of being repaired. It is kept as a record of the bug and is **not** used below.

How to read the columns:
- **First try / after repair:** valid structured output without, and with, the one schema-repair re-ask.
- **Grounding checks** (agent tasks only):
  - `grade` grades exactly the passage ids given;
  - `rank` and `advise` cite only passage ids that were given.
- **Judge known-answer:** `check_claim` on 5 claims copied from the passage (should be supported) and 5 unrelated claims (should not be).

| Model key | Quant | First try | After repair | Fallbacks | Median s/call | GPU memory | LM Studio RAM increase | Load time |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5-7b-instruct` | Q4_K_M | 87.5% | 100% | 0 | 1.68 s | 4.9 GB (4,924 MiB) | +4.8 GB | 5.1 s |
| `meta-llama-3.1-8b-instruct` | Q4_K_M | **100%** | 100% | 0 | 1.35 s | 5.6 GB (5,712 MiB) | +5.0 GB | 4.1 s |
| `llama-3.2-3b-instruct` | Q4_K_M | 100% | 100% | 0 | **0.95 s** | 3.0 GB (3,056 MiB) | +2.3 GB | 2.1 s |

Grounding and judge checks:

| Model | Agent grounding checks (grade + rank + advise) | `rank` cites given ids | `advise` cites given ids | Judge `check_claim` known-answer | `extract_claims` valid ids |
|---|---|---|---|---|---|
| `qwen2.5-7b-instruct` | 66.7% | 0% | 100% | **90%** (9/10) | 100% |
| `meta-llama-3.1-8b-instruct` | **76.7%** | 40% | 90% | 80% (8/10) | 80% |
| `llama-3.2-3b-instruct` | 63.3% | 40% | 50% | 50% (5/10, chance level) | 100% |

Per task, chosen agent model (`meta-llama-3.1-8b-instruct`):

| Task type | First try valid | After repair | Median s/call |
|---|---|---|---|
| plan | 100% | 100% | 0.80 |
| summarise | 100% | 100% | 0.41 |
| grade | 100% | 100% | 6.58 |
| rewrite | 100% | 100% | 0.59 |
| rank | 100% | 100% | 3.40 |
| advise | 100% | 100% | 3.56 |

Chosen judge model (`qwen2.5-7b-instruct`) on its two tasks:

| Task | Valid output rate | Median s/call |
|---|---|---|
| extract claims (1 call per report) | 100% | 2.18 |
| check claim, yes/no (1 call per claim) | 100% (90% correct on known answers) | 0.76 |

**What the grounding failures were.** When asked to rank causes, all three models often put quoted sentences in `citations` instead of passage ids. For example, Qwen cited "White faecal strands floating on the surface…" rather than `CIBA-EHP-2022:p13`. Qwen also gave `confidence` as a percentage (85) on every first try; the repair fixed the number but not the citations.

This is a prompt and schema problem, not just a model one. The Phase 6 fix:
- Describe `citations` in the schema as passage ids.
- Build the schema for each call so `citations` may only contain the ids actually retrieved (a JSON-schema `enum`), so LM Studio's constrained decoding cannot produce anything else.
- The `rank` step and the Critic still drop any citation outside the retrieved set.

**Agent model: `meta-llama-3.1-8b-instruct`.**
- Only model with valid structured output on the first try for every agent task (100%, no repairs).
- Best grounding on agent tasks (76.7%).
- Second fastest (1.35 s median).
- Fits comfortably in GPU memory (5.6 of 12 GB).

**Judge model: `qwen2.5-7b-instruct`.**
- A different family from the agent (Qwen vs Llama), so no model grades its own output.
- Best known-answer accuracy on `check_claim` (9/10) and 100% valid `extract_claims`.
- Its `rank` weakness does not matter here: the judge never ranks.

**Not chosen: `llama-3.2-3b-instruct`.** Fastest and smallest, but 5/10 on known-answer claim checks (chance level) and only 50% grounded advice.

**Limits of this test:**
- 10 calls per task with smoke-test prompts. It ranks the models on format reliability, speed and basic grounding; it does not measure diagnosis quality.
- The choice is revisited only if the Phase 6 prompts on the named scenarios show a clear problem, and only before the tuning freeze, with a rerun and a note in `docs/tuning_log.md`.

**Judge validation:** agreement with 10 hand-checked cases: ____ / 10 (Phase 10)

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
