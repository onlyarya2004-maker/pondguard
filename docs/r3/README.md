# Review 3 evidence

Review 3 rubric: loop running live (3), visible behaviour (3), tracing plus feedback loop (2), code quality (2). All runs below are on the real model (`meta-llama-3.1-8b-instruct`, Q4_K_M, LM Studio), not mock mode. Regenerate everything here with `python scripts\export_r3_evidence.py`.

## What each file shows

| File | Shows | Rubric line |
|---|---|---|
| `steps_HEALTHY-01.md` | Plan, monitor finds no anomaly, **diagnosis skipped**, L0. | Visible behaviour (planning decides what to skip) |
| `steps_EHP-03.md` | Tool calls, corrective RAG, cited EHP diagnosis, L2, critic pass. The first search is already good, so there is **no re-query** here (see below). | Loop running live; visible behaviour |
| `steps_AMBIG-01.md` | Weak grade, **two re-queries**, then abstain with a follow-up question. This is where the re-query is shown. | Loop running live; visible behaviour |
| `steps_CRITIC-DEMO-01.md` | **Injected** uncited step, critic rejects, advisor revises once, critic passes. The fault is deliberate and labelled `FAULT INJECTED`; the rejection and fix are real. | Feedback loop |
| `steps_WSSV-01.md` | Critical risk, L3, the human-approval gate (a Review 4 feature). | Bonus |
| `traces/*.jsonl` | The full JSONL trace behind each summary: every span with inputs, outputs and duration. | Tracing |
| `phoenix/phoenix_*.json` | The same runs as exported from the Phoenix database: span tree, LLM calls, token counts. Present for HEALTHY-01 and EHP-03 (two runs), plus the Phase 2b dummy-tool check. | Tracing |

`traces/` and `phoenix/` are **gitignored** because the traces contain passages from the copyrighted manuals. They exist on this machine only; share them with teammates by file copy, not Git. The `steps_*.md` summaries contain no passage text and are committed.

## Honest notes
- EHP-03 does not re-query on the real model; the answer is right on the first search. Do not claim a re-query there. Show AMBIG-01 for the re-query.
- Phoenix holds no complete trace for AMBIG-01, CRITIC-DEMO-01 or WSSV-01 (those runs were made with Phoenix off, or were interrupted). Their JSONL traces are complete. Re-run them with Phoenix on to get UI screenshots (below).
- Phoenix span data is read straight from its SQLite file, so it shows what the UI would show, but the UI view itself still needs a screenshot.

## What still needs a person

1. **Screenshots from the Phoenix UI** (project `pondguard`) and a **screen recording** of the live demo. Both need the screen. Steps:
   1. Start LM Studio's server and load the model: `lms server start --port 1234`, then `lms load meta-llama-3.1-8b-instruct --identifier meta-llama-3.1-8b-instruct --gpu max -c 8192 -y`.
   2. In `.env` set `TRACING=phoenix`. In a spare terminal from `pondguard/`: `.venv\Scripts\python -m pondguard.cli phoenix`. Open http://localhost:6006 and choose project `pondguard`.
   3. In another terminal run, one at a time: `run --scenario HEALTHY-01`, `run --scenario EHP-03`, `run --scenario AMBIG-01`, `run --scenario CRITIC-DEMO-01 --inject-fault uncited_step` (each as `.venv\Scripts\python -m pondguard.cli ...`, add `--no-cache` for fresh calls).
   4. After each run, in Phoenix open the newest trace and screenshot the span tree expanded (windows key + Shift + S). Save as `docs/r3/screenshots/<scenario>.png`. Needed: the trace list; HEALTHY-01 (no diagnosis spans); EHP-03 (tool spans, `check_guardrails`); AMBIG-01 (several `search_disease_kb` spans); CRITIC-DEMO-01 (two `check_guardrails` spans).
   5. Recording (about 3 minutes, Xbox Game Bar: Windows key + Alt + R, or OBS): terminal running EHP-03 live, then switch to Phoenix and open its trace. Save as `docs/r3/demo_ehp03.mp4`. Screenshots are fine to commit; check the recording is small enough, or keep it out of Git.
2. Before presenting, read `docs/r3_demo_checklist.md`.
