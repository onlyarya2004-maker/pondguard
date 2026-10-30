# Viva notes: Review 4 modules

## Human gate (`agents/gate.py`, `memory.py`)
L3/L4 actions call LangGraph `interrupt()`; the state is checkpointed to SQLite (`data/pondguard.db`), so the run survives a process restart and `approve` resumes it with `Command(resume=...)`. Approve keeps the action, edit sets a lower or equal allowed level, reject falls back to the highest level that needs no approval. A bad decision is validated before resuming, otherwise it would be stored on the paused thread.
- *Why SQLite checkpoints?* Durable workflow: a reviewer can answer hours later, from another terminal.
- *What if the reviewer edits to an illegal level?* The edit is refused against the risk's allowed set and the thread stays paused.
- *In evaluation?* The gate auto-approves but logs `gate_would_pause`, so the metric is measured, not assumed.

## Memory (`memory.py`)
Tables `pond_history`, `runs`, `approvals`, `pending_approvals`. Same alert on 3 consecutive days raises the risk one level (`repeats` in `agents/advisor.py`). Memory is off in evaluation so scenarios stay independent.

## Evaluation (`evaluation/metrics.py`, `runner.py`)
40 scenarios, labels written before any run. Rule-based metrics only: risk accuracy, critical recall, top-3, action within one level, abstention, safety, gate correctness. Reported on all 40 and on the untuned slice.
- *Did you tune on the test set?* Three scenarios, logged in `docs/tuning_log.md`; hence the untuned slice.
- *Why is risk accuracy 1.0?* Deterministic risk rules plus clean simulated data. It says little about real ponds.

## Judge (`evaluation/judge.py`)
Different family from the agent (Qwen vs Llama). Claims are extracted deterministically; one yes/no call per (claim, passage); a claim is supported if any of its first three cited passages supports it. Faithfulness was 0.38 (A) and 0.28 (B), far below 0.8, and the inspection showed real unsupported citations. The judge's own accuracy needs the 10 hand labels.

## Experiment (`evaluation/experiment.py`)
Rule fixed in advance: adopt B if top-3 or faithfulness rises >= 5 points at <= 2x tokens. Result: keep A (top-3 fell 13 points). B still wins risk accuracy, action and abstention; the paired lists name every scenario that changed.
- *Is 40 enough?* One scenario is about 3 points, so we read the paired lists, not only averages.
