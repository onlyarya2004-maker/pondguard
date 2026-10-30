# Concept evidence (R3)

One row per agentic concept: where it is, which test shows it, and what to show live. Every statement here comes from a run or a test in this repo.

| Concept (tier) | Code | Test | Show live |
|---|---|---|---|
| Tool / function use (mandatory) | `tools/telemetry_tools.py`, `kb_tools.py`, `action_tools.py`, `guardrail_tools.py` | `tests/test_tools.py` | Phoenix: `get_telemetry`, `detect_anomalies`, `search_disease_kb`, `check_guardrails` tool spans |
| Agentic RAG (mandatory) | `agents/diagnostician.py`; `kb/retriever.py` | `tests/test_diagnostician.py`, `tests/test_kb.py` | `run --scenario AMBIG-01`: round 1 weak, re-query, round 2, abstain + question |
| Planning and reasoning (mandatory) | `agents/supervisor.py`, `graph.py` | `tests/test_graph.py` | `run --scenario HEALTHY-01`: plan, then "skipping diagnosis" |
| Observability / tracing (mandatory) | `tracing.py` | `tests/test_tracing.py` | Phoenix project `pondguard`; `results/runs/<run>.jsonl` |
| Multi-agent (bonus) | Supervisor + 4 specialists in `graph.py` | `tests/test_graph.py` | The `[supervisor] next: ...` lines in any run |
| Reflection / feedback loop | `agents/critic.py`, `tools/guardrail_tools.py` | `test_critic_sends_an_uncited_draft_back_once_then_passes` | `run --scenario CRITIC-DEMO-01 --inject-fault uncited_step` |
| Guardrails, bounded loops (bonus) | `config/settings.yaml`, `llm.py` token cap | step-bound, token-cap and critic-limit tests in `tests/test_graph.py` | Config values; bounded re-queries in AMBIG-01 |
| Human gate, durable workflow (bonus) | `agents/gate.py`, `memory.py` (SqliteSaver), `tools/memory_tools.py` | gate, restart, edit/reject tests in `tests/test_graph.py` | `run --scenario WSSV-01` pauses; `pending`; `approve` resumes in a new process |
| Long-term memory (bonus) | `memory.py` (`pond_history`, `runs`, `approvals`), repeat-alert escalation in `agents/advisor.py` | memory and escalation tests in `tests/test_graph.py` | `data/pondguard.db` rows after a run |
| Evaluation (mandatory) | `evaluation/metrics.py`, `runner.py`; labels in `config/scenario_specs.yaml` | `tests/test_evaluation.py` | `results/eval_*.csv`, `docs/r4_results.md` |
| LLM-as-judge (bonus) | `evaluation/judge.py` (Qwen2.5 7B, per-claim yes/no) | `tests/test_judge_experiment.py` | `results/judge_*.json` |
| Experiment, A vs B (bonus) | `evaluation/experiment.py`, `plots.py`, Variant A in `agents/diagnostician.py` | `tests/test_judge_experiment.py` | `results/ab_decision.md`, `results/figures/` |
| Dashboard | `app/streamlit_app.py` | `tests/test_app.py` | `streamlit run app/streamlit_app.py` |

## Honest notes for the walkthrough

- The critic demo uses **fault injection**: the Advisor's first draft is deliberately given one uncited step. The CLI and the trace say so (`FAULT INJECTED`). The rejection, the send-back and the fixed draft are the real behaviour.
- EHP-03 does **not** re-query on the real model: the first search is already good, and the answer is correct. The re-query is shown on AMBIG-01, where evidence is genuinely weak.
- AMBIG-01 abstains through a deterministic rule (`vague` in `diagnostician.py`, sign sets in `config/sign_sets.yaml`) because the 8B grader over-scores vague passages. This is logged in `docs/tuning_log.md`.
