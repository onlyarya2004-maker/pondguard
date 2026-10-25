# Viva notes — Phase 2: contracts

Owners: M1 (config, state, LLM layer), M2 (models review). Files:
- `config/*.yaml`
- `src/pondguard/config.py`, `models.py`, `state.py`, `llm.py`
- `tests/test_config.py`, `test_models.py`, `test_state.py`, `test_llm.py`

## What it does

### `config/` and `config.py`: limits and rules as validated data

| File | What it holds |
|---|---|
| `settings.yaml` | Loop bounds (8 supervisor steps, 2 re-queries, 1 critic revision, recursion limit 25, 20,000-token cap), retrieval settings (k=5, grade ≥ 0.6, ≥ 2 relevant passages), temperature 0, and the fault-injection switch |
| `thresholds.yaml` | Water-quality limits. Every value carries a `source`, and all are still `TODO-TEAM` until checked against the KB |
| `action_ladder.yaml` | The meaning of each level L0–L4, the actions allowed for each risk level, each condition's default risk, and the 3-day escalation rule |
| `banned_substances.yaml` | The two substances the spec names, marked `verified: false` until the CAA list is checked |

`config.py` loads all four into typed Pydantic objects, plus `.env` (provider, model keys, tracing, variant).

Rules that are checked at load time:
- L3 and L4 must require human approval.
- Every risk level has at least one allowed action.
- A threshold's min must be below its max.
- Unknown keys are rejected.

### `models.py`: input and output contracts (spec §6.4)

- **Bad sensor values are rejected.** `Reading` rejects physically impossible values, such as dissolved oxygen above 20 mg/L or pH below 4.
- **Unknown fields are an error** in every model, so typos are caught instead of silently dropped.
- **Grounding rules are enforced in the types:**
  - A `CauseHypothesis` must cite at least one passage.
  - A `Diagnosis` has at most 3 hypotheses, sorted by confidence. An abstained diagnosis must ask a follow-up question, and one that doesn't abstain needs at least one hypothesis.
  - A `PondReport` must carry the advisory disclaimer, and its risk level must match the action plan's.

### `state.py`: the LangGraph state (spec §6.5)

This is the run's working memory: the plan, what each specialist found, loop counters and the human-approval fields. `events` uses an append-only reducer (`operator.add`), so each node adds to the step log instead of overwriting it.

### `llm.py`: one interface for every model call

- **Named tasks.** Each call is a task (`plan`, `summarise`, `grade`, `rewrite`, `rank`, `advise`, `extract_claims`, `check_claim`) with its own output schema.
- **Two roles.** `agent` runs the agent tasks and `judge` runs the two judge tasks, and a task is refused on the wrong role. For real models, the judge model must differ from the agent model.
- **Providers:**
  - `mock` (default) returns fixed, schema-valid output.
  - `lmstudio` uses `ChatOpenAI` pointed at `http://localhost:1234/v1`, with `with_structured_output(method="json_schema", include_raw=True)`.
  - `ollama` is an optional fallback.
- **Token accounting.** Real runs use the token counts the model reports; mock runs use an estimate. A per-run `TokenBudget` refuses the next call once the 20,000-token cap is reached, and records that the cap was hit.
- **Not yet built (Phase 2c):** response cache, retry, schema repair and fallback, and model-identity capture. Until then, invalid model output raises `LLMOutputError` and is not charged to the budget.

## Agentic concepts shown

- **Guardrails and input/output validation:** the models and the config validators.
- **Bounded loops and cost control:** the loop bounds and the token cap.
- **Short-term memory:** the graph state with an append-only event log.
- **Structured reasoning steps:** every LLM call is a typed task with a schema.

## Environment issue found in this phase

Windows **Smart App Control** (enforce mode) on the team laptop blocked the compiled module inside `ormsgpack` 1.13.0, which LangGraph needs. The Code Integrity log recorded the block.

- **Fix:** pin `ormsgpack==1.12.2`. That build is allowed, and LangGraph accepts any version from 1.12.0 up.
- **Not changed:** the Smart App Control setting itself.
- **For teammates:** if a future upgrade is blocked, the error says "An Application Control policy has blocked this file".

## Likely viva questions

1. **Why put thresholds and the action ladder in YAML instead of code?**
   Reviewers and teammates can read and change them without touching agent code. The validators still enforce the safety rules (L3/L4 need approval). Each threshold also carries its citation, which `check_thresholds` will return with every flag.
2. **How do you stop the agent from making up a diagnosis?**
   At the type level, a `CauseHypothesis` with no citations can't even be created. In Phase 6, the Diagnostician drops any hypothesis whose citations aren't in the retrieved set, and the Critic checks every citation again.
3. **What happens when a run uses too many tokens?**
   Every call is charged to a per-run `TokenBudget`. Once usage reaches the cap, the next call raises `TokenCapExceeded`. The graph will catch this, stop the run, and record `loop_bound_hit` in the run log.
