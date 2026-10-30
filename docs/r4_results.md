# Review 4 results: single-pass RAG (A) vs agentic RAG (B)

Every number below comes from files in `results/` (`eval_*`, `judge_*`, `ab_summary.csv`, `ab_decision.md`). Regenerate them with the commands in the README. Nothing here is from mock mode.

**Setup:** `meta-llama-3.1-8b-instruct` (Q4_K_M) as agent, `qwen2.5-7b-instruct` as judge, temperature 0, cache on, one run per variant, 40 golden scenarios (5 per group), labels written before any run (`config/scenario_specs.yaml`). Runtime `llama.cpp ... cuda12-avx2@2.67.0`.

## Headline

| Metric (all 40) | Target | A: single-pass | B: agentic |
|---|---|---|---|
| Risk accuracy | >= 0.80 | 0.925 | **1.000** |
| Critical recall | >= 0.90 | 1.000 | 1.000 |
| Top-3 hit | >= 0.75 | **1.000** | 0.867 |
| Action within one level | >= 0.85 | 0.925 | **1.000** |
| Abstention accuracy | >= 0.80 | 0.875 | **0.950** |
| Citation faithfulness (SLM judge) | >= 0.80 | 0.385 | 0.278 |
| Safety violations | 0 | 0 | 0 |
| Human-gate correct | reported | 0.900 | 0.900 |
| Mean tokens per run | reported | 4,580 | 5,835 (1.27x) |
| Mean latency per run | reported | 6.5 s | 10.1 s |

Untuned slice (the 37 scenarios not used while tuning): the picture is the same (risk 0.946 vs 1.000, top-3 1.000 vs 0.862). See `results/ab_summary.csv`.

## The decision

The pre-agreed rule was: adopt B if top-3 or faithfulness rises by >= 5 points at <= 2x cost. Neither rose (top-3 fell 13.3 points, faithfulness fell 10.6), so **the rule says keep A.** `results/ab_decision.md` has the paired lists.

That is not the whole story, and we report it as it is:
- **B is better at knowing when not to answer.** All five AMBIG scenarios are abstained by B and none by A (abstention recall 1.0 vs 0.0). A over-calls three ambiguous cases as Alert (AMBIG-01, -03, -05); B gets every risk level right.
- **A is better at naming the condition.** B lost the top-3 hit on four scenarios (AHPND-03, WSSV-05, WQ-01, WQ-02). In WQ-01 and WQ-02 B abstained although the label expected a water-quality answer; in AHPND-03 and WSSV-05 the re-query or ranking dropped the right passage.
- B costs 1.27x the tokens and 1.6x the latency.

So the rule's answer (A) and the safety argument (B) point in different directions. The report should state the rule's outcome and discuss this trade-off, not hide either.

## Where the system is weak

1. **Citation faithfulness is far below the 0.8 target in both variants (0.28 to 0.38).** We inspected the low scores: the Advisor's steps restate the ladder text but carry citations to manual passages that do not say it. This is a genuine fault in the step-writing prompt, not only a judge artefact. It was found after the tuning freeze for this run, so it was not fixed here. A fix (cite only what a passage states, or leave the ladder steps uncited) is the first thing to do and the run must then be repeated.
2. **The judge is a small model.** Its agreement with hand labels is not yet measured: the 10-claim sheet (`results/judge_validation/sheet.csv`) still needs the team's labels, then `python -m pondguard.cli judge-validate`.
3. **Human-gate correctness is 0.90 (36/40) for both variants.** It is identical because the gate depends on the risk and action level, which the deterministic Advisor sets. The four mismatches (AHPND-03, WQ-04, WQ-05, WSSV-05) are all the same kind: the label expects L3 but the agent chose L2, so the gate did not pause. These are under-calls, not unsafe over-actions.
4. **The data is simulated and easy to separate.** Risk accuracy of 1.0 is partly because the Advisor's risk rules are deterministic and the simulated anomalies are clean. This says little about real ponds.
5. **Tuning.** Three scenarios were used while tuning (EHP-03, AMBIG-01, HEALTHY-01); the untuned slice excludes them. The Advisor's water-quality rule was designed before any golden-set run (`docs/tuning_log.md`). No setting was changed after the golden-set runs began.
6. **One run per variant.** One scenario is 2.5 points; a 5-point gap is one or two scenarios. The stability check (3 repeats) was not run.
7. **Runtime drift.** Auto-update was meant to be off, but the evaluation ran on llama.cpp 2.67.0, while the smoke test used 2.64.0 (`docs/compute_budget.md`). Both variants ran on the same runtime, so the comparison is fair, but the smoke-test table is from an older runtime.

## Cost of the evaluation

Wall-clock: A 276 s, B 413 s for 40 scenarios each (cache on). Judge: A 221 claims, B 169 claims, one call per (claim, passage), plus two model swaps in LM Studio.

## What the team still has to do

- Review the 40 labels (`config/scenario_specs.yaml`) and sign them off.
- Label the 10 judge claims, run `judge-validate`, and put the agreement in the report.
- Decide whether to fix the step-citation prompt and re-run both variants (everything reported must be on one commit, model and runtime).
- Confirm the Review 4 date with the instructor; write the report, slides and viva notes.
