# Tuning log

Every scenario, setting or prompt used to tune the system is recorded here, so the
"untuned" evaluation slice (R4) can be computed honestly. Scenarios listed under
"tuned on" leave the untuned subset.

## Tools (tuned on simulated healthy ponds, not on golden scenarios)

| Date | Change | Evidence |
|---|---|---|
| 27 Oct | Spike rule cutoff set to |z| > 4 instead of the spec's 3 (`SPIKE_Z` in `tools/telemetry_tools.py`) | 300 generated healthy 72 h windows: z > 3 false alarm 37.7%, z > 3.5 5.7%, z > 4 0.3%, z > 4.5 0% |
| 27 Oct | Dawn-oxygen trend counts only if the three minima fall by at least 0.3 mg/L (`MIN_DAWN_DROP`) | A healthy seed fell 0.04 mg/L across three dawns (sensor noise) |
| 27 Oct | Dosage check ignores the word "dissolved" in "dissolved oxygen" | "keep dissolved oxygen above 5 mg/L" was wrongly flagged as a dosage |

## Prompts and agents (real model: meta-llama-3.1-8b-instruct Q4_K_M, cache off, temperature 0)

| Date | Change | Why, and what happened |
|---|---|---|
| 27 Oct | **First run, before any tuning** (query built from anomalies, then symptoms) | EHP-03: no re-query. 3/5 passages graded relevant (all about oxygen/water quality), diagnosis water_quality_stress 0.80 / vibriosis 0.60: **wrong** (label white_faeces or ehp) |
| 27 Oct | **A: first query built around signs and likely conditions, water quality last** (`build_query`; sign-to-condition map from spec §3.5 in `SIGN_TO_CONDITIONS`; non-specific signs such as lethargy add no condition). A general rule, not scenario-specific. | EHP-03 now retrieves EHP/white-feces pages and ranks **ehp 0.80: correct**, with no re-query. CRITIC-DEMO-01: **vibriosis 0.80: correct**. AMBIG-01: a **real** re-query (0/5 relevant, then 4/5 after the rewrite), but then a confident vibriosis/EHP ranking from one non-specific sign. |
| 27 Oct | **B: grader prompt tightened**: high scores only if the passage helps EXPLAIN the signs as a cause; passages matching one general sign or only a word score at most 0.4 | EHP-03 and CRITIC-DEMO-01 stay correct (EHP-03 now just meets the 2-passage bar). AMBIG-01: two real re-queries (0/5, 1/5, 2/5) but still ends in a confident vibriosis ranking. |
| 27 Oct | **C: passages are graded against the original situation, not the rewritten query** | The rewrite adds unreported disease names (yellow head virus, white spot), which made unrelated passages look relevant. After C, AMBIG-01 still ranks vibriosis 0.80 from lethargy alone: the 8B grader gives 1.0 to disease passages that merely list lethargy among many symptoms, despite the prompt rule. **Open problem; see the team report.** |

No score threshold was changed (still 0.6, 2 relevant passages).

Tuned on: **EHP-03, AMBIG-01, CRITIC-DEMO-01** (all run repeatedly against the real model while changing the query builder and the grader prompt). HEALTHY-01 was only run to check that no diagnosis is needed. All four demo scenarios are therefore outside the "untuned" evaluation slice (R4). CRITIC-DEMO-01 is outside the golden set anyway.
