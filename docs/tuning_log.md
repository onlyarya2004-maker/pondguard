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

| 27 Oct | **Design rule D: abstain deterministically on vague evidence** (`pointed_conditions` / `vague` in `agents/diagnostician.py`; sign sets in `config/sign_sets.yaml`) | **Justification:** the 8B grader over-scores passages that merely list a general sign among many symptoms (it gave 1.0 to a yellow-head-virus passage for "lethargy" alone, even after B and C). A sign is *specific* if the config maps it to at least one condition, *vague* if it maps to none; a sensor anomaly points to the conditions the config maps its parameter to. If no sign and no anomaly points to a condition, the Diagnostician still retrieves, grades and re-queries, then abstains with a follow-up question (first unreported sign in the config's question bank). It never suppresses a diagnosis: a water-quality anomaly counts as pointing to `water_quality_stress`, and the model can still abstain on its own when evidence is weak (the rule only adds an abstain). No scenario names and no hardcoded conditions in the code. **Result:** AMBIG-01 now shows a real re-query (0/5 relevant), then abstains and asks "Do you see white spots on the shell?"; EHP-03 (ehp 0.80) and CRITIC-DEMO-01 (vibriosis 0.90) unchanged. |

| 27 Oct | **Advisor: the ladder's own wording is always the first step** of a diagnosed plan, cited `[ladder:L<n>]` | The model did not write "send a sample for PCR" for L2 on its own; the action itself should not depend on the model. |
| 27 Oct | **Monitor: plain-language fallback** when the summary is a bare label | The model once returned the placeholder `water_quality_summary` instead of a sentence. |

| 30 Oct | **Design change before any golden-set run: water-quality risk rule** (`decide_risk` in `agents/advisor.py`; `water_quality_symptoms` in `config/sign_sets.yaml`) | Aligned with the spec's own risk definitions (§3.4): watch = one mild anomaly, alert = several co-occurring signals (anomalous parameters plus the signs surfacing at dawn and reduced feeding), critical = a high-severity anomaly. It replaced a severity-only rule. Made **before** any agent ran on the 40 scenarios, and not from their results. |

No score threshold was changed (still 0.6, 2 relevant passages).

**Full-graph real-model results (Llama 3.1 8B, cache off, all four scenarios):** HEALTHY-01: normal/L0, diagnosis skipped. EHP-03: ehp 0.90, alert/L2, no re-query. AMBIG-01: real re-query, abstains, asks "Do you see white spots on the shell?", watch/L0. CRITIC-DEMO-01 with fault injection: draft rejected for the uncited step, sent back once, then passes; vibriosis 0.90, alert/L2. Quality caveat: some of the 8B model's management steps are generic or questionable (for example "increase water exchange"); steps are cited to the passages but are advisory text, not verified practice.

Tuned on: **EHP-03, AMBIG-01, CRITIC-DEMO-01** (all run repeatedly against the real model while changing the query builder and the grader prompt). HEALTHY-01 was only run to check that no diagnosis is needed. All four demo scenarios are therefore outside the "untuned" evaluation slice (R4). CRITIC-DEMO-01 is outside the golden set anyway.
