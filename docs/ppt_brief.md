# PondGuard: briefing for building the slides

This file holds every fact needed for the presentation. All numbers come from files in `results/` or the repo; nothing is invented. Paste it into Claude chat and ask for the deck. Source files are named next to each fact so a slide can cite them.

**Rules for the deck (please follow):** use only the facts and numbers below; do not add statistics, citations or claims; show the limitations slide; label scripted parts as scripted.

## 1. Identity
- Project: **PondGuard**, an agentic early-warning advisor for shrimp-pond disease.
- Course: Agentic AI mini-project (26CD7PEAAI), B.E. CSE (Data Science), BMSCE. Instructor: Dr. Kalyan Nagaraj.
- Team: Arya Sharma (1BM22CD010, Team Lead; orchestration, config, LLM layer, Supervisor, Advisor, Critic, graph, CLI); Nazeef Ahsan Rafiqui (1BM23CD040; simulator, telemetry and memory tools, Monitor); Yasser Ahmed Mohammed (1BM23CD071; knowledge base, retrieval, Diagnostician, prompts); Ayush Khanuja (1BM23CD075; tracing, evaluation, judge, experiment, plots, report).
- Repo: https://github.com/onlyarya2004-maker/pondguard (public).
- AI assistance: Claude Code was used for coding throughout; the team reviews and must be able to explain every module (viva notes in `docs/viva/`). Say this on a slide.

## 2. Problem
For pond P on day D, combine pond telemetry (DO, pH, temperature, salinity, ammonia, nitrite), the farmer's daily log and published disease manuals into one decision:
- a risk level: Normal, Watch, Alert, Critical;
- the likely cause, with citations to the manuals;
- a graded action L0 to L4 (L3 and L4 need human approval).
Shrimp disease (WSSV, EHP, white faeces, AHPND, vibriosis) and water-quality stress cause heavy losses; early, cited, cautious advice helps a farmer decide when to call a lab. PondGuard is **advisory only**: it never gives drug doses, blocks banned substances, and says to confirm with a qualified aquaculture professional or a lab test.

## 3. Architecture (draw this)
LangGraph StateGraph. A **Supervisor** plans the run (plan-and-execute) and routes four specialists, each returning to it:
- **Monitor**: 4 telemetry tools (get_telemetry, compute_trends, detect_anomalies, check_thresholds), then an LLM summary.
- **Diagnostician**: corrective RAG. build query (signs, likely conditions, water quality) -> retrieve k=5 -> grade each passage 0 to 1 (threshold 0.6, need 2 good) -> if weak, rewrite and re-query (max 2) -> rank hypotheses with schema-constrained citations, or **abstain** and ask one follow-up question.
- **Advisor**: deterministic risk rules + one L0-L4 action within the ladder's allowed set; the LLM writes cited non-chemical steps.
- **Critic**: guardrails (every claim cites a retrieved passage, no banned substance, no dosage, action allowed for the risk, disclaimer present). Sends a failing draft back **once**.
- **Human gate** (Review 4): L3/L4 pause the run via `interrupt()`, state saved in SQLite, resumed by Approve, Edit or Reject.
- **Finish**: builds the report and writes pond memory.
Bounds: max 8 supervisor steps, max 2 re-queries, 1 critic revision, token cap 20,000 per run, recursion limit 25.

## 4. Agentic concepts to evidence (table slide)
| Concept | Where | Proof |
|---|---|---|
| Tool use | `tools/` (telemetry, KB, action ladder, guardrails, memory), all traced | tool spans in Phoenix |
| Agentic RAG | `agents/diagnostician.py` | AMBIG-01: weak grade, 2 re-queries, abstain |
| Planning | `agents/supervisor.py` | HEALTHY-01: plan, then diagnosis skipped |
| Reflection | `agents/critic.py` | CRITIC-DEMO-01: reject, revise, pass |
| Multi-agent | Supervisor + 4 specialists | the event log |
| Human-in-the-loop, durable workflow | `agents/gate.py`, `memory.py` (SqliteSaver) | WSSV-01 pauses, survives restart, resumes |
| Memory | `memory.py`; same alert 3 days running raises risk one level | tests |
| Observability | `tracing.py`: JSONL always, Arize Phoenix UI | trace screenshots |
| Evaluation + judge + A/B experiment | `evaluation/` | `results/` |
| Guardrails, bounded loops | `config/`, token cap | tests |

## 5. Runs fully locally (a strength)
No cloud APIs. Agent model `meta-llama-3.1-8b-instruct` (Q4_K_M) and judge model `qwen2.5-7b-instruct` (different family so the judge does not grade its own work), both run in LM Studio. Embeddings `BAAI/bge-small-en-v1.5` (CPU), vector store ChromaDB, tracing Arize Phoenix on localhost. A mock mode runs the whole graph offline with no model (for tests and a demo fallback; its outputs are placeholders, never reported as results).

## 6. Knowledge base
Real manuals, PDFs kept out of Git (copyright), titles and links in `data/kb/sources.csv`: ICAR-CIBA manuals on EHP, on disease management of brackishwater farming, on inland saline shrimp culture, on molecular diagnosis; FAO Fisheries Technical Papers 446 and 450 (hatchery practice, health and biosecurity; background). Chunks ~2,000 characters with 320 overlap, metadata id/source/page. A caveat: the water-quality limits in `config/thresholds.yaml` are indicative and still marked `TODO-TEAM` (not yet verified against a manual).

## 7. Data
Simulated telemetry and farmer logs (72 hours, fixed seeds): baseline cycles + noise, with injected events (DO crash, ammonia and nitrite build-up, temperature and salinity drop, sensor fault), symptoms from each condition's sign set, plus confounders so cases are not trivially separable. **40 golden scenarios**, 5 per group: WSSV, EHP, white faeces, AHPND, vibriosis, water-quality stress, healthy, ambiguous. Labels (risk, acceptable conditions, action level, should-abstain) were committed **before any agent code ran**. Named demo scenarios: HEALTHY-01, EHP-03, AMBIG-01, WSSV-01, plus CRITIC-DEMO-01 (outside the 40).

## 8. Review 3 demo (live, ~5 min), real model
Run with `python -m pondguard.cli run --scenario <ID>`:
1. **HEALTHY-01**: plan, monitor finds no anomaly, diagnosis skipped, risk normal, L0. Shows planning.
2. **EHP-03** (pond P-07, day 42): tools, corrective RAG, cited EHP diagnosis (about 0.90 confidence), risk alert, L2, critic pass. **Honest note: on the real model the first search is already good, so there is no re-query here.** Do not claim one.
3. **AMBIG-01**: weak evidence, two re-queries, then abstains and asks "Do you see white spots on the shell?". This is where the re-query is shown. The abstain is driven by a deterministic vague-evidence rule (config-driven sign sets) because the 8B grader over-scores vague passages; logged in `docs/tuning_log.md`.
4. **CRITIC-DEMO-01 with `--inject-fault uncited_step`**: the first draft is **deliberately** given one uncited step (labelled FAULT INJECTED in code, CLI and trace); the critic rejects it, the Advisor revises once, the critic passes. The rejection and fix are real; the fault is scripted.
Evidence files: `docs/r3/steps_*.md` (step-by-step summaries), JSONL traces, Phoenix exports. The team must still add **Phoenix UI screenshots** and a **recording** (instructions in `docs/r3/README.md`).

## 9. Review 4 demo (~7 min)
WSSV-01: risk critical, action L3, run **pauses** at the human gate; `pending` lists it; `approve` resumes it from the checkpoint even in a new process; result saved to memory. Then AMBIG scenario abstains; then the Streamlit dashboard (Results, Run, Approvals, About).

## 10. Results (Review 4). Source: `results/ab_summary.csv`, `ab_decision.md`, `docs/r4_results.md`
Setup: 40 golden scenarios, temperature 0, one run per variant, same model for both. **Variant A** = single-pass RAG (one retrieval, no grading, no re-query, no abstain). **Variant B** = agentic RAG.

| Metric (all 40) | Target | A | B |
|---|---|---|---|
| Risk accuracy | >= 0.80 | 0.925 | 1.000 |
| Critical recall | >= 0.90 | 1.000 | 1.000 |
| Top-3 hit | >= 0.75 | 1.000 | 0.867 |
| Action within one level | >= 0.85 | 0.925 | 1.000 |
| Abstention accuracy | >= 0.80 | 0.875 | 0.950 |
| Citation faithfulness (SLM judge) | >= 0.80 | 0.385 | 0.278 |
| Safety violations | 0 | 0 | 0 |
| Human-gate correct | reported | 0.90 | 0.90 |
| Mean tokens per run | reported | 4,580 | 5,835 (1.27x) |
| Mean latency per run | reported | 6.5 s | 10.1 s |
Wall clock for 40 scenarios: A 276 s, B 413 s. Mean LLM calls: A 3.8, B 5.1.

**A/B decision rule (fixed in advance):** adopt B if top-3 or faithfulness rises >= 5 points at <= 2x cost, otherwise keep A. Top-3 fell 13.3 points and faithfulness fell 10.6, so **the rule says keep A.**
Paired comparison: B fixed A on risk for 3 scenarios (AMBIG-01, -03, -05, which A over-called as Alert) and on abstention for 5 (all AMBIG); B broke A on top-3 for 4 (AHPND-03, WQ-01, WQ-02, WSSV-05) and on abstention for 2 (WQ-01, WQ-02). Tell the story honestly: B is safer (right risk level, abstains when evidence is vague) but A names the condition more often; the rule's answer and the safety argument point in different directions.
Untuned slice (37 scenarios, excluding EHP-03, AMBIG-01, HEALTHY-01 which were tuned on): same picture (risk 0.946 vs 1.000, top-3 1.000 vs 0.862).
Figures (ready PNGs in `results/figures/`): `metrics_ab.png` (metric bars A vs B), `cost_quality.png` (cost vs quality), `confusion_A.png`, `confusion_B.png` (risk confusion matrices).

## 11. Limitations (must be on a slide)
- **Citation faithfulness is far below target (0.28 to 0.38 vs 0.8).** Inspected: the Advisor's steps restate ladder text but cite passages that do not say it. A real fault, not fixed in the reported run; fixing it means re-running both variants.
- Judge is a small model; its agreement with hand labels is not yet measured (10-claim sheet to be labelled by the team).
- Data is simulated and fairly separable; risk accuracy of 1.0 is partly because the Advisor's risk rules are deterministic. Not evidence about real ponds.
- Three scenarios were tuned on (reported separately); one run per variant; stability repeats not run. One scenario = 2.5 to 3 points, so a 5-point gap is one or two scenarios.
- Human-gate correctness 0.90: all 4 misses (AHPND-03, WQ-04, WQ-05, WSSV-05) are under-calls (label L3, agent chose L2).
- Thresholds not yet verified against a manual; KB holds the manuals listed above only.
- Advisory only; the 8B model's steps are cited but can be generic.

## 12. Engineering quality
248 automated tests (mock mode) plus CI on every push (ruff + pytest), pinned dependencies, fixed seeds, fresh-clone setup guide (`docs/SETUP.md`), tuning log of every change made against a scenario (`docs/tuning_log.md`), viva notes per module.

## 13. Suggested slide outlines
**Review 3 deck (~8 slides):** 1 Title + team; 2 Problem and why agentic; 3 Architecture diagram; 4 Concept-to-evidence table; 5 Knowledge base and local stack; 6 Live demo (four scenarios, with the honest notes); 7 Tracing and the critic feedback loop (screenshots); 8 Limitations and what's next (Review 4).
**Review 4 deck (~12 slides):** 1 Title; 2 Recap of the system; 3 Human gate and durable workflow (WSSV-01 demo); 4 Memory; 5 Golden set design (40 scenarios, labels before code); 6 Metrics and targets; 7 Results table A vs B; 8 A/B decision rule and paired comparison; 9 Cost vs quality figure; 10 Judge and faithfulness finding; 11 Limitations and honesty (tuned slice, simulated data); 12 Contribution table and AI-assistance statement.

## 14. Likely questions (short answers)
- *Why agentic RAG over single pass?* Better risk and abstention; costs 1.27x tokens; but lost top-3 here, so our own rule keeps A.
- *Did you tune on the test set?* Three scenarios, logged; untuned slice reported.
- *Why two models?* A judge from another family avoids self-agreement; both are small, so the judge is a stated limitation.
- *What stops it giving dangerous advice?* No doses, banned-substance filter, L3/L4 need a human, abstain when unsure, advisory disclaimer.
- *What happens if the model output is invalid?* Schema repair once, then a safe fallback, logged.
