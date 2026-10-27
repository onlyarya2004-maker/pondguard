# Viva notes: Review 3 modules

Each member should be able to explain any section. Short answers; the code has the detail.

## Simulator and scenarios (`simulator/`, `config/scenario_specs.yaml`)

**What:** hourly pond readings with day-night cycles plus noise (seeded), and a `do_crash` injector that lowers the dawn oxygen minimum a little more each night. Four scenarios, each with a label written in the recipe **before any agent code existed**.

- **Q: Why simulated data?** No real sensors or outbreak records are available; the scenarios come from documented disease signs. Results show behaviour on these scenarios only.
- **Q: How do you know the labels weren't fitted to the output?** The labels are in the recipe file and in the commit (`61be2dd`) that precedes every agent commit.
- **Q: What does "reproducible" mean here?** The same seed regenerates identical files (tested).

## Knowledge base (`kb/`)

**What:** pypdf text per page, cleaned (headers, footers, contents pages removed), chunked to about 2,000 characters without crossing a page, tagged by disease keywords, embedded with BGE-small into ChromaDB. Each passage keeps document, title and page, so a citation is `CIBA-EHP-2022:p17:c0`.

- **Q: Why not chunk across pages?** So every citation points to one exact page.
- **Q: Why the `hash` embedder?** Tests and mock mode must run offline without downloading a model. It matches shared words only and is never used for results.

## Tools (`tools/`)

Eight plain Python functions, none calls an LLM except `grade_passages`. `detect_anomalies` rules: falling dawn minima (at least 0.3 mg/L), spikes, sudden 24 h drops, steady ammonia/nitrite rises. `check_thresholds` returns each limit's source string.

- **Q: Why a spike cutoff of 4, not 3?** Measured: z>3 raised false alarms on 38% of healthy 72 h windows, z>4 on 0.3% (`docs/tuning_log.md`).
- **Q: Why does the threshold source say TODO-TEAM?** The limits are indicative values from the spec and have not been checked against a manual yet. The system shows that instead of pretending.

## Diagnostician (`agents/diagnostician.py`)

**Loop:** build query, retrieve 5, grade each 0–1, then good (at least 2 graded ≥ 0.6) gives rank; weak with re-queries left gives rewrite and retry (max 2); weak with none left gives abstain. Ranking may only cite the relevant passage ids (the schema's `citations` is an enum of those ids), and anything uncited is dropped.

- **Q: Where is the "agentic" part?** The agent decides what to ask, judges what came back, and changes its query or declines to answer.
- **Q: Why did you change the query builder and grader?** First run on EHP-03 was wrong (oxygen passages ranked water-quality stress). Fixes A, B, C in `docs/tuning_log.md`, each with evidence.
- **Q: Why a deterministic abstain rule?** The 8B grader gave 1.0 to a yellow-head-virus passage for "lethargy" alone. If no sign and no anomaly points to a condition (config-driven), the agent still retrieves and re-queries but abstains.

## Supervisor, Advisor, Critic (`agents/`)

- **Supervisor:** LLM plan, validated (an incomplete plan falls back to the default); skips diagnosis when there are no anomalies or symptoms; one critic revision; step bound 8; stops on the token cap.
- **Advisor:** risk and action levels come from **rules** (testable), the model only writes step wording, every step ends with a citation tag. For L2 the first step is the ladder's own text (send a sample for PCR), so the action never depends on the model.
- **Critic:** deterministic `check_guardrails`: citations on hypotheses and every step, banned substances, dosages, action level allowed for the risk, disclaimer.
- **Q: What stops an infinite loop?** Re-queries ≤ 2, critic revisions ≤ 1, supervisor steps ≤ 8, recursion limit 25, token cap 20,000; all tested.
- **Q: What is fault injection?** `--inject-fault uncited_step` adds one uncited step to the first draft so the critic loop can be shown with a real model. It is always announced in the event log.
- **Q: Why is L3/L4 only "flagged"?** The human approval gate is a Review 4 item; none of the R3 scenarios reach L3/L4.

## Tracing (`tracing.py`)

JSONL is always written; Phoenix gets the same spans over OpenTelemetry when it is running. Tracing never breaks a run. Every span carries the agent and judge model identity.
