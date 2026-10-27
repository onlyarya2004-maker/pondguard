# Review 3 demo checklist (Sat 31 Oct)

Demo laptop: the one with LM Studio, the models and the GPU (M1's).

## Before Saturday (team)

- [ ] Real team names in `LICENSE`, `pyproject.toml` and `README.md`; teammates added as GitHub collaborators
- [ ] Verify at least dissolved oxygen and ammonia limits against the CIBA PDFs and put document + page in `config/thresholds.yaml` (otherwise the demo shows `TODO-TEAM` as the limit source)
- [ ] Review the 4 scenario labels in `config/scenario_specs.yaml`
- [ ] Reuse the R2 architecture slide; add one slide for the live demo
- [ ] Phoenix screenshots (project `pondguard`: one run per scenario) saved to `docs/r3/`
- [ ] Screen recording of the four runs saved to `docs/r3/` (backup if the live demo fails)
- [ ] One timed rehearsal; each member explains a module from `docs/viva/r3_modules.md`

## Demo day, 30 minutes before

- [ ] Laptop charged, on mains power, phone hotspot ready
- [ ] `lms server start --port 1234` and `lms load meta-llama-3.1-8b-instruct ... -y`; `lms ps` shows the model
- [ ] Phoenix running (`pondguard phoenix`), http://localhost:6006 opens
- [ ] `.env`: `LLM_PROVIDER=lmstudio`, both model keys set, `TRACING=phoenix`
- [ ] `pondguard simulate` and the knowledge base built (`data/chroma` exists; if not, `pondguard ingest`)
- [ ] `pytest` passes (about 10 seconds, shows tests in mock mode)
- [ ] One dry run of `pondguard run --scenario EHP-03`
- [ ] Mock fallback command ready: `LLM_PROVIDER=mock` after `pondguard ingest --embedder hash` (label it as structure only, never as results)

## Demo script (about 5 minutes)

1. **Architecture and repo** (30 s): R2 slide; show the folder layout and `docs/concept_evidence.md`.
2. **HEALTHY-01** (30 s): `pondguard run --scenario HEALTHY-01`. Point out the plan and "skipping diagnosis" (planning).
3. **EHP-03** (90 s): point out the tool calls, the corrective round (2 of 5 passages relevant), the cited EHP diagnosis, the L2 steps with citations.
4. **AMBIG-01** (60 s): weak round, **re-query**, then abstain with the follow-up question (agentic RAG, abstention).
5. **CRITIC-DEMO-01** (60 s): `--inject-fault uncited_step`. Say it is injected; show the rejection, the send-back, the corrected draft (feedback loop).
6. **Phoenix** (45 s): open one run, walk the spans: node spans, tool spans, retrieval round, LLM spans, model identity on each span.
7. **Tests** (15 s): `python -m pytest`.

## Say honestly

- The critic demo uses fault injection; the rest is unscripted.
- EHP-03 needs no re-query on the real model (first search is already good); the re-query shows on AMBIG-01.
- Evaluation, the approval gate, memory and the UI are Review 4.
