# Phase 0 — Team inputs tracker

Things only the team can supply. Update the **Status** column as items land. Nothing here is invented: any value not confirmed by a team member stays `TODO-TEAM`.

Status key: ⬜ not started · 🟨 in progress · ✅ done

| # | Item | Owner | Due | Status | Notes |
|---|---|---|---|---|---|
| 1 | Real names and final role split; put them into the R1 charter and both R1/R2 decks | M1 | Sun 25 Oct | ⬜ | Fill the roles table below |
| 2 | R1/R2 rehearsal: both decks, timed, with likely questions | All | Sun 25 Oct evening | ⬜ | |
| 3 | GitHub repo with all 4 members; branch per member, merge via pull request | M4 | Sun 25 Oct | 🟨 | Public repo `onlyarya2004-maker/pondguard` created 25 Oct. Still to do: add the 3 teammates as collaborators and turn on branch protection for `main` |
| 4 | Knowledge-base PDFs in `data/kb/raw/`, each recorded in `data/kb/sources.csv` | M3 | ≥ 2 by Wed 28 Oct; rest by Sun 1 Nov | 🟨 | 6 downloaded: 4 ICAR-CIBA manuals and 2 FAO papers. Both FAO papers are general background only, and FAO-TP450 is only a 2-page summary. Still to get by hand: 2 WOAH chapters, the CAA banned list, CAA vannamei Dos and Don'ts, MPEDA/NaCSA BMPs. Contents of the downloaded PDFs not reviewed yet |
| 5 | Local SLMs in LM Studio (≤ 8B, GGUF Q4_K_M): `qwen2.5-7b-instruct`, Llama 3.1 8B Instruct, Llama 3.2 3B Instruct | M1 | Mon 26 Oct | ✅ | All 3 downloaded and stored in `AI team project/models/`. Smoke test on 25 Oct picked agent `meta-llama-3.1-8b-instruct` and judge `qwen2.5-7b-instruct` (`docs/compute_budget.md`). Each teammate who runs the system needs the same two GGUF files. Ollama only for teammates without LM Studio |
| 6 | Local tracing: install Arize Phoenix and confirm `phoenix serve` opens on localhost | M4 | Mon 26 Oct | ⬜ | Replaces Langfuse Cloud. Self-hosted Langfuse via Docker only as a fallback (Docker 29.7 installed on M1's laptop) |
| 7 | Water-quality limits in `thresholds.yaml` checked against a KB document, with page citations | M2 + M3 | Sun 1 Nov | ⬜ | Needs item 4 first |
| 8 | Banned-substance names and synonyms from the CAA document | M3 | Sun 1 Nov | ⬜ | CAA PDF must be downloaded by hand (site certificate error) |
| 9 | Signed-off review of the 40 golden-set labels | All | Mon 2 Nov | ⬜ | Needs Phase 3 |
| 10 | 10 blind scenario recipes in `config/blind_specs.yaml` (M4 only; no one else opens it until the tuning freeze) | M4 | Mon 2 Nov | ⬜ | |
| 11 | Team check of the blind labels, only after the `tuning-freeze` tag | All | Tue 3 Nov morning | ⬜ | |

## Team and roles (item 1)

| Member | Real name | Role | Owns |
|---|---|---|---|
| M1 | TODO-TEAM | Team Lead, orchestration | config, models, graph, Supervisor, Advisor, Critic, human gate, memory, CLI, submission |
| M2 | TODO-TEAM | Data and UI | simulator, golden set, telemetry and memory tools, Monitor, Streamlit UI |
| M3 | TODO-TEAM | Knowledge and retrieval | KB, retrieval, KB tools, Diagnostician, prompts, A/B analysis, report §4–6 |
| M4 | TODO-TEAM | Evidence | tracing, CI, blind scenarios, evaluation, judge, experiments, plots, report assembly |

## Knowledge-base coverage (item 4)

Candidates are in [`data/kb/sources.csv`](../data/kb/sources.csv). Coverage of the six conditions so far, based only on document titles and listings (nothing read yet):

| Condition | Candidate source(s) | Gap? |
|---|---|---|
| WSSV | WOAH ch. 2.2.9; CIBA Disease Management (TM 27); FAO TP 446 | URL for the English WOAH chapter still needed |
| EHP | CIBA EHP manual (2022) | — |
| White faeces | CIBA EHP manual (2022) | Confirm it covers white faeces syndrome in enough depth |
| AHPND | WOAH ch. 2.2.1; CIBA Disease Management (TM 27) | URL for the English WOAH chapter still needed |
| Vibriosis | CIBA Disease Management (TM 27); FAO TP 446 | **Weak:** look for a vannamei grow-out vibriosis source |
| Water-quality stress | CIBA inland-saline manual (TM 30) | **Weak:** thresholds need a coastal grow-out water-quality source |
| SOPs / BMPs (`sop_kb`) | CAA vannamei Dos and Don'ts; MPEDA/NaCSA; FAO TP 450 | **Weak:** no MPEDA/NaCSA BMP PDF found yet |

**Download rule:** save each file to `data/kb/raw/<id>.pdf`, using the `id` column of `sources.csv`. Then set its status to `downloaded`, and to `read` once someone has checked what it covers.
